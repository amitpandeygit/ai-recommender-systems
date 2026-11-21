"""How fast each open-weight model cleans listings on two processor cores (section 17.3).

The times stored in llm_cache are those of the runs that filled it, and some of those
runs shared the machine with other jobs. This script measures each model alone, with
nothing else running, on the same 72 requests: the first 6 listings of each language in
the size test's sample (section 17.3) sent alone (Listing 17.3's request), and the first
6 of each language that the fuzzy dictionary could not read completely, sent with the
dictionary's findings (Listing 17.4's request). The 9B model is left out: its file does
not fit in the book's 6 GB job limit beside its working memory (see enrich.py). It loads each model, reads the shared
instruction once, and times every request; the answers go to a temporary cache and are
not used.

Needs the GGUF files (fetch_models.py). Run from the repository root, with nothing else
running (about 30 minutes):
    python chapter_17/speed.py
"""
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ["CH17_CACHE"] = tempfile.mkdtemp(prefix="ch17_speed_")
os.environ["CH17_RUN"] = "1"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402

import llm  # noqa: E402
from enrich import (ENGLISH_NAME, FALLBACK, FALLBACK_SCHEMA, SAMPLE_PER_LANGUAGE, catalogue_text, sample,  # noqa: E402
                    schema, system_prompt, user_message)
from listings import LANGUAGES, _forms, fuzzy_dictionary, load_listings  # noqa: E402

PER_LANGUAGE = 6
LOCAL = [n for n, m in llm.MODELS.items() if m[0] == "llama.cpp" and n != "Qwen3.5-9B"]


def requests(L, ids):
    forms_all = _forms(range(len(LANGUAGES)))
    alone, fallback = [], []
    system, fsystem = system_prompt(), FALLBACK.format(catalogue=catalogue_text())
    for g in range(len(LANGUAGES)):
        mine = [i for i in ids if L[i]["language"] == g]
        alone += [(system, user_message(L[i]), schema(L[i]["kind"])) for i in mine[:PER_LANGUAGE]]
        unread = []
        for i in mine:
            fz = fuzzy_dictionary(L[i], forms_all)
            if fz["unmatched"]:
                unread.append((fsystem, f"{L[i]['kind'].capitalize()} listing. Text: {L[i]['text']} | Dictionary "
                                        f"found: {', '.join(ENGLISH_NAME[t] for t in fz['concepts']) or 'nothing'} | "
                                        f"Unread words: {', '.join(dict.fromkeys(fz['unmatched']))}", FALLBACK_SCHEMA))
            if len(unread) == PER_LANGUAGE:
                break
        fallback += unread
    return alone, fallback


def timed(name, task, reqs, max_tokens):
    out = []
    for s, u, sc in reqs:
        t = time.time()
        r = llm.ask(name, task, s, u, sc, max_tokens)
        out.append((time.time() - t, r["prompt_tokens"], r["completion_tokens"]))
    return out


if __name__ == "__main__":
    L, seed = load_listings()
    ids = sample(L, SAMPLE_PER_LANGUAGE, seed)
    alone, fallback = requests(L, ids)
    print(f"{len(alone)} listings sent alone and {len(fallback)} sent with the dictionary's findings; "
          f"{llm.engine_version(LOCAL[0])}; greedy decoding with the schema's grammar")
    print(f"  {'':<14}{'load s':>8}{'first s':>9}{'alone: median s':>17}{'out tokens':>12}"
          f"{'with dictionary: median s':>27}{'out tokens':>12}{'requests':>10}")
    for name in LOCAL:
        t = time.time()
        llm._local(name)
        load = time.time() - t
        a = timed(name, "speed_alone", alone, 96)
        f = timed(name, "speed_fallback", fallback, 48)
        print(f"  {name:<14}{load:>8.1f}{a[0][0]:>9.1f}{np.median([x[0] for x in a[1:]]):>17.2f}"
              f"{np.mean([x[2] for x in a]):>12.1f}{np.median([x[0] for x in f[1:]]):>27.2f}"
              f"{np.mean([x[2] for x in f]):>12.1f}{len(a):>6}{len(f):>4}", flush=True)
        llm._loaded.clear()
    print("  first: the first request alone, which also reads the shared instruction; median: the other "
          "requests, which restore the saved state and read only their own part (the first request with the "
          "dictionary's findings, which reads that instruction, is left out of its median)")
