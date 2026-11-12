"""Vision-language tags for new videos with no useful title (section 11.5).

Many creators upload a video with a title that says nothing about it: "new", "best",
"2026". Here 30% of the new videos at day 23, drawn at random (seed cfg.seed + 1120),
have such a title: only their generic word, in their own language. A vision-language
model can look at the video and write what it shows. Tesselo asks it for a category
and three topic words from the catalogue's own vocabulary (Listing 11.4); its answer is
checked and anything outside the vocabulary is dropped. No real model runs here: the
answers are simulated (content_data.vlm_tags) at two qualities, "good" (category right
95% of the time, each word right 80%) and "weak" (70% and 40%), and parse_answer is
tested on hand-written answers, well formed and not.

The test is the "New for you" slot of section 10.5: for each test request, the new video
with the highest score among those it may show, the score being the user's language
first, then the cosine between the video's title vector (Qwen3-Embedding) and the
request's profile (Listing 11.2). The untitled videos' text is, in turn:

  creator's title   - the title the creator would have written (a reference)
  generic word only - what Tesselo actually has
  weak tags         - the weak model's three words, written in the video's language
  good tags         - the good model's three words

The result is the chosen video's expected satisfied plays at slot 4, over all requests
and when the choice is limited to the untitled videos, paired against "generic word only".

Needs Python 3.12 or later, PyTorch and text_embeddings.npz. Run from the repository
root (about 1 minute):
    python chapter_11/vlm_tags.py
"""
import os
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "2"
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from content_data import MODELS, category_words, load, text_embeddings, vlm_tags  # noqa: E402
from content_models import profiles  # noqa: E402
from graph_data import environment, paired, phase  # noqa: E402
from cold_items import new_video_choice, slot_plays  # noqa: E402
from tesselo_shop import CATEGORIES, GENERIC_WORDS, VIDEO_CATEGORIES  # noqa: E402
from title_words import WORDS  # noqa: E402

UNTITLED = 0.30

PROMPT = """You are tagging a short video for a video app.
Look at the thumbnail and the frames. Answer with JSON only, in this form:
{"category": "<one of: %s>", "topics": ["<word>", "<word>", "<word>"]}
Choose the three topic words from this list only: %s.
If you cannot tell, use "unknown" for the category and an empty list for the topics."""


def vlm_prompt():
    """Listing 11.4 (first part). The request sent with the video's thumbnail and frames."""
    words = [WORDS[t][0] for c in range(len(VIDEO_CATEGORIES)) for t in category_words(c)]
    return PROMPT % (", ".join(VIDEO_CATEGORIES), ", ".join(words))


def parse_answer(answer):
    """Listing 11.4 (second part). The model's answer as Chapter 3 tokens: the category's
    words that the answer names, at most three; an empty list if the answer is not valid
    JSON, names no known category, or names no word of that category."""
    try:
        a = json.loads(answer)
        c = VIDEO_CATEGORIES.index(a["category"])
    except (ValueError, KeyError, TypeError):
        return []
    english = {WORDS[t][0]: t for t in category_words(c)}
    topics = a.get("topics", [])
    tokens = [english[w] for w in topics if isinstance(w, str) and w in english] if isinstance(topics, list) else []
    return list(dict.fromkeys(tokens))[:3]


if __name__ == "__main__":
    t0 = time.time()
    print(environment())
    W = load()
    items, users = W["items"], W["users"]
    print(f"Listing 11.4's prompt ({len(vlm_prompt()):,} characters) begins: {vlm_prompt()[:70]}...")
    for answer in ('{"category": "cooking", "topics": ["curry", "rice", "spicy"]}',
                   '{"category": "cooking", "topics": ["curry", "guitar", "curry", "soup", "rice"]}',
                   '{"category": "food", "topics": ["curry"]}',
                   'This video shows someone cooking a curry.'):
        print(f"  parse_answer({answer}) -> {parse_answer(answer)}")

    E, idx = text_embeddings(W, MODELS[-1])
    T, lang = idx["titles"], items["language"]
    title = E[T[np.arange(len(T)), lang]]
    tags = {q: vlm_tags(W, q) for q in ("weak", "good")}
    for q, t in tags.items():
        cat = np.mean([str(w[0]).rsplit("_", 1)[0] == CATEGORIES[c].replace(" ", "_")
                       for w, c in zip(t, items["category"])])
        word = np.mean([len(set(w) & {str(x) for x in tt}) / 3 for w, tt in zip(t, items["title"])])
        print(f"Simulated {q} model: category right for {cat:.1%} of items; {word:.1%} of its words are in the "
              f"creator's title")

    S = W["snaps"][23]
    cols = np.flatnonzero(W["video"] & ~S["ok"])
    rng = np.random.default_rng(W["cfg"].seed + 1120)
    untitled = np.zeros(len(T), dtype=bool)
    untitled[rng.choice(cols, int(round(UNTITLED * len(cols))), replace=False)] = True
    generic = np.array([GENERIC_WORDS.index(str(t[3])) for t in items["title"]])   # each title's generic word
    versions = {"creator's title": title.copy(),
                "generic word only": np.where(untitled[:, None], E[idx["generic"][generic, lang]], title),
                "weak tags": np.where(untitled[:, None], E[idx["tags"]["weak"]], title),
                "good tags": np.where(untitled[:, None], E[idx["tags"]["good"]], title)}
    R, H, _, bad = phase(W, 23, S)
    bn = bad[:, cols]
    del bad
    P = profiles(title, H)                                     # history videos are old, with their titles
    same = (users["language"][R["user"]][:, None] == lang[cols][None, :]).astype(np.float32)
    un = untitled[cols]
    print(f"\nNew videos at day 23: {len(cols):,}, of which {un.sum():,} have only a generic word as title; "
          f"{len(R['user']):,} test requests  [{time.time() - t0:.0f} s]")
    print(f"  {'':<22}{'all new videos':>16}{'against generic':>18}{'untitled chosen':>17}"
          f"{'only untitled':>15}{'against generic':>18}")
    res = {}
    for name, X in versions.items():
        s = P @ X[cols].T + 1_000.0 * same
        choice = new_video_choice(s, bn, cols)
        only = new_video_choice(s[:, un], bn[:, un], cols[un])
        res[name] = (slot_plays(W, R, choice), slot_plays(W, R, only), np.isin(choice, cols[un]).mean())
    for name, (allp, onlyp, share) in res.items():
        g_all, g_only = res["generic word only"][:2]
        d1 = "" if name == "generic word only" else "{:+.4f} ({:.4f})".format(*paired(allp, g_all, R["user"])[:2])
        d2 = "" if name == "generic word only" else "{:+.4f} ({:.4f})".format(*paired(onlyp, g_only, R["user"])[:2])
        print(f"  {name:<22}{allp.mean():>16.4f}{d1:>18}{share:>17.1%}{onlyp.mean():>15.4f}{d2:>18}")
    print("  (expected satisfied plays of the chosen new video alone at slot 4; all new videos: the choice among "
          "every new video the request may show; only untitled: the choice limited to the untitled ones; untitled "
          "chosen: share of requests whose choice is an untitled video; against generic: paired difference over the "
          f"same requests, standard error from 1,000 bootstrap draws of whole users)  [{time.time() - t0:.0f} s]")
