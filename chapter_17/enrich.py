"""Cleaning new listings with a language model (section 17.3).

Every new upload's listing (listings.py, Listing 17.2) is sent to a model with Tesselo's
catalogue in all six languages and a few worked examples, and the model answers with
the listing's category and the catalogue concepts its words name, in a JSON form that
the schema fixes (Listing 17.3). The record is scored against the truth: the item's
real category and the concepts still present in the listing.

  size     - the size test: Qwen3.5 0.8B, 2B and 4B, and the hosted reference, on 600
             listings (100 in each language, drawn at random, seed cfg.seed + 1720); the
             9B alone on the first 5 listings of English, Hindi and Portuguese only (its
             5.68 GB file does not fit in the book's 6 GB job limit beside its working
             memory, so it reads its weights from disk again and again and takes minutes
             per listing). Each model is used two ways: alone, on
             every listing, and after the fuzzy dictionary, on the words it could not read
             (hybrid, Listing 17.4). Each row reports quality, speed and, for the hosted
             model, cost. The two dictionaries are the baselines.
  full     - the dictionary with the chosen model (WORKHORSE) on all 2,369 listings, by
             language and by fault, beside a third dictionary that also knows each Hindi
             word typed in Latin letters the way the listings type it (romanised_forms),
             alone and with the model behind it.
  heldout  - the held-out spelling test: the Hindi listings typed in Latin letters written
             again the second way (listings.romanise, style "second"), which no dictionary
             knows; the dictionaries, and the romanised dictionary with the model behind it
             (and, as a reference, the hosted model).

Needs Python 3.12 or later and llama-cpp-python; reads llm_cache, so it runs without the
models once the cache is filled. From the repository root:
    python chapter_17/enrich.py size
    python chapter_17/enrich.py full
    python chapter_17/enrich.py heldout
"""
import os
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "2"
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from listings import (CATEGORIES, FAULTS, GENERIC_WORDS, LANGUAGES, N_VIDEO_CATEGORIES, _forms, exact_dictionary,  # noqa: E402
                      fuzzy_dictionary, heldout_listings, listings, majority_category, romanised_forms, score)
from title_words import LANGUAGE_NAMES, WORDS  # noqa: E402
import llm  # noqa: E402

TASK = "clean"
SAMPLE_PER_LANGUAGE, NINE_B_PER_LANGUAGE, NINE_B_LANGUAGES = 100, 5, 3
SIZES = ["Qwen3.5-0.8B", "Qwen3.5-2B", "Qwen3.5-4B", "Qwen3.5-9B", "Claude Sonnet 5.5"]
WORKHORSE = "Qwen3.5-4B"
ENGLISH_NAME = {t: ws[0] for t, ws in WORDS.items()}
TOKEN_OF = {ws[0]: t for t, ws in WORDS.items()}


# ---------------------------------------------------------------- Listing 17.3: the request
def catalogue_text():
    """Tesselo's catalogue for the prompt: every concept in English, then its word in the
    other five languages."""
    lines = []
    for c in CATEGORIES + ["generic"]:
        toks = list(GENERIC_WORDS) if c == "generic" else [f"{c.replace(' ', '_')}_{k}" for k in range(10)]
        kind = "generic words" if c == "generic" else ("video" if CATEGORIES.index(c) < N_VIDEO_CATEGORIES
                                                       else "product") + f" category {c}"
        lines.append(f"{kind}:")
        lines += [f"  {WORDS[t][0]} = " + " / ".join(WORDS[t][1:]) for t in toks]
    return "\n".join(lines)


SYSTEM = """You clean listings for Tesselo, a video and shopping app. Sellers and creators write a listing by hand, in English, Hindi, Portuguese, Spanish, Indonesian or Arabic. A listing can have typos, Hindi typed in Latin letters, English words mixed into another language, sales phrases such as "free delivery", and a wrong declared category.

Tesselo's catalogue, with each concept in English = Hindi / Portuguese / Spanish / Indonesian / Arabic:
{catalogue}

Answer with the concepts that the listing's words name, in English, from the catalogue only. A word can be misspelt or written in Latin letters and still name a concept. Do not add a concept that no word names, and leave out sales phrases. Then give the listing's true category: the category that most of its concepts belong to, even if the declared category is different. If no word names a concept of a category, keep the declared category.

Examples:
Video listing. Declared category: cooking. Text: chaval kari tikha free delivery
Answer: {{"concepts": ["rice", "curry", "spicy"], "category": "cooking"}}
Product listing. Declared category: toys. Text: frete grátis frigideira faca chalera fácil
Answer: {{"concepts": ["frying pan", "knife", "kettle", "easy"], "category": "kitchen"}}
Video listing. Declared category: music. Text: 2026 oferta
Answer: {{"concepts": ["2026"], "category": "music"}}
Product listing. Declared category: home. Text: مرآة ستارة blanket توصيل مجاني
Answer: {{"concepts": ["mirror", "curtain", "blanket"], "category": "home"}}"""


def system_prompt():
    return SYSTEM.format(catalogue=catalogue_text())


def user_message(listing):
    return (f"{listing['kind'].capitalize()} listing. Declared category: {CATEGORIES[listing['declared']]}. "
            f"Text: {listing['text']}")


def schema(kind):
    """The answer's form: up to 6 concepts from the catalogue and one category of the
    listing's kind. The model cannot write anything else."""
    cats = CATEGORIES[:N_VIDEO_CATEGORIES] if kind == "video" else CATEGORIES[N_VIDEO_CATEGORIES:]
    return {"type": "object", "properties": {
        "concepts": {"type": "array", "maxItems": 6, "items": {"type": "string", "enum": list(TOKEN_OF)}},
        "category": {"type": "string", "enum": cats}}, "required": ["concepts", "category"]}


def to_record(answer, listing):
    """The answer as a record: Chapter 3's tokens, each once, and the category's index.
    An answer that was cut off keeps the declared category and names nothing."""
    if not answer:
        return {"category": listing["declared"], "concepts": []}
    return {"category": CATEGORIES.index(answer["category"]),
            "concepts": sorted({TOKEN_OF[c] for c in answer["concepts"]})}


def clean(name, L, ids):
    """Each listing's record from a model, and the cached request records."""
    system = system_prompt()
    raw = llm.ask_many(name, TASK, [(system, user_message(L[i]), schema(L[i]["kind"])) for i in ids], max_tokens=96)
    return {i: to_record(r["answer"], L[i]) for i, r in zip(ids, raw)}, raw


FALLBACK = """You help clean listings for Tesselo, a video and shopping app. Sellers and creators write a listing by hand, in English, Hindi, Portuguese, Spanish, Indonesian or Arabic, sometimes with typos, Hindi typed in Latin letters and sales phrases. A dictionary has already found the catalogue concepts it could read. You are given the listing, the concepts the dictionary found, and the words it could not read.

Tesselo's catalogue, with each concept in English = Hindi / Portuguese / Spanish / Indonesian / Arabic:
{catalogue}

Answer with the catalogue concepts, in English, that the unread words name and the dictionary did not find. A word can be misspelt or be Hindi written in Latin letters and still name a concept, and two words can name one concept together. Sales phrases and words outside the catalogue name nothing. If the unread words name no concept, or you are not sure, answer with an empty list.

Examples:
Video listing. Text: chaval curry tikha free delivery | Dictionary found: curry | Unread words: chaval, tikha, free, delivery
Answer: {{"concepts": ["rice", "spicy"]}}
Product listing. Text: frete grátis frigideira chalera | Dictionary found: frying pan | Unread words: frete, grátis, chalera
Answer: {{"concepts": ["kettle"]}}
Video listing. Text: oferta gol futebol | Dictionary found: goal, football | Unread words: oferta
Answer: {{"concepts": []}}"""

FALLBACK_SCHEMA = {"type": "object", "properties": {
    "concepts": {"type": "array", "maxItems": 4, "items": {"type": "string", "enum": list(TOKEN_OF)}}},
    "required": ["concepts"]}


def hybrid(name, L, ids, forms_all=None):
    """Listing 17.4. The fuzzy dictionary first; a listing with words it could not read goes
    to the model, which names only the concepts those words add. Listings the dictionary
    read completely cost no request."""
    forms_all = forms_all or _forms(range(len(LANGUAGES)))
    fz = {i: fuzzy_dictionary(L[i], forms_all) for i in ids}
    system = FALLBACK.format(catalogue=catalogue_text())
    asked = [i for i in ids if fz[i]["unmatched"]]
    reqs = [(system, f"{L[i]['kind'].capitalize()} listing. Text: {L[i]['text']} | Dictionary found: "
                     f"{', '.join(ENGLISH_NAME[t] for t in fz[i]['concepts']) or 'nothing'} | Unread words: "
                     f"{', '.join(dict.fromkeys(fz[i]['unmatched']))}", FALLBACK_SCHEMA) for i in asked]
    raw = llm.ask_many(name, "clean_fallback", reqs, max_tokens=48)
    recs = {i: {"category": fz[i]["category"], "concepts": fz[i]["concepts"]} for i in ids}
    for i, r in zip(asked, raw):
        found = sorted(set(fz[i]["concepts"]) | {TOKEN_OF[c] for c in (r["answer"] or {"concepts": []})["concepts"]})
        recs[i] = {"category": majority_category(found, L[i]), "concepts": found}
    return recs, raw


def sample(L, per_language, seed, seed_offset=1720):
    rng = np.random.default_rng(seed + seed_offset)
    ids = np.array(sorted(L))
    langs = np.array([L[i]["language"] for i in ids])
    return [int(i) for lang in range(len(LANGUAGES))
            for i in sorted(rng.choice(ids[langs == lang], per_language, replace=False))]


def invented(recs, L, ids):
    """Share of 'says nothing' listings for which the record names a concept that the
    listing does not hold: a concept made up from nothing."""
    nothing = [i for i in ids if "says nothing" in L[i]["faults"]]
    return np.mean([bool(set(recs[i]["concepts"]) - set(L[i]["present"])) for i in nothing]), len(nothing)


def report(label, recs, L, ids, extra=""):
    s = score({i: recs[i] for i in ids}, {i: L[i] for i in ids})
    inv = invented(recs, L, ids)[0]
    print(f"  {label:<22}{s['category']:>9.1%}{s['precision']:>11.3f}{s['recall']:>8.3f}{s['f1']:>7.3f}{inv:>10.1%}"
          f"{extra}")
    return s


def by_group(recs, L, ids, label):
    print(f"\n{label}: F1 of the concepts and category accuracy, by language and by fault")
    print(f"  {'':<22}{'listings':>9}{'F1':>7}{'category':>10}")
    for lang in range(len(LANGUAGES)):
        sub = [i for i in ids if L[i]["language"] == lang]
        s = score({i: recs[i] for i in sub}, {i: L[i] for i in sub})
        print(f"  {LANGUAGE_NAMES[lang]:<22}{len(sub):>9,}{s['f1']:>7.3f}{s['category']:>10.1%}")
    for f in ["none"] + list(FAULTS):
        sub = [i for i in ids if (f in L[i]["faults"]) or (f == "none" and not L[i]["faults"])]
        s = score({i: recs[i] for i in sub}, {i: L[i] for i in sub})
        print(f"  {'fault: ' + f:<22}{len(sub):>9,}{s['f1']:>7.3f}{s['category']:>10.1%}")


if __name__ == "__main__":
    from content_data import load
    from llm import engine_version, model_id, usage
    part = sys.argv[1] if len(sys.argv) > 1 else "size"
    t0 = time.time()
    W = load()
    L = listings(W)
    forms_all = _forms(range(len(LANGUAGES)))
    exact = {i: exact_dictionary(v) for i, v in L.items()}
    fuzzy = {i: fuzzy_dictionary(v, forms_all) for i, v in L.items()}
    forms_rom = romanised_forms()
    rom = {i: fuzzy_dictionary(v, forms_rom) for i, v in L.items()}
    header = (f"  {'':<30}{'category':>9}{'precision':>11}{'recall':>8}{'F1':>7}{'invented':>10}{'requests':>10}"
              f"{'s/listing':>11}{'USD/1,000':>11}")

    def row(label, recs, ids, raw=None, name=None):
        extra = ""
        if raw is not None:
            u = usage(raw, name)
            if u["usd"] is not None:                             # the hosted model: its latency and its price
                extra = f"{u['requests']:>10,}{u['seconds'] / len(ids):>11.2f}{1000 * u['usd'] / len(ids):>11.2f}"
            else:                                                # local models: times in speed.py, measured alone
                extra = f"{u['requests']:>10,}"
        report(label, recs, L, ids, extra)
    if part == "size":
        ids = sample(L, SAMPLE_PER_LANGUAGE, W["cfg"].seed)
        small = [i for lang in range(NINE_B_LANGUAGES) for i in [j for j in ids if L[j]["language"] == lang][
            :NINE_B_PER_LANGUAGE]]
        print(f"The size test: {len(ids)} listings, {SAMPLE_PER_LANGUAGE} in each language; the 9B model alone on "
              f"{len(small)}, the first {NINE_B_PER_LANGUAGE} of English, Hindi and Portuguese  "
              f"[{time.time() - t0:.0f} s]")
        print(f"Model alone: the model reads every listing. Dictionary + model: the fuzzy dictionary reads it first, "
              f"and only a listing with words it could not read goes to the model (Listing 17.4).")
        for subset, label in ((ids, "all 600"), (small, f"the {len(small)} the 9B model read")):
            print(f"\nOn {label}: category accuracy; precision, recall and F1 of the concepts; invented: share of "
                  "listings that say nothing for which a concept was made up; requests to the model; for the hosted "
                  "model, seconds per listing and USD per 1,000 listings (local models' times: speed.py)")
            print(header)
            row("exact dictionary", exact, subset)
            row("fuzzy dictionary", fuzzy, subset)
            row("romanised dictionary", rom, subset)
            for name in SIZES:
                these = small if name == "Qwen3.5-9B" else ids
                if not set(subset) <= set(these):
                    continue
                for how, f in (("alone", clean), ("dictionary +", hybrid)):
                    try:
                        recs, raw = f(name, L, these)
                    except SystemExit:
                        continue                                  # not run (the hosted model needs a key)
                    keep = set(subset)
                    sent = these if f is clean else [i for i in these if fuzzy[i]["unmatched"]]
                    raw_sub = [r for i, r in zip(sent, raw) if i in keep]
                    row(f"{name} {how}" if how == "alone" else f"dictionary + {name}", recs, subset, raw_sub, name)
        nine = clean("Qwen3.5-9B", L, small)[1]
        print(f"\nQwen3.5-9B, run with nothing else on the machine: a median of "
              f"{np.median([r['seconds'] for r in nine]):.0f} s per listing, because its 5.68 GB file and its working "
              f"memory do not fit in the 6 GB the job may use")
        print("\nModels: " + "; ".join(model_id(n) for n in SIZES) + f". Engine: {engine_version('Qwen3.5-2B')}, "
              "greedy decoding, the answer constrained by the JSON schema.")
    elif part == "full":
        ids = sorted(L)
        recs, raw = hybrid(WORKHORSE, L, ids, forms_all)
        recs_rom, raw_rom = hybrid(WORKHORSE, L, ids, forms_rom)
        print(f"All {len(ids):,} new listings  [{time.time() - t0:.0f} s]")
        print(header)
        row("exact dictionary", exact, ids)
        row("fuzzy dictionary", fuzzy, ids)
        row("romanised dictionary", rom, ids)
        row(f"dictionary + {WORKHORSE}", recs, ids, raw, WORKHORSE)
        row(f"romanised + {WORKHORSE}", recs_rom, ids, raw_rom, WORKHORSE)
        by_group(fuzzy, L, ids, "fuzzy dictionary")
        by_group(rom, L, ids, "romanised dictionary")
        by_group(recs, L, ids, f"dictionary + {WORKHORSE}")
        by_group(recs_rom, L, ids, f"romanised dictionary + {WORKHORSE}")
        for label, r in ((f"dictionary + {WORKHORSE}", raw), (f"romanised dictionary + {WORKHORSE}", raw_rom)):
            u = usage(r, WORKHORSE)
            print(f"\n{label}: {u['requests']:,} of {len(ids):,} listings sent to the model; {u['input']:,} input and "
                  f"{u['output']:,} output tokens; {llm.failures(r, [FALLBACK_SCHEMA] * len(r))} (times: speed.py, "
                  f"measured with nothing else running)")
    else:
        H = heldout_listings(W)
        ids = sorted(H)
        print(f"The held-out spelling test: the {len(ids)} Hindi listings typed in Latin letters, written again the "
              f"second way (for example {H[ids[0]]['text']!r}); no dictionary knows these spellings  "
              f"[{time.time() - t0:.0f} s]")
        L = H                                                    # row() and report() score against these
        print(header)
        row("exact dictionary", {i: exact_dictionary(H[i]) for i in ids}, ids)
        row("fuzzy dictionary", {i: fuzzy_dictionary(H[i], forms_all) for i in ids}, ids)
        row("romanised dictionary", {i: fuzzy_dictionary(H[i], forms_rom) for i in ids}, ids)
        for name in (WORKHORSE, "Claude Sonnet 5.5"):
            try:
                recs, raw = hybrid(name, H, ids, forms_rom)
            except SystemExit:
                continue
            row(f"romanised + {name}", recs, ids, raw, name)
            print(f"    {name}: {llm.failures(raw, [FALLBACK_SCHEMA] * len(raw))}")
        print("  (Hindi only, so the by-language view is this table; the hosted model is a reference on item data)")
