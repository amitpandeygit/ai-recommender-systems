"""Messy listings for Tesselo's new uploads, and two dictionary cleaners (section 17.3).

Earlier chapters read every item's title as Chapter 3's clean tokens. A real seller or
creator writes a listing by hand: in one of six languages, with typos, Hindi typed in
Latin letters, English words mixed in, sales phrases, and sometimes the wrong category.
Listing 17.2 writes such a listing for every new upload: the videos that are new at a
cutoff (below Chapter 8's pool threshold) and the 150 launched products of section
10.5. Established items are taken to have been corrected already by Tesselo's
catalogue team, so they keep their clean records, and every earlier result stays the
same.

The faults, each drawn independently with its own rate (FAULTS):

  wrong category  - the seller or creator picks another category of the same kind
  says nothing    - the title is only its generic word ("new", "2026"), as in section 11.5
  word dropped    - one of the title's words is left out
  English mixed   - in a non-English title, one word is written in English instead
  romanised Hindi - a Hindi title is typed in Latin letters (a simple informal romanisation)
  typo            - each word, with its own chance, gets one wrong, missing, doubled or
                    swapped character
  sales phrase    - one or two phrases such as "free delivery" are added

Every fault is recorded, so each cleaner's mistakes can be traced. A cleaner turns a
listing into a record: a category and the catalogue concepts (Chapter 3's tokens) that
the title names. The truth is the item's real category and the tokens still present in
the listing. Two cleaners need no model:

  exact dictionary - words of the declared language matched exactly; the declared
                     category is kept
  fuzzy dictionary - words of any language matched within a small edit distance; the
                     category is the most common category of the matched words, or the
                     declared one if none matches

Run from the repository root to see the listings and the two cleaners (about a minute):
    python chapter_17/listings.py
"""
import os
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "2"
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for _ch in ("chapter_17", "chapter_11", "chapter_10", "chapter_03"):
    sys.path.insert(0, str(HERE.parent / _ch))
from tesselo_shop import CATEGORIES, GENERIC_WORDS, LANGUAGES, VIDEO_CATEGORIES, vocabulary  # noqa: E402
from title_words import LANGUAGE_NAMES, WORDS  # noqa: E402

N_VIDEO_CATEGORIES = len(VIDEO_CATEGORIES)
HINDI, ENGLISH = LANGUAGES.index("hi"), LANGUAGES.index("en")
SEED_OFFSET = 1700
FAULTS = {"wrong category": 0.15, "says nothing": 0.10, "word dropped": 0.15, "English mixed": 0.20,
          "romanised Hindi": 0.50, "typo": 0.12, "sales phrase": 0.30}

# Sales phrases a seller adds, in each of Tesselo's languages (none is a catalogue concept)
SALES = {
    "en": ["sale", "100% original", "free delivery", "hot deal", "limited stock"],
    "hi": ["सेल", "100% ओरिजिनल", "फ्री डिलीवरी", "धमाका ऑफर", "सीमित स्टॉक"],
    "pt": ["promoção", "100% original", "frete grátis", "oferta imperdível", "últimas unidades"],
    "es": ["oferta", "100% original", "envío gratis", "precio especial", "últimas unidades"],
    "id": ["promo", "100% ori", "gratis ongkir", "diskon besar", "stok terbatas"],
    "ar": ["تخفيضات", "أصلي 100%", "توصيل مجاني", "عرض خاص", "كمية محدودة"],
}


# ---------------------------------------------------------------- an informal romanisation of Hindi
_CONS = {"क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "n", "च": "ch", "छ": "chh", "ज": "j", "झ": "jh",
         "ञ": "n", "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n", "त": "t", "थ": "th", "द": "d",
         "ध": "dh", "न": "n", "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m", "य": "y", "र": "r",
         "ल": "l", "व": "v", "श": "sh", "ष": "sh", "स": "s", "ह": "h"}
_NUKTA = {"क": "q", "ख": "kh", "ग": "g", "ज": "z", "ड": "r", "ढ": "rh", "फ": "f"}
_VOWELS = {"अ": "a", "आ": "a", "इ": "i", "ई": "i", "उ": "u", "ऊ": "u", "ए": "e", "ऐ": "ai", "ओ": "o",
           "औ": "au", "ऋ": "ri", "ऑ": "o"}
_SIGNS = {"ा": "a", "ि": "i", "ी": "i", "ु": "u", "ू": "u", "े": "e", "ै": "ai", "ो": "o", "ौ": "au",
          "ृ": "ri", "ॉ": "o"}


# A second way of typing Hindi, for the held-out test of section 17.3: long vowels doubled
# ("aa", "ee", "oo"), "w" for व, "j" and "ph" for the nukta letters ज़ and फ़, and "n" for
# every nasal. It was written from common informal spellings ("chaawal", "teekha"), not
# from the listings, and no dictionary in this chapter contains its forms.
_SECOND = {"cons": {"व": "w"}, "nukta": {"ज": "j", "फ": "ph"},
           "long": {"आ": "aa", "ई": "ee", "ऊ": "oo", "ा": "aa", "ी": "ee", "ू": "oo"}}


def romanise(text, style="first"):
    """Devanagari in Latin letters, the way many people type Hindi on a phone. Each
    consonant carries an "a" unless a vowel sign or a virama follows; that "a" is dropped
    at the end of a word (but kept after a cluster ending in r, y or v, as in "samudra") and between two
    pronounced vowels, a rough form of Hindi's schwa deletion ("चावल" becomes "chaval",
    "फ़ुटबॉल" becomes "futbol"). style="second" types the same words the second way
    (_SECOND): "चावल" becomes "chaawal"."""
    second = style == "second"
    text = unicodedata.normalize("NFD", text)
    words = []
    for word in text.split(" "):
        units, i = [], 0                                    # [consonant, vowel, kind]
        while i < len(word):
            ch = word[i]
            nxt = word[i + 1] if i + 1 < len(word) else ""
            if ch in _CONS:
                base = _SECOND["cons"].get(ch, _CONS[ch]) if second else _CONS[ch]
                if nxt == "\u093c":                         # nukta
                    base = (_SECOND["nukta"] if second and ch in _SECOND["nukta"] else _NUKTA).get(ch, base)
                    i += 1
                    nxt = word[i + 1] if i + 1 < len(word) else ""
                if nxt in _SIGNS:
                    units.append([base, _SECOND["long"].get(nxt, _SIGNS[nxt]) if second else _SIGNS[nxt], "sign"])
                    i += 1
                elif nxt == "\u094d":                       # virama: no vowel
                    units.append([base, "", "virama"])
                    i += 1
                else:
                    units.append([base, "a", "inherent"])
            elif ch in _VOWELS:
                units.append(["", _SECOND["long"].get(ch, _VOWELS[ch]) if second else _VOWELS[ch], "sign"])
            elif ch in ("\u0902", "\u0901"):               # anusvara, candrabindu
                units.append(["n", "", "nasal"])
            elif ch == "\u0903":
                units.append(["h", "", "virama"])
            elif ch != "\u093c":
                units.append([ch, "", "other"])
            i += 1
        spoken = lambda u: u[2] in ("sign", "inherent")    # noqa: E731
        for k in range(len(units) - 1):                     # a nasal before p, b or m is written "m"
            if not second and units[k][2] == "nasal" and units[k + 1][0][:1] in ("p", "b", "m"):
                units[k][0] = "m"
        cluster = len(units) > 1 and units[-2][2] == "virama" and units[-1][0] in ("r", "y", "v")
        if len(units) > 1 and units[-1][2] == "inherent" and not cluster:
            units[-1][1], units[-1][2] = "", "virama"
        for k in range(1, len(units) - 1):
            if units[k][2] == "inherent" and spoken(units[k - 1]) and spoken(units[k + 1]):
                units[k][1], units[k][2] = "", "virama"
        words.append("".join(c + v for c, v, _ in units))
    return " ".join(words)


def typo(word, rng):
    """One wrong, missing, doubled or swapped character in a word of 4 or more characters."""
    chars = list(word)
    if len(chars) < 4:
        return word
    k = int(rng.integers(1, len(chars) - 1))
    op = int(rng.integers(4))
    if op == 0:
        del chars[k]
    elif op == 1:
        chars.insert(k, chars[k])
    elif op == 2:
        chars[k], chars[k + 1] = chars[k + 1], chars[k]
    else:
        pool = "aeiou" if chars[k].isascii() else "".join(sorted(set(word)))
        chars[k] = pool[int(rng.integers(len(pool)))]
    return "".join(chars)


# ---------------------------------------------------------------- Listing 17.2: a messy listing
def write_listing(tokens, language, category, kind_categories, rng):
    """Listing 17.2. A listing as a seller or creator might write it, from the item's clean
    tokens (Chapter 3), its language and its real category. Returns the text, the
    declared category, the tokens still present, and the faults applied."""
    lang = LANGUAGES[language]
    faults = []
    words = list(tokens)
    generic = [t for t in words if t in GENERIC_WORDS]
    if rng.random() < FAULTS["says nothing"]:
        words, faults = generic[:1] or words[:1], ["says nothing"]
    elif rng.random() < FAULTS["word dropped"] and len(words) > 2:
        words.pop(int(rng.integers(len(words))))
        faults.append("word dropped")
    order = rng.permutation(len(words))
    words = [words[k] for k in order]
    surface = [WORDS[t][language] for t in words]
    if language != ENGLISH and rng.random() < FAULTS["English mixed"] and len(words) > 1:
        k = int(rng.integers(len(words)))
        surface[k] = WORDS[words[k]][ENGLISH]
        faults.append("English mixed")
    if language == HINDI and rng.random() < FAULTS["romanised Hindi"]:
        surface = [romanise(s) for s in surface]
        faults.append("romanised Hindi")
    typed = []
    for s in surface:
        if rng.random() < FAULTS["typo"]:
            parts = s.split(" ")
            j = int(rng.integers(len(parts)))
            parts[j] = typo(parts[j], rng)
            s = " ".join(parts)
            if "typo" not in faults:
                faults.append("typo")
        typed.append(s)
    if rng.random() < FAULTS["sales phrase"]:
        for p in rng.choice(SALES[lang], int(rng.integers(1, 3)), replace=False):
            typed.insert(0 if rng.random() < 0.5 else len(typed), str(p))
        faults.append("sales phrase")
    declared = category
    if rng.random() < FAULTS["wrong category"]:
        declared = int(rng.choice([c for c in kind_categories if c != category]))
        faults.append("wrong category")
    return " ".join(typed), declared, sorted(set(words)), faults


def new_uploads(W, cutoff_day):
    """The items whose listings are messy at a cutoff: the videos below Chapter 8's pool
    threshold then, and the 150 launched products."""
    from cold_items import launches
    return np.flatnonzero((W["video"] & ~W["snaps"][cutoff_day]["ok"]) | launches(W))


def listings(W):
    """Every new upload's listing (videos new at day 16 or 23, and the launched products),
    from its own generator (seed cfg.seed + 1700), in item order. Returns a dict by item id."""
    items = W["items"]
    rng = np.random.default_rng(W["cfg"].seed + SEED_OFFSET)
    ids = np.union1d(new_uploads(W, 16), new_uploads(W, 23))
    vid_cats = list(range(N_VIDEO_CATEGORIES))
    prod_cats = list(range(N_VIDEO_CATEGORIES, len(CATEGORIES)))
    out = {}
    for i in ids:
        c = int(items["category"][i])
        text, declared, present, faults = write_listing([str(t) for t in items["title"][i]], int(items["language"][i]),
                                                        c, prod_cats if items["is_product"][i] else vid_cats, rng)
        out[int(i)] = {"text": text, "declared": declared, "language": int(items["language"][i]),
                       "kind": "product" if items["is_product"][i] else "video", "category": c,
                       "present": present, "faults": faults}
    return out


FILE = HERE / "listings.json"
HELDOUT_FILE = HERE / "listings_heldout.json"


def heldout_listings(W):
    """The held-out test of section 17.3: every listing written again with the same
    generator, Hindi typed in Latin letters the second way (romanise(style="second"));
    the Hindi listings typed in Latin letters are kept. Their faults are drawn as before,
    but a typo's draws depend on a word's length, so a few can differ from listings()."""
    global romanise
    first = romanise
    romanise = lambda s: first(s, style="second")           # noqa: E731
    try:
        L = listings(W)
    finally:
        romanise = first
    return {i: v for i, v in L.items() if "romanised Hindi" in v["faults"]}


def save_heldout(W):
    import json
    H = heldout_listings(W)
    HELDOUT_FILE.write_text(json.dumps({"listings": {str(i): v for i, v in H.items()}}, ensure_ascii=False,
                                       indent=0), encoding="utf-8")
    return H


def load_heldout():
    import json
    return {int(i): v for i, v in json.loads(HELDOUT_FILE.read_text(encoding="utf-8"))["listings"].items()}


def save_listings(W):
    """Write every listing to listings.json, so that the model runs need not load the world."""
    import json
    L = listings(W)
    FILE.write_text(json.dumps({"seed": int(W["cfg"].seed), "listings": {str(i): v for i, v in L.items()}},
                               ensure_ascii=False, indent=0), encoding="utf-8")
    return L


def load_listings():
    """The listings and the world's seed, from listings.json."""
    import json
    d = json.loads(FILE.read_text(encoding="utf-8"))
    return {int(i): v for i, v in d["listings"].items()}, d["seed"]


# ---------------------------------------------------------------- cleaners without a model
def _norm(text):
    text = unicodedata.normalize("NFC", text.lower())
    return re.sub(r"[^\w\sऀ-ॿ؀-ۿ-]", " ", text)


def _forms(languages):
    """Every surface form of every concept in the given languages: {form: [tokens]}."""
    forms = {}
    for t, ws in WORDS.items():
        for lang in languages:
            forms.setdefault(_norm(ws[lang]).strip(), []).append(t)
    return forms


def romanised_forms():
    """The fuzzy dictionary's forms, plus each Hindi word typed in Latin letters the first
    way (romanise), the way the listings type it: a dictionary that knows that spelling."""
    forms = _forms(range(len(LANGUAGES)))
    for t, ws in WORDS.items():
        forms.setdefault(_norm(romanise(ws[HINDI])).strip(), []).append(t)
    return forms


def _category_of(token):
    return None if token in GENERIC_WORDS else CATEGORIES.index(token.rsplit("_", 1)[0].replace("_", " "))


def _resolve(cands, declared):
    """Of the concepts a form can mean, prefer those of the declared category."""
    own = [t for t in cands if _category_of(t) == declared]
    return own or cands


def _edit_distance(a, b, cap):
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        if min(cur) > cap:
            return cap + 1
        prev = cur
    return prev[-1]


def exact_dictionary(listing):
    """The declared category, and the concepts whose word in the declared language appears
    exactly in the text (longest phrases first)."""
    forms = _forms([listing["language"]])
    text = " " + " ".join(_norm(listing["text"]).split()) + " "
    found = []
    for form in sorted(forms, key=len, reverse=True):
        if f" {form} " in text:
            found += _resolve(forms[form], listing["declared"])
            text = text.replace(f" {form} ", " | ")
    return {"category": listing["declared"], "concepts": sorted(set(found))}


def fuzzy_dictionary(listing, forms_all=None):
    """Concepts whose word in any of the six languages is within one edit of a word or
    phrase of the text (two for forms of 8 or more characters; exact for forms of 3 or
    fewer). The category is the most common category of the matched concepts of the
    listing's kind (ties broken by the declared category), or the declared one."""
    forms = forms_all or _forms(range(len(LANGUAGES)))
    words = _norm(listing["text"]).split()
    used = [False] * len(words)
    found = []
    for n in (3, 2, 1):
        for s in range(len(words) - n + 1):
            if any(used[s:s + n]):
                continue
            phrase = " ".join(words[s:s + n])
            best, best_d = None, None
            for form, toks in forms.items():
                if form.count(" ") != n - 1:
                    continue
                cap = 0 if len(form) <= 3 else (2 if len(form) >= 8 else 1)
                d = _edit_distance(phrase, form, cap)
                if d <= cap and (best_d is None or d < best_d):
                    best, best_d = toks, d
            if best is not None:
                found += _resolve(best, listing["declared"])
                used[s:s + n] = [True] * n
    unmatched = [w for w, u in zip(words, used) if not u]
    return {"category": majority_category(found, listing), "concepts": sorted(set(found)), "unmatched": unmatched}


def majority_category(found, listing):
    """The most common category among the concepts of the listing's kind (ties broken by
    the declared category, then by the lower index); the declared one if none."""
    kind = range(N_VIDEO_CATEGORIES) if listing["kind"] == "video" else range(N_VIDEO_CATEGORIES, len(CATEGORIES))
    cats = Counter(c for c in (_category_of(t) for t in found) if c in kind)
    if not cats:
        return listing["declared"]
    top = max(cats.values())
    best = [c for c, k in cats.items() if k == top]
    return listing["declared"] if listing["declared"] in best else min(best)


# ---------------------------------------------------------------- scoring a cleaner
def score(records, truth):
    """Category accuracy, and the precision, recall and F1 of the concepts, summed over
    listings (each concept named counts once). records and truth: {item: ...}."""
    tp = fp = fn = right = 0
    for i, t in truth.items():
        r = records[i]
        right += r["category"] == t["category"]
        got, want = set(r["concepts"]), set(t["present"])
        tp += len(got & want)
        fp += len(got - want)
        fn += len(want - got)
    p = tp / max(tp + fp, 1)
    rc = tp / max(tp + fn, 1)
    return {"category": right / len(truth), "precision": p, "recall": rc, "f1": 2 * p * rc / max(p + rc, 1e-12),
            "n": len(truth), "tp": tp, "fp": fp, "fn": fn}


def as_items(W, records):
    """A copy of the items in which each listed item's title and category come from its
    record (a cleaner's output). Everything else is unchanged."""
    items = dict(W["items"])
    title = list(items["title"])
    category = items["category"].copy()
    for i, r in records.items():
        title[i] = np.array(r["concepts"], dtype=object)
        category[i] = r["category"]
    items["title"] = np.array(title, dtype=object) if isinstance(W["items"]["title"], np.ndarray) else title
    items["category"] = category
    return items


if __name__ == "__main__":
    import time
    from content_data import load
    t0 = time.time()
    W = load()
    L = save_listings(W)
    n_vid = sum(v["kind"] == "video" for v in L.values())
    print(f"{len(L):,} messy listings: {n_vid:,} videos new at day 16 or 23 and {len(L) - n_vid} launched products  "
          f"[{time.time() - t0:.0f} s]")
    counts = Counter(f for v in L.values() for f in v["faults"])
    print("Faults: " + ", ".join(f"{f} {counts[f]:,} ({counts[f] / len(L):.1%})" for f in FAULTS)
          + f"; clean {sum(not v['faults'] for v in L.values()):,}")
    hindi = [v for v in L.values() if v["language"] == HINDI]
    print(f"Hindi listings typed in Latin letters: {sum('romanised Hindi' in v['faults'] for v in hindi):,} of "
          f"{len(hindi):,}")
    print("\nThe first listing of each language:")
    for lang in range(len(LANGUAGES)):
        i, v = next((i, v) for i, v in L.items() if v["language"] == lang)
        clean = " ".join(WORDS[str(t)][lang] for t in W["items"]["title"][i])
        print(f"  item {i} ({LANGUAGE_NAMES[lang]} {v['kind']}, {CATEGORIES[v['category']]}, declared "
              f"{CATEGORIES[v['declared']]}): {v['text']!r}\n      clean title: {clean!r}; faults: "
              f"{', '.join(v['faults']) or 'none'}")
    H = save_heldout(W)
    print(f"\nHeld-out spelling test: {len(H)} Hindi listings typed in Latin letters the second way, for example "
          f"{next(iter(H.values()))['text']!r}")
    forms_all, forms_rom = _forms(range(len(LANGUAGES))), romanised_forms()
    for name, f in (("exact dictionary", exact_dictionary), ("fuzzy dictionary", lambda x: fuzzy_dictionary(x, forms_all)),
                    ("romanised dictionary", lambda x: fuzzy_dictionary(x, forms_rom))):
        s = score({i: f(v) for i, v in L.items()}, L)
        print(f"\n{name}: category {s['category']:.1%}, concepts precision {s['precision']:.3f}, recall "
              f"{s['recall']:.3f}, F1 {s['f1']:.3f}  [{time.time() - t0:.0f} s]")
