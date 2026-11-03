"""Tesselo at one million items (Chapter 6, section 6.7).

Chapter 3's tesselo_shop.py gives every one of Tesselo's 5,000 items a category, a
language and a short title, derived from the hidden taste vectors. Vector search
only becomes interesting at a much larger scale, so this file generates 995,000
more items with the same kinds of attributes, made the same way, and adds them to
the 5,000. Together they form a catalogue of 1,000,000 items.

It uses its own random generator (seed cfg.seed + 600) and changes nothing that
Chapters 2 to 5 use: build() is called unchanged, and the attribute directions of
tesselo_shop.py are recovered by replaying its random generator, not by editing it.
The replay is checked: the recovered directions must give the 5,000 original items
exactly the categories and languages they already have.

The new items have no history. Their only description is their content: title
words, category, language and creator, encoded with the same TF-IDF vocabulary and
weights as the original items. Each item also gets an upload day (0 to 364), used
for the filtered-search example of section 6.7.

Needs Python 3.12 or later. Run from the repository root (under a minute):
    python chapter_06/tesselo_catalogue.py
"""
import os
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "2"              # NumPy's linear algebra: the same thread count on every run
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "chapter_03"))
from tesselo_shop import (CATEGORIES, GENERIC_WORDS, LANGUAGES, VIDEO_CATEGORIES,  # noqa: E402
                          WORDS_PER_CATEGORY, Config, build, vocabulary)

N_TOTAL = 1_000_000


def unit_rows(m):
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def attribute_directions(cfg, items):
    """Replay add_attributes() in tesselo_shop.py to recover the directions it drew:
    language, video category, product category and title-word directions. Every draw
    is repeated in the same order with the same sizes, so the random generator ends
    each step in the same state. Returns the directions, and the languages and
    categories the replay gives the original items, which check_replay() compares."""
    rng = np.random.default_rng(cfg.seed + 300)
    lang_seed = rng.integers(1 << 30)
    v = items["vec"]
    r = np.random.default_rng(lang_seed)                          # items' own language draws
    lang = unit_rows(r.normal(size=(len(LANGUAGES), cfg.dim)))
    language = np.argmax(v @ lang.T, axis=1)
    noisy = r.random(len(v)) < 0.10
    language[noisy] = r.integers(0, len(LANGUAGES), noisy.sum())
    prod = items["is_product"]
    category = np.empty(len(v), dtype=np.int64)
    dirs = {}
    for name, n_dir, m, offset in (("video", len(VIDEO_CATEGORIES), ~prod, 0),
                                   ("product", len(CATEGORIES) - len(VIDEO_CATEGORIES), prod,
                                    len(VIDEO_CATEGORIES))):
        dirs[name] = unit_rows(rng.normal(size=(n_dir, cfg.dim)))
        pick = np.argmax(v[m] @ dirs[name].T, axis=1)
        noisy = rng.random(int(m.sum())) < 0.15
        pick[noisy] = rng.integers(0, n_dir, noisy.sum())
        category[m] = offset + pick
    words = unit_rows(rng.normal(size=(len(CATEGORIES) * WORDS_PER_CATEGORY, cfg.dim)))
    return (lang, dirs["video"], dirs["product"], words), {"language": language, "category": category}


def idf_weights(items):
    """The TF-IDF vocabulary weights of Chapter 3's item_features(), from the original items."""
    raw = raw_counts(items["title"], items["category"], items["language"])
    df = (raw > 0).sum(axis=0)
    return np.log((1 + len(raw)) / (1 + df)) + 1.0


def raw_counts(titles, category, language):
    words = vocabulary()
    col = {w: k for k, w in enumerate(words)}
    f = np.zeros((len(titles), len(words) + len(CATEGORIES) + len(LANGUAGES)), dtype=np.float32)
    for i, title in enumerate(titles):
        for w in title:
            f[i, col[w]] += 1.0
    f[np.arange(len(titles)), len(words) + category] = 1.0
    f[np.arange(len(titles)), len(words) + len(CATEGORIES) + language] = 1.0
    return f


def content_features(titles, category, language, idf):
    f = raw_counts(titles, category, language) * idf.astype(np.float32)
    return f / np.linalg.norm(f, axis=1, keepdims=True)


def new_items(cfg, n, directions, n_creators, seed_offset=600):
    """n new items, made the way tesselo_shop.py makes attributes, in batches."""
    lang_dirs, video_dirs, product_dirs, word_dirs = directions
    rng = np.random.default_rng(cfg.seed + seed_offset)
    words = vocabulary()
    vec = unit_rows(rng.normal(size=(n, cfg.dim))).astype(np.float32)
    is_product = rng.random(n) < cfg.product_share
    language = np.argmax(vec @ lang_dirs.T, axis=1)
    noisy = rng.random(n) < 0.10
    language[noisy] = rng.integers(0, len(LANGUAGES), noisy.sum())
    category = np.empty(n, dtype=np.int64)
    nv = len(VIDEO_CATEGORIES)
    category[~is_product] = np.argmax(vec[~is_product] @ video_dirs.T, axis=1)
    category[is_product] = nv + np.argmax(vec[is_product] @ product_dirs.T, axis=1)
    noisy = rng.random(n) < 0.15
    category[noisy & ~is_product] = rng.integers(0, nv, (noisy & ~is_product).sum())
    category[noisy & is_product] = nv + rng.integers(0, len(CATEGORIES) - nv, (noisy & is_product).sum())
    titles = []
    for s in range(0, n, 100_000):                                # 3 own-category words, by taste
        v, c = vec[s:s + 100_000], category[s:s + 100_000]
        own = c[:, None] * WORDS_PER_CATEGORY + np.arange(WORDS_PER_CATEGORY)[None, :]
        logits = 4.0 * np.einsum("nd,nkd->nk", v, word_dirs[own].astype(np.float32))
        pick = np.argsort(-(logits + rng.gumbel(size=logits.shape)), axis=1)[:, :3]   # without replacement
        generic = rng.integers(0, len(GENERIC_WORDS), len(v))
        extra = rng.random(len(v)) < 0.2
        other = (c + rng.integers(1, len(CATEGORIES), len(v))) % len(CATEGORIES)     # any other category
        other_word = other * WORDS_PER_CATEGORY + rng.integers(0, WORDS_PER_CATEGORY, len(v))
        for r in range(len(v)):
            t = [words[own[r, k]] for k in pick[r]] + [GENERIC_WORDS[generic[r]]]
            if extra[r]:
                t.append(words[other_word[r]])
            titles.append(t)
    return {"vec": vec, "is_product": is_product, "language": language, "category": category,
            "title": titles, "creator": rng.integers(0, n_creators, n),
            "upload_day": rng.integers(0, 365, n)}


def build_catalogue(cfg=Config()):
    """Chapter 2 and 3's world, unchanged, plus the million-item catalogue."""
    users, items, served, events, orders, truth = build(cfg)
    directions, _ = attribute_directions(cfg, items)
    idf = idf_weights(items)
    n_creators = int(items["creator"].max()) + 1
    extra = new_items(cfg, N_TOTAL - len(items["vec"]), directions, n_creators)
    rng = np.random.default_rng(cfg.seed + 601)
    cat = {key: np.concatenate([items[key], extra[key]]) for key in ("vec", "is_product", "language",
                                                                    "category", "creator")}
    cat["title"] = list(items["title"]) + extra["title"]
    cat["upload_day"] = np.concatenate([rng.integers(0, 365, len(items["vec"])), extra["upload_day"]])
    cat["feats"] = content_features(cat["title"], cat["category"], cat["language"], idf)
    cat["n_original"] = len(items["vec"])
    return users, items, events, cat, directions


if __name__ == "__main__":
    import time
    t0 = time.time()
    cfg = Config()
    users, items, events, cat, directions = build_catalogue(cfg)
    _, replayed = attribute_directions(cfg, items)
    print(f"Replay check on the {len(items['vec']):,} original items: the replayed draws give the same language "
          f"to {int(np.sum(replayed['language'] == items['language'])):,} of them and the same category to "
          f"{int(np.sum(replayed['category'] == items['category'])):,}")
    sys.path.insert(0, str(ROOT / "chapter_03"))
    from similarity_recs import item_features
    orig = item_features(items, len(items["vec"]))
    print(f"TF-IDF check: largest difference from Chapter 3's item_features() on the original items: "
          f"{np.abs(cat['feats'][:len(orig)] - orig).max():.2e}")
    n = len(cat["vec"])
    print(f"Catalogue: {n:,} items ({cat['n_original']:,} original); videos {np.mean(~cat['is_product']):.1%}; "
          f"distinct content vectors: {len(np.unique(cat['feats'].round(6), axis=0)):,}")
    print("Items per language:", dict(zip(LANGUAGES, np.bincount(cat["language"]).tolist())))
    print("Example new titles:", [" ".join(cat["title"][i]) for i in range(5_000, 5_003)])
    print(f"[{time.time() - t0:.0f} s]")
