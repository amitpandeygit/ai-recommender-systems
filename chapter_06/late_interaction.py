"""Listing 6.2 - Late interaction (ColBERT's MaxSim) against one pooled vector (section 6.3).

Tesselo's assistant receives typed requests such as "cooking_3 travel_7": two
title words from two different topics. A single-vector (two-tower) retriever
pools the words of the request into one vector and the words of each title into
one vector, and compares the two. A late-interaction retriever (Khattab and
Zaharia's ColBERT, 2020) keeps one vector per word on both sides and scores

    MaxSim(q, d) = sum over request words w of  max over title words t of  sim(w, t)

so that every request word must find its own best match in the title.

The word vectors here are the hidden title-word directions of Chapter 3's
generator, recovered by tesselo_catalogue.py. They stand in for the token vectors
a trained text encoder would produce (Chapter 11 trains real ones). Only the
title words are used: category and language tokens are left out, so that the two
methods see exactly the same words.

Needs Python 3.12 or later. Run from the repository root (under a minute):
    python chapter_06/late_interaction.py
"""
import os
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "2"              # NumPy's linear algebra: the same thread count on every run
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "chapter_03"))
from tesselo_catalogue import attribute_directions  # noqa: E402
from tesselo_shop import WORDS_PER_CATEGORY, Config, build, vocabulary  # noqa: E402


def maxsim(query_vecs, doc_vecs):
    """ColBERT's late interaction: each query token takes its best match in the document."""
    return (query_vecs @ doc_vecs.T).max(axis=1).sum()


def pooled(vecs):
    """One vector for a whole text: the mean of its token vectors, scaled to length 1."""
    m = vecs.mean(axis=0)
    return m / np.linalg.norm(m)


def worked_example():
    """Section 6.3: two request words, two titles, two dimensions, by hand."""
    q = np.array([[1.0, 0.0], [0.0, 1.0]])                    # "cooking", "travel"
    a = np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]])         # title A: cooking, travel, both
    b = np.array([[0.8, 0.6], [0.6, 0.8], [0.7, 0.7]])         # title B: three in-between words
    for name, d in (("A", a), ("B", b)):
        print(f"Title {name}: MaxSim {maxsim(q, d):.4f}; pooled cosine {pooled(q) @ pooled(d):.4f}")


if __name__ == "__main__":
    worked_example()
    cfg = Config()
    users, items, served, events, orders, truth = build(cfg)
    (_, _, _, word_dirs), _ = attribute_directions(cfg, items)
    words = vocabulary()
    col = {w: k for k, w in enumerate(words)}
    titled = [[col[w] for w in t if col[w] < len(word_dirs)] for t in items["title"]]  # title words only
    docs = [word_dirs[t] for t in titled]
    doc_pooled = np.stack([pooled(d) for d in docs])
    rng = np.random.default_rng(cfg.seed + 620)
    multi = [i for i, t in enumerate(titled) if len({w // WORDS_PER_CATEGORY for w in t}) >= 2]
    n_queries, found = 300, {"MaxSim (one vector per word)": [], "pooled (one vector per text)": []}
    avail = []
    for _ in range(n_queries):                                   # a request = one word from each of
        t = titled[multi[rng.integers(len(multi))]]               # two topics of a real title
        w1 = t[0]
        w2 = next(w for w in t if w // WORDS_PER_CATEGORY != w1 // WORDS_PER_CATEGORY)
        q = word_dirs[[w1, w2]]
        match = np.array([w1 in d and w2 in d for d in titled])   # titles with both requested words
        avail.append(match.sum())
        s_max = np.array([maxsim(q, d) for d in docs])
        s_pool = doc_pooled @ pooled(q)
        for name, s in (("MaxSim (one vector per word)", s_max), ("pooled (one vector per text)", s_pool)):
            top = np.argsort(-s, kind="stable")[:10]
            found[name].append(match[top].sum() / min(10, match.sum()))
    print(f"\n{n_queries} two-topic requests over {len(docs):,} titles; titles holding both requested "
          f"words: median {np.median(avail):.0f}, at most {max(avail)}")
    print("Share of those titles (up to 10) found in the top 10:")
    for name, v in found.items():
        print(f"  {name:<30} {np.mean(v):.1%}")
    tokens = np.mean([len(t) for t in titled])
    print(f"Storage per item: pooled {word_dirs.shape[1]} numbers; MaxSim {tokens:.2f} words x "
          f"{word_dirs.shape[1]} = {tokens * word_dirs.shape[1]:.0f} numbers on average")
