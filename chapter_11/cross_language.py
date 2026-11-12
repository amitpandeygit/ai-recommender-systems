"""One space for many languages: TF-IDF against three pretrained text encoders (section 11.3).

Part 1, translation retrieval. Every item's title is written in all six languages
(title_words.py). For each pair of languages, each title in the first is compared
with all 5,000 titles in the second, and the test counts how often the nearest one
is its own translation (the same words of Chapter 3; items whose titles have the same
words count as the same title). A method that puts one meaning in one place, whatever
the language, scores 1. The same test is then run on the 148 title words on their own.

Part 2, the next-video test of Chapters 9 and 10 (requests of days 23 to 30, each
answered from the clicks the server had received before it). Each method scores
every video by the similarity of its title, written in its own language, to the
request's profile: the mean vector of the titles in its history (Listing 11.2). The
methods differ only in how a title becomes a vector:

  Chapter 3's tokens   - TF-IDF over the language-free tokens: titles as if perfectly
                         translated, a reference no real platform has
  TF-IDF               - TF-IDF over the written words
  all-MiniLM-L6-v2, paraphrase-multilingual-MiniLM-L12-v2, Qwen3-Embedding-0.6B -
                         the pretrained encoders of embed_texts.py

For TF-IDF and the two multilingual models, a second row puts every video in the
user's app language above every other video, as Chapter 10's "New for you" source does,
and orders each group by the same similarity. Nothing is fitted, so there is no
validation choice. "Other languages" is the share of
the videos shown that are not in the user's app language.

Needs Python 3.12 or later, PyTorch and text_embeddings.npz (embed_texts.py). Run from
the repository root (about 4 minutes):
    python chapter_11/cross_language.py
"""
import os
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "2"
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from content_data import MODELS, embedding_texts, load, text_embeddings  # noqa: E402
from content_models import profiles, scorer, tfidf  # noqa: E402
from graph_data import chapter8_rows, environment, ndcg, online, paired, phase, top10  # noqa: E402
from slate_data import split  # noqa: E402
from tesselo_shop import LANGUAGES  # noqa: E402
import baselines as B  # noqa: E402

DAY = 86_400


def translation_accuracy(E, title_idx):
    """acc[a, b]: share of titles in language a whose nearest title in language b is
    their own translation. E holds one unit vector per text."""
    n_lang = title_idx.shape[1]
    acc = np.eye(n_lang)
    for a in range(n_lang):
        Q = E[title_idx[:, a]]
        for b in range(n_lang):
            if a != b:
                K = E[title_idx[:, b]]
                nearest = np.argmax(Q @ K.T, axis=1)
                acc[a, b] = np.mean(title_idx[nearest, b] == title_idx[:, b])
    return acc


def line(label, W, R, top, base):
    on = online(W, R, top)
    later = R["place"] >= 1
    users, items = W["users"], W["items"]
    shown = top >= 0
    other = (items["language"][np.where(shown, top, 0)] != users["language"][R["user"]][:, None])[shown].mean()
    against = "" if base is None else "{:+.4f} ({:.4f})".format(*paired(on, base, R["user"])[:2])
    print(f"  {label:<62}{ndcg(R, top):>9.4f}{on.mean():>9.4f}{on[later].mean():>9.4f}{against:>18}"
          f"{other:>17.1%}")
    return on


if __name__ == "__main__":
    t0 = time.time()
    print(environment())
    W = load()
    items, users = W["items"], W["users"]
    texts, idx = embedding_texts(W)
    T = idx["titles"]
    print(f"Chapter 2's world; {len(texts):,} distinct texts, {T.shape[0]:,} titles in {T.shape[1]} languages  "
          f"[{time.time() - t0:.0f} s]")
    for i in range(3):
        print(f"  item {i}: " + " | ".join(f"{LANGUAGES[g]}: {texts[T[i, g]]}" for g in range(T.shape[1])))

    # ---- Part 1: translation retrieval
    words = tfidf([t.split() for t in texts])
    tf = words.toarray().astype(np.float32)
    emb = {"TF-IDF": tf} | {m: text_embeddings(W, m)[0] for m in MODELS}
    ex = int(np.flatnonzero((items["category"] == 0) & ~items["is_product"])[0])     # the first cooking video
    other = int(np.flatnonzero((items["category"] == 4) & ~items["is_product"])[0])  # the first travel video
    print(f"\nWorked example: item {ex} in English \"{texts[T[ex, 0]]}\" and in Hindi \"{texts[T[ex, 1]]}\"; item "
          f"{other} in Hindi \"{texts[T[other, 1]]}\". Cosine similarities:")
    for name, E in emb.items():
        print(f"  {name:<40} translation {float(E[T[ex, 0]] @ E[T[ex, 1]]):.3f}   other video "
              f"{float(E[T[ex, 0]] @ E[T[other, 1]]):.3f}")
    print("\nTranslation retrieval: share of titles whose nearest title in another language is their own "
          "translation (mean over the 30 ordered pairs of different languages)")
    acc = {}
    for name, E in emb.items():
        acc[name] = translation_accuracy(E, T)
        off = acc[name][~np.eye(len(LANGUAGES), dtype=bool)]
        print(f"  {name:<40}{off.mean():>8.3f}   lowest pair {off.min():.3f}, highest {off.max():.3f}")
    print("\nWord translation: share of the 148 title words whose nearest word in another language is the same "
          "word (mean over the 30 ordered pairs)")
    for name, E in emb.items():
        a_ = translation_accuracy(E, idx["words"])
        print(f"  {name:<40}{a_[~np.eye(len(LANGUAGES), dtype=bool)].mean():>8.3f}")
    for name in ("TF-IDF", MODELS[-1]):
        print(f"\n  {name}: from the row's language to the column's")
        print("        " + "".join(f"{g:>8}" for g in LANGUAGES))
        for a, g in enumerate(LANGUAGES):
            print(f"  {g:>6}" + "".join(f"{acc[name][a, b]:>8.3f}" for b in range(len(LANGUAGES))))
    print(f"  [{time.time() - t0:.0f} s]")

    # ---- Part 2: the next-video test
    sp = split(W, "test")
    R, H, _, bad = phase(W, 23, sp["S"])
    pop = B.popularity(W, 23 * DAY)
    own = T[np.arange(len(T)), items["language"]]                 # each title in its own language
    print(f"\nNext-video test: {len(R['user']):,} requests of days 23 to 30; each video scored by the similarity "
          f"of its title (in its own language) to the request's history  [{time.time() - t0:.0f} s]")
    print(f"  {'':<62}{'NDCG@10':>9}{'online':>9}{'later':>9}{'against Chapter 8':>18}{'other languages':>17}")
    ch8, _, _ = chapter8_rows(W, sp, R, bad)
    base = line("Chapter 8's ranker", W, R, ch8, None)
    line("popularity, last 7 days", W, R, top10(pop, bad), base)
    feats = {"Chapter 3's tokens (as if translated)": tfidf([[str(w) for w in t] for t in items["title"]]),
             "TF-IDF on the written words": words[own]}
    feats |= {m: emb[m][own] for m in MODELS}
    same = (users["language"][R["user"]][:, None] == items["language"][None, :]).astype(np.float32)
    results = {}
    for name, F in feats.items():
        P = profiles(F, H)
        results[name] = line(name, W, R, top10(scorer(P, F), bad, tie=pop), base)
        if name in ("TF-IDF on the written words", MODELS[1], MODELS[-1]):
            f = scorer(P, F)
            results[name + ", first"] = line(name + ", same language first", W, R, top10(
                lambda a, b, f=f: f(a, b) + 1_000.0 * same[a:b], bad, tie=pop), base)
    for a_, b_ in ((MODELS[-1], "TF-IDF on the written words"),
                   (MODELS[-1] + ", first", "TF-IDF on the written words, first"),
                   (MODELS[-1] + ", first", MODELS[1] + ", first")):
        print(f"  {a_} against {b_}: " + "{:+.4f} ({:.4f}) [{:+.4f}, {:+.4f}]".format(
            *paired(results[a_], results[b_], R["user"])))
    print("  (online: expected satisfied plays per row; later: requests after the first of their visit; against: "
          "paired difference over the same requests, standard error from 1,000 bootstrap draws of whole users; "
          f"other languages: share of the videos shown not in the user's app language)  [{time.time() - t0:.0f} s]")
