"""Listing 6.9 - Multi-channel retrieval: merging candidate sources (section 6.8).

Tesselo's retrieval stage runs several sources side by side and merges their
candidates into one pool for the ranking stages (Chapter 7). Each source gets a
quota of slots; the merge takes the sources in priority order, skips items
already in the pool, and gives unused slots to the next source.

The question for retrieval is not "is the best item first?" but "is it in the
pool at all?", because an item dropped here can never be shown (Chapter 1). So
the measure is recall of the pool: the share of each user's relevant videos that
the pool contains, and the share of users whose pool holds at least one.

Sources: Chapter 4's iALS, the two-tower model of section 6.6 (MAIN settings),
Chapter 3's item-based CF, Chapter 3's "New for you" row (videos below the
eligibility threshold, in your language first, then by content) and popular in
your language. Quotas are chosen on the validation week and tested once on the
test week, as everywhere in this book.

Needs Python 3.12 or later. Run from the repository root (about five minutes):
    python chapter_06/retrieval_channels.py
"""
import os
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "2"              # NumPy's linear algebra: the same thread count on every run
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from two_tower import MAIN, embed_all, make_data, train_two_tower  # noqa: E402
from tesselo_shop import DAY, LANGUAGES, Config, build  # noqa: E402
from similarity_recs import (MIN_BUYERS, NO, cf_scores, content_scores, group_popularity,  # noqa: E402
                             item_features, item_neighbours, test_positives, top_k, user_item_scores)
from implicit_mf import eligible_and_scored, ials  # noqa: E402

POOL = 200


def merge(source_lists, quotas, pool=POOL):
    """One user's pool: each source in order adds up to its quota of items not already
    in the pool; slots a source leaves unused pass to the sources after it, and any
    still empty at the end go back through the sources in order. Returns the pool and,
    for each item in it, the index of the source that added it."""
    out, src, seen = [], [], set()
    owed = 0
    for j, (lst, q) in enumerate(zip(source_lists, quotas)):
        take = q + owed
        added = 0
        for i in lst:
            if added == take:
                break
            if i >= 0 and i not in seen:
                out.append(i)
                src.append(j)
                seen.add(i)
                added += 1
        owed = take - added
    for j, lst in enumerate(source_lists):                      # backfill, if anything is left
        for i in lst:
            if len(out) == pool:
                return out, src
            if i >= 0 and i not in seen:
                out.append(i)
                src.append(j)
                seen.add(i)
    return out, src


def sources_at(events, users, items, cutoff, n_users, n_items, video, feats):
    """Each source's ranked list (up to POOL items) for every user, from data known at cutoff."""
    weight, used = user_item_scores(events, users, cutoff, n_users, n_items)
    ok = video & ((weight > 0).sum(axis=0) >= MIN_BUYERS)
    data = make_data(weight, items, users, feats, int(items["creator"].max()) + 1)
    pu, pi = np.nonzero(weight > 0)
    U, V = embed_all(train_two_tower(data, pu, pi, **MAIN), data)
    X, Y = ials(weight, 16, 1.0, 10.0)
    lang = group_popularity(weight, users["language"], len(LANGUAGES))[users["language"]]
    sim = item_neighbours(weight, k=50, strength=0.0)
    new_ok = video & ~ok                                           # below the threshold: "New for you" only
    same_lang = lambda r: (users["language"][r, None] == items["language"][None, :]).astype(np.float64)  # noqa: E731
    score = {"iALS": lambda r: eligible_and_scored(X[r] @ Y.T, weight[r], ok),
             "two-tower": lambda r: eligible_and_scored(U[r] @ V.T, weight[r], ok),
             "item-based CF": lambda r: cf_scores(weight[r], sim),
             "New for you": lambda r: np.where(new_ok[None, :],
                                               same_lang(r) + 0.5 * content_scores(weight[r], feats)[0], NO),
             "popular in your language": lambda r: lang[r]}
    return weight, used, score


def pools(score, rows, used, video, quotas):
    lists = {name: top_k(fn(rows), used[rows], video, POOL) for name, fn in score.items()}
    merged = [merge([lists[n][r] for n in score], quotas) for r in range(len(rows))]
    return lists, [m for m, _ in merged], [s for _, s in merged]


def recall(lists, rows, pos):
    """Mean share of relevant videos in the list, and share of users with at least one."""
    r = [len(pos[u] & set(np.asarray(t).tolist())) / len(pos[u]) for u, t in zip(rows, lists)]
    return float(np.mean(r)), float(np.mean([x > 0 for x in r]))


if __name__ == "__main__":
    import time
    t0 = time.time()
    cfg = Config()
    users, items, served, events, orders, truth = build(cfg)
    n_users, n_items = cfg.num_users, cfg.num_items
    video = ~items["is_product"]
    feats = item_features(items, n_items)
    names = ("iALS", "two-tower", "item-based CF", "New for you", "popular in your language")
    options = [(200, 0, 0, 0, 0), (100, 100, 0, 0, 0), (100, 50, 30, 20, 0), (80, 40, 40, 20, 20),
               (100, 40, 20, 20, 20), (60, 60, 40, 20, 20)]

    # ---- validation week: choose the quotas
    w16, used16, score16 = sources_at(events, users, items, 16 * DAY, n_users, n_items, video, feats)
    pos16 = test_positives(events, users, 16 * DAY, 23 * DAY, used16, video)
    r16 = np.array(sorted(pos16))
    print(f"Validation week, {len(r16):,} users: recall@{POOL} of the merged pool for each set of quotas "
          f"({', '.join(names)})")
    best = None
    for q in options:
        _, merged, _ = pools(score16, r16, used16, video, q)
        rc, any_ = recall(merged, r16, pos16)
        print(f"  quotas {str(q):<26} recall {rc:.3f}   users with a relevant video in the pool {any_:.1%}")
        if best is None or round(rc, 9) > round(best[0], 9):
            best = (rc, q)
    print(f"Chosen: {best[1]}  [{time.time() - t0:.0f} s]")

    # ---- test week: each source alone, the chosen merge, and (for reference only) the other quotas
    weight, used, score = sources_at(events, users, items, 23 * DAY, n_users, n_items, video, feats)
    pos = test_positives(events, users, 23 * DAY, 30 * DAY, used, video)
    rows = np.array(sorted(pos))
    print(f"\nTest week, {len(rows):,} users, pools of {POOL} videos "
          f"(distinct = videos that appear in at least one user's pool, of {int(video.sum()):,}):")
    lists = None
    for q in options:
        lists, merged, src = pools(score, rows, used, video, q)
        rc, any_ = recall(merged, rows, pos)
        src = np.concatenate([np.array(x) for x in src])
        print(f"  quotas {str(q):<26} recall {rc:.3f}   users {any_:.1%}   distinct "
              f"{len(np.unique(np.concatenate(merged))):,}   added by source: "
              + ", ".join(f"{np.mean(src == j):.0%}" for j in range(len(names)))
              + ("   <- chosen on the validation week" if q == best[1] else ""))
    for name in names:
        rc, any_ = recall([lists[name][r] for r in range(len(rows))], rows, pos)
        print(f"  {name + ' alone':<40} recall {rc:.3f}   users with a relevant video {any_:.1%}")
    print("  relevant videos (user, video pairs) that only one source finds in its top 200:")
    total = sum(len(pos[u]) for u in rows)
    for name in names:
        others = [n for n in names if n != name]
        only = sum(len({i for i in pos[u] if i in set(lists[name][r].tolist())
                        and not any(i in set(lists[o][r].tolist()) for o in others)})
                   for r, u in enumerate(rows))
        print(f"    {name:<34} {only:>4} of {total:,}")
    print(f"[{time.time() - t0:.0f} s]")
