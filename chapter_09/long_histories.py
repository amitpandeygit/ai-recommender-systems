"""Listing 9.6 - Very long histories: cut, search or compress (section 9.6).

Tesselo's month of logs gives each user about a dozen clicks, so long histories need
users of their own. This script makes 3,000 synthetic users (seed cfg.seed + 950),
each with 2,000 past clicks, from Tesselo's real videos and their hidden vectors:

  * every user has 4 lasting interests: a video category's direction plus a
    personal offset, with shares drawn at random (some interests are rare)
  * the history is 400 visits of 5 clicks; each visit follows one interest, and
    picks 5 videos with probability growing with exp(8 x the video's match to it)
  * the target is the next visit: its 5 videos, hidden among 95 random videos. It is
    run twice. "Repeats allowed" lets the next visit pick videos already in the
    history. "New videos only" draws the next visit only from videos the user never
    clicked (seed cfg.seed + 951), and the 95 others exclude the history too, as on
    Tesselo's feed, which never shows a clicked video again

A rare interest may not appear in the last 50 clicks at all, so this is where long
histories should pay. Each method scores a candidate c by how close it is to the
history it reads, with Tesselo's own iALS video vectors (Chapter 4, day 23):

    score(c) = log( sum_j n_j exp(tau x y_c . y_j) / sum_j n_j )

a soft version of "the most similar thing the user did" (n_j = 1 for a click, the
cluster size for a centroid). The methods differ only in what they read:

  last 50 / last 500   - the most recent clicks
  everything           - all 2,000 clicks: the quality to aim for, at the most cost
  search: category     - SIM's hard search (Pi and colleagues, 2020): the user's clicks
                         in the candidate's category, the 50 most recent
  search: nearest 50   - SIM's soft search: the 50 clicks whose vectors are closest to
                         the candidate (exact here; ETA and TWIN make it cheap)
  compressed: 16 / 64 / 200 - k-means: that many centroids of the history's vectors,
                         each with its size, built offline: a simple compression inspired
                         by TWIN V2, which adds grouping by completion ratio, recursive
                         clustering, feature aggregation and cluster-aware attention
                         (200 is about a tenth of the history, TWIN V2's ratio)

tau is chosen for each method on 500 validation users; results are on the other
2,500, with each method's NDCG@10 compared with reading everything, paired over
users. The cost is the history vectors examined to choose what to read (all 2,000
for an exact nearest search; the category search reads a per-user index of clicks by
category, built offline as in SIM, so it examines only the 50 it scores), the vectors
scored per candidate, and the bytes kept per user.

Needs Python 3.12 or later and PyTorch. Run from the repository root (about six minutes):
    python chapter_09/long_histories.py
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
from seq_data import WORLDS, environment, load, paired  # noqa: E402
from planted_world import category_directions  # noqa: E402
from metrics import ndcg_at_k  # noqa: E402

N_USERS, N_INTERESTS, VISITS, PER_VISIT = 3_000, 4, 400, 5
N_NEG, N_VAL = 95, 500
TAUS = (1, 2, 4, 8, 16, 32, 64)
CLUSTERS = (16, 64, 200)
K = 50


def synthetic_users(W, rng, rng_new):
    """Histories (users x 2,000 video ids, oldest first), targets (users x 5) and each
    user's interest shares. Two targets for the same next visit: one that may repeat
    videos already in the history, and one drawn only from videos the user never
    clicked (rng_new), as on Tesselo's feed, which never shows a clicked video again."""
    items = W["items"]
    video = np.flatnonzero(~items["is_product"])
    vec = items["vec"][video].astype(np.float64)
    dirs = category_directions(W["cfg"])
    hist = np.empty((N_USERS, VISITS * PER_VISIT), dtype=np.int64)
    target = np.empty((N_USERS, PER_VISIT), dtype=np.int64)
    target_new = np.empty((N_USERS, PER_VISIT), dtype=np.int64)
    shares = np.empty((N_USERS, N_INTERESTS))
    unseen = np.zeros(N_USERS, dtype=bool)                       # target interest absent from the last 10 visits
    for u in range(N_USERS):
        c = rng.choice(len(dirs), N_INTERESTS, replace=False)
        interest = dirs[c] + 0.5 * rng.normal(size=(N_INTERESTS, dirs.shape[1])) / np.sqrt(dirs.shape[1])
        interest /= np.linalg.norm(interest, axis=1, keepdims=True)
        w = np.sort(rng.dirichlet(np.full(N_INTERESTS, 0.6)))[::-1]
        shares[u] = w
        logits = 8.0 * interest @ vec.T
        p = np.exp(logits - logits.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        visit_interest = rng.choice(N_INTERESTS, VISITS + 1, p=w)
        unseen[u] = visit_interest[-1] not in visit_interest[-11:-1]
        for v, k in enumerate(visit_interest):
            pick = video[rng.choice(len(video), PER_VISIT, replace=False, p=p[k])]
            if v < VISITS:
                hist[u, v * PER_VISIT:(v + 1) * PER_VISIT] = pick
            else:
                target[u] = pick
        q = p[visit_interest[-1]] * ~np.isin(video, hist[u])             # the same visit, new videos only
        target_new[u] = video[rng_new.choice(len(video), PER_VISIT, replace=False, p=q / q.sum())]
    return hist, target, target_new, shares, video, unseen


def soft_score(y_c, Y_h, n_h, tau):
    """Listing 9.6 (first part). log of the weighted mean of exp(tau x similarity)."""
    s = tau * np.einsum("cd,cjd->cj", y_c, Y_h) if Y_h.ndim == 3 else tau * y_c @ Y_h.T
    m = s.max(axis=-1, keepdims=True)
    return (m[..., 0] + np.log((n_h * np.exp(s - m)).sum(axis=-1) / n_h.sum(axis=-1)))


def kmeans(Y, k=16, iters=10, rng=None):
    """Listing 9.6 (second part). k-means on one user's history vectors (cosine):
    k centroids and how many clicks each holds."""
    c = Y[rng.choice(len(Y), k, replace=False)]
    for _ in range(iters):
        a = np.argmax(Y @ c.T, axis=1)                           # each click's nearest centroid
        m = np.zeros_like(c)
        np.add.at(m, a, Y)                                       # the sum of each centroid's clicks
        norm = np.linalg.norm(m, axis=1, keepdims=True)
        c = np.where(norm > 0, m / np.maximum(norm, 1e-12), c)   # a centroid with no clicks stays put
    a = np.argmax(Y @ c.T, axis=1)
    return c, np.bincount(a, minlength=k).astype(float)


def select(method, u_hist, cand, Yv, cat_of, index):
    """The history each method reads for each candidate: (vectors, weights). index:
    the user's clicks by category, built offline (SIM's per-user index)."""
    if method == "last 50":
        h = u_hist[-50:]
        return Yv[h], np.ones(len(h))
    if method == "last 500":
        h = u_hist[-500:]
        return Yv[h], np.ones(len(h))
    if method == "everything":
        return Yv[u_hist], np.ones(len(u_hist))
    if method == "search: category":
        out, w = np.zeros((len(cand), K, Yv.shape[1])), np.zeros((len(cand), K))
        for k, c in enumerate(cand):
            h = index.get(int(cat_of[c]), u_hist[:0])[-K:]          # a lookup, not a scan of the history
            if len(h) == 0:
                h = u_hist[-1:]                                   # nothing in that category: the last click
            out[k, :len(h)], w[k, :len(h)] = Yv[h], 1.0
        return out, w
    if method == "search: nearest 50":
        sim = Yv[cand] @ Yv[u_hist].T
        top = np.argpartition(-sim, K, axis=1)[:, :K]
        return Yv[u_hist[top]], np.ones((len(cand), K))
    raise ValueError(method)


if __name__ == "__main__":
    t0 = time.time()
    print(environment())
    W = load(WORLDS[0])
    S = W["snaps"][23]
    Y = S["Y"] / np.maximum(np.linalg.norm(S["Y"], axis=1, keepdims=True), 1e-12)
    rng = np.random.default_rng(W["cfg"].seed + 950)
    rng_new = np.random.default_rng(W["cfg"].seed + 951)
    hist, target, target_new, shares, video, unseen = synthetic_users(W, rng, rng_new)
    cat_of = W["items"]["category"]
    print(f"Section 9.6: {N_USERS:,} synthetic users, {hist.shape[1]:,} past clicks each, the next visit's "
          f"{PER_VISIT} videos among {PER_VISIT + N_NEG} candidates  [{time.time() - t0:.0f} s]")
    print(f"  videos in the catalogue: {len(video):,}; iALS video vectors: {Y.shape[1]} numbers; history ids kept "
          f"as 4-byte integers")
    print("  interest shares, mean from largest to smallest: " + ", ".join(f"{x:.3f}" for x in shares.mean(0)))
    cands = np.empty((N_USERS, PER_VISIT + N_NEG), dtype=np.int64)
    cands_new = np.empty((N_USERS, PER_VISIT + N_NEG), dtype=np.int64)
    for u in range(N_USERS):
        pool = np.setdiff1d(video, target[u])
        cands[u] = np.r_[target[u], rng.choice(pool, N_NEG, replace=False)]
        pool = np.setdiff1d(video, np.r_[target_new[u], hist[u]])         # no video the user already clicked
        cands_new[u] = np.r_[target_new[u], rng_new.choice(pool, N_NEG, replace=False)]
    repeats = np.mean([np.isin(target[u], hist[u]).mean() for u in range(N_USERS)])
    print(f"  target videos already in the user's history: {repeats:.1%} (repeats allowed), 0.0% (new videos only)")
    index = [{int(g): hist[u][cat_of[hist[u]] == g] for g in np.unique(cat_of[hist[u]])}
             for u in range(N_USERS)]                            # built offline: each category's clicks, oldest first
    seen_recent = np.array([np.isin(cat_of[target[u]], cat_of[hist[u, -50:]]).mean() for u in range(N_USERS)])
    print(f"  target videos whose category appears in the last 50 clicks: {seen_recent.mean():.1%}; users whose "
          f"next visit follows an interest absent from their last 10 visits: {unseen.mean():.1%}")
    centroids = {k: [kmeans(Y[hist[u]], k, rng=np.random.default_rng(u)) for u in range(N_USERS)] for k in CLUSTERS}
    methods = ("last 50", "last 500", "everything", "search: category", "search: nearest 50") + tuple(
        f"compressed: {k}" for k in CLUSTERS)
    reads = {"last 50": 50, "last 500": 500, "everything": hist.shape[1], "search: category": K,
             "search: nearest 50": K} | {f"compressed: {k}": k for k in CLUSTERS}
    examined = {"last 50": 50, "last 500": 500, "everything": hist.shape[1], "search: category": K,
                "search: nearest 50": hist.shape[1]} | {f"compressed: {k}": k for k in CLUSTERS}
    kept = {"last 50": 50 * 4, "last 500": 500 * 4, "everything": hist.shape[1] * 4,
            "search: category": hist.shape[1] * 4, "search: nearest 50": hist.shape[1] * 4,
            } | {f"compressed: {k}": k * (Y.shape[1] * 4 + 4) for k in CLUSTERS}   # float32 centroids and sizes
    for mode, tgt, cnd in (("repeats allowed", target, cands), ("new videos only", target_new, cands_new)):
        print(f"\n  {mode}: the next visit's videos {'may' if mode == 'repeats allowed' else 'may not'} be in the "
              f"history")
        print(f"  {'':<22}{'tau':>5}{'NDCG@10':>9}{'recall@10':>11}{'rare interest':>15}{'examined':>10}"
              f"{'scored':>8}{'bytes / user':>14}{'ms / request':>14}  against everything")
        rows = {}
        for method in methods:
            res = {}
            for tau in TAUS:
                nd, rc, ms = [], [], []
                for u in range(N_USERS):
                    c = cnd[u]
                    t1 = time.perf_counter()
                    if method.startswith("compressed"):
                        cu = centroids[int(method.split(": ")[1])][u]
                        s = soft_score(Y[c], cu[0], cu[1], tau)
                    else:
                        Yh, n = select(method, hist[u], c, Y, cat_of, index[u])
                        s = soft_score(Y[c], Yh, n, tau)
                    ms.append((time.perf_counter() - t1) * 1e3)
                    top = c[np.argsort(-s, kind="stable")[:10]].tolist()
                    nd.append(ndcg_at_k(top, {int(i): 1.0 for i in tgt[u]}))
                    rc.append(len(set(top) & set(tgt[u].tolist())) / PER_VISIT)
                res[tau] = (np.array(nd), np.array(rc), np.array(ms))
            best = max(TAUS, key=lambda t: round(res[t][0][:N_VAL].mean(), 9))
            rows[method] = (best, *res[best], time.time() - t0)
        test = np.arange(N_USERS) >= N_VAL
        for method, (best, nd, rc, ms, sec) in rows.items():
            d, se, lo, hi = paired(nd[test], rows["everything"][1][test], np.flatnonzero(test))
            print(f"  {method:<22}{best:>5}{nd[test].mean():>9.4f}{rc[test].mean():>11.4f}"
                  f"{nd[test & unseen].mean():>15.4f}{examined[method]:>10,}{reads[method]:>8,}{kept[method]:>14,}"
                  f"{np.median(ms[test]):>14.3f}{d:>+11.4f} ({se:.4f}) [{lo:+.4f}, {hi:+.4f}]  [{sec:.0f} s]")
        print("  (against everything: NDCG@10 minus that of all 2,000 clicks, paired over the same test users, "
              "with its standard error and 95% interval from 1,000 bootstrap draws of users)")
