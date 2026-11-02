"""Listings 4.2 to 4.6 - Matrix factorisation for implicit feedback (sections 4.4 to 4.8).

Section 4.4: iALS (Hu, Koren and Volinsky, 2008), trained by alternating least
             squares, and BPR (Rendle and colleagues, 2009), trained by SGD on
             pairs, with uniform or popularity-weighted negative samples.
Section 4.5: what the embeddings learn, checked against the hidden tastes.
Section 4.6: "Because you watched" from item vectors.
Section 4.7: cold start - onboarding fold-in from a held-out user's first videos, and item
             vectors predicted from content for videos with no history.
Section 4.8: the same honest test as Chapter 3: the "Because you watched" video
             row, a time split, settings chosen on the validation week, Chapter
             3's eligibility rules (five engaged users per item, no evidence ->
             no recommendation) and its fallback chain.

Inputs are Chapter 3's implicit weights: log(1 + decayed score) for positive,
unblocked pairs. This is a laptop-scale NumPy implementation for teaching.

Needs Python 3.12 or later. Run from the repository root (about five minutes):
    python chapter_04/implicit_mf.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "chapter_03"))
from tesselo_shop import CATEGORIES, DAY, LANGUAGES, Config, build  # noqa: E402
from similarity_recs import (DECIMALS, MIN_BUYERS, NO, content_scores, group_popularity,  # noqa: E402
                             hit_rate, item_features, item_neighbours, lists, mcnemar_exact,
                             test_positives, top_k, user_item_scores, cf_scores)
from implicit_scores import EVENT_WEIGHTS, prepare  # noqa: E402


# ---------------------------------------------------------------- section 4.4: iALS
def ials_solve(R, Y, reg, alpha):
    """One half-step of iALS: the best vector for every row of R, with Y fixed.

    Row u minimises  sum over all items i of c_ui (p_ui - x_u . y_i)^2 + reg |x_u|^2,
    with p_ui = 1 and c_ui = 1 + alpha * r_ui if r_ui > 0, else p_ui = 0 and c_ui = 1.
    Solution: x_u = (Y'Y + Y_u' (C_u - I) Y_u + reg I)^-1  Y_u' c_u.
    Y'Y is shared by all rows, so each row only touches its own few items.
    """
    YtY = Y.T @ Y                                            # k x k, computed once per half-step
    X = np.zeros((R.shape[0], Y.shape[1]))
    rows, cols = np.nonzero(R)                               # teaching code: scans the dense R
    start = np.searchsorted(rows, np.arange(R.shape[0] + 1))
    for u in range(R.shape[0]):
        idx = cols[start[u]:start[u + 1]]
        if len(idx) > 0:                                     # no history: no vector (zeros)
            X[u] = solve_row(Y, YtY, idx, R[u, idx], reg, alpha)
    return X


def solve_row(Y, YtY, idx, r, reg, alpha):
    """The closed-form solve for one row that engaged with items idx, with weights r.
    Costs about len(idx) * k^2 + k^3 operations, whatever the size of the catalogue."""
    Yu, c = Y[idx], 1.0 + alpha * r
    A = YtY + (Yu.T * (c - 1.0)) @ Yu + reg * np.eye(Y.shape[1])
    return np.linalg.solve(A, Yu.T @ c)


def ials(R, k, reg, alpha, iters=10, seed=0):
    """Alternate: users given items, then items given users."""
    Y = np.random.default_rng(seed).normal(0, 0.1, (R.shape[1], k))
    for _ in range(iters):
        X = ials_solve(R, Y, reg, alpha)
        Y = ials_solve(R.T, X, reg, alpha)
    return X, Y


# ---------------------------------------------------------------- section 4.4: BPR
def bpr(R, k, reg, lr=0.05, epochs=100, neg_power=0.0, batch=256, seed=0):
    """BPR with an item bias: score(u, i) = b_i + p_u . q_i.

    Each step takes an observed pair (u, i) and a sampled item j the user has not
    engaged with, and pushes score(u, i) above score(u, j) by gradient ascent on
    ln sigmoid(score(u, i) - score(u, j)) minus an L2 penalty on every parameter,
    the biases included. Negatives are drawn in proportion to
    (engaged users + 1) ** neg_power: 0 is uniform, 1 follows popularity.
    All updates in a batch are computed from the values before that batch.
    """
    rng = np.random.default_rng(seed)
    n_users, n_items = R.shape
    P = rng.normal(0, 0.1, (n_users, k))
    Q = rng.normal(0, 0.1, (n_items, k))
    b = np.zeros(n_items)
    pos = R > 0
    us, its = np.nonzero(pos)
    weight = (np.bincount(its, minlength=n_items) + 1.0) ** neg_power
    for _ in range(epochs):
        order = rng.permutation(len(us))
        for s in range(0, len(order), batch):
            o = order[s:s + batch]
            u, i = us[o], its[o]
            j = rng.choice(n_items, len(o), p=weight / weight.sum())
            keep = ~pos[u, j]                                # j must be unobserved for u
            u, i, j = u[keep], i[keep], j[keep]
            pu, qi, qj, bi, bj = P[u].copy(), Q[i].copy(), Q[j].copy(), b[i].copy(), b[j].copy()
            g = 1.0 / (1.0 + np.exp(bi - bj + np.einsum("nk,nk->n", pu, qi - qj)))  # sigmoid(-x)
            np.add.at(P, u, lr * (g[:, None] * (qi - qj) - reg * pu))
            np.add.at(Q, i, lr * (g[:, None] * pu - reg * qi))
            np.add.at(Q, j, lr * (-g[:, None] * pu - reg * qj))
            np.add.at(b, i, lr * (g - reg * bi))
            np.add.at(b, j, lr * (-g - reg * bj))
    return P, Q, b


# ---------------------------------------------------------------- sections 4.5 to 4.7
def taste_correlations(scorers, users, items, history_users, eligible_items, n=100_000, seed=3):
    """Pooled Pearson correlation between each model's score and the hidden match,
    over one sample of random (user, item) pairs: users with history, eligible items.
    Every model is judged on the same pairs, those that all of them can score; each
    model's coverage (the share of sampled pairs it can score) is reported apart.
    Possible only in a simulation. Returns {name: (coverage, correlation)} and the
    share of pairs that every model can score."""
    rng = np.random.default_rng(seed)
    u = rng.choice(history_users, n)
    i = rng.choice(np.flatnonzero(eligible_items), n)
    true = np.einsum("nd,nd->n", users["vec"][u], items["vec"][i])
    s = {name: f(u, i) for name, f in scorers.items()}
    common = np.logical_and.reduce([np.isfinite(v) for v in s.values()])
    return ({name: (float(np.isfinite(v).mean()), float(np.corrcoef(v[common], true[common])[0, 1]))
             for name, v in s.items()}, float(common.mean()))


def cosine_neighbours(Y, item, k=10, allowed=None):
    """Up to k items whose vectors point most nearly the same way as item's.
    Candidates are filtered first: never the item itself, never a disallowed item,
    never a zero vector. A zero-vector query gets an empty list."""
    norms = np.linalg.norm(Y, axis=1)
    cand = norms > 0
    if allowed is not None:
        cand &= allowed
    cand[item] = False
    if norms[item] == 0:
        return np.array([], dtype=int)
    idx = np.flatnonzero(cand)
    sim = np.round(Y[idx] @ Y[item] / (norms[idx] * norms[item]), DECIMALS)
    return idx[np.argsort(-sim, kind="stable")[:k]]


def table_neighbours(sim, item, k=10, allowed=None):
    """Up to k neighbours from a precomputed similarity table (Chapter 3's CF).
    Only supported links (similarity > 0) count: a list may be short or empty."""
    cand = sim[item] > 0
    if allowed is not None:
        cand &= allowed
    cand[item] = False
    idx = np.flatnonzero(cand)
    return idx[np.argsort(-np.round(sim[item, idx], DECIMALS), kind="stable")[:k]]


def fold_in(Y, YtY, items_used, weights, reg, alpha):
    """A vector for a new or changed user, with item vectors fixed: one iALS row solve.
    YtY = Y.T @ Y is computed once per item-model version and reused for every user,
    so the work depends on the user's few items, not on the size of the catalogue.
    No items: a zero vector, which scores nothing, so the fallback chain takes over."""
    idx = np.asarray(items_used, dtype=np.int64)
    if len(idx) == 0:
        return np.zeros(Y.shape[1])
    return solve_row(Y, YtY, idx, np.asarray(weights, dtype=np.float64), reg, alpha)


def content_to_factors(feats, Y, known, ridge=1.0):
    """Ridge regression from content features to item vectors, fitted on items
    with history. Returns predicted vectors for every item, and the fitted map W:
    serving a video uploaded later needs W (and the same feature vocabulary), so W
    belongs to the same model version as Y."""
    F = feats[known]
    W = np.linalg.solve(F.T @ F + ridge * np.eye(F.shape[1]), F.T @ Y[known])
    return feats @ W, W


def worked_example():
    """Section 4.4: one user's iALS update with four items in two dimensions."""
    Y = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, -1.0]])   # A, B, C, D
    R = np.array([[1.0, 0.0, 0.5, 0.0]])                                # watched A and C
    x = ials_solve(R, Y, reg=1.0, alpha=10.0)[0]
    print(f"Worked iALS step: x_u = ({x[0]:.4f}, {x[1]:.4f}); scores "
          + ", ".join(f"{n} {s:.3f}" for n, s in zip("ABCD", Y @ x)))


def bpr_worked_example():
    """Section 4.4: one BPR step for one (user, positive, negative) triple."""
    p, qi, qj, bi, bj, lr, reg = (np.array([0.5, 0.2]), np.array([0.4, 0.1]), np.array([0.3, 0.3]),
                                  0.1, 0.2, 0.05, 0.01)
    x = bi - bj + p @ (qi - qj)
    g = 1.0 / (1.0 + np.exp(x))                             # sigmoid(-x), the same as in bpr()
    p2 = p + lr * (g * (qi - qj) - reg * p)
    qi2 = qi + lr * (g * p - reg * qi)
    qj2 = qj + lr * (-g * p - reg * qj)
    bi2, bj2 = bi + lr * (g - reg * bi), bj + lr * (-g - reg * bj)
    x2 = bi2 - bj2 + p2 @ (qi2 - qj2)
    print(f"Worked BPR step: x {x:.4f}, sigmoid(-x) {g:.4f}; p_u ({p2[0]:.5f}, {p2[1]:.5f}); "
          f"q_i ({qi2[0]:.5f}, {qi2[1]:.5f}); q_j ({qj2[0]:.5f}, {qj2[1]:.5f}); "
          f"b_i {bi2:.5f}, b_j {bj2:.5f}; new x {x2:.4f}")


def eligible_and_scored(score, weight, eligible_items):
    """No history or an ineligible item: no evidence, so no score (NO)."""
    has = (weight > 0).any(axis=1)[:, None]
    return np.where(has & eligible_items[None, :], score, NO)


if __name__ == "__main__":
    worked_example()
    bpr_worked_example()
    cfg = Config()
    users, items, served, events, orders, truth = build(cfg)
    n_users, n_items = cfg.num_users, cfg.num_items
    video = ~items["is_product"]

    # ---- validation week: train to day 16, test days 16 to 23
    w16, used16 = user_item_scores(events, users, 16 * DAY, n_users, n_items)
    pos16 = test_positives(events, users, 16 * DAY, 23 * DAY, used16, video)
    r16 = np.array(sorted(pos16))
    ok16 = video & ((w16 > 0).sum(axis=0) >= MIN_BUYERS)
    hist16 = np.flatnonzero((w16 > 0).any(axis=1))
    print(f"\nValidation week: {len(r16):,} users with a hit. Hit rate@10 and hidden-taste correlation:")
    best = None
    for k in (16, 32):
        for alpha in (1.0, 10.0):
            for reg in (1.0, 10.0):
                X, Y = ials(w16, k, reg, alpha)
                hr = hit_rate(top_k(eligible_and_scored(X[r16] @ Y.T, w16[r16], ok16), used16[r16], video),
                              r16, pos16)
                tc = taste_correlations({"m": lambda u, i: np.einsum("nk,nk->n", X[u], Y[i])},
                                        users, items, hist16, ok16)[0]["m"][1]
                print(f"  iALS k={k:>2} alpha={alpha:>4.0f} penalty={reg:>4.0f}: hit rate {hr:.3f}, "
                      f"taste correlation {tc:.2f}")
                if best is None or hr > best[0]:
                    best = (hr, k, alpha, reg)
    bpr_best = None
    for k in (16, 32):
        for neg in (0.0, 1.0):
            P, Q, b = bpr(w16, k, reg=0.01, neg_power=neg)
            hr = hit_rate(top_k(eligible_and_scored(P[r16] @ Q.T + b, w16[r16], ok16), used16[r16], video),
                          r16, pos16)
            tc = taste_correlations({"m": lambda u, i: np.einsum("nk,nk->n", P[u], Q[i]) + b[i]},
                                    users, items, hist16, ok16)[0]["m"][1]
            print(f"  BPR  k={k:>2} negatives {'uniform' if neg == 0 else 'popular'}: hit rate {hr:.3f}, "
                  f"taste correlation {tc:.2f}")
            if bpr_best is None or hr > bpr_best[0]:
                bpr_best = (hr, k, neg)
    _, k, alpha, reg = best
    print(f"Chosen: iALS k={k}, alpha={alpha:.0f}, penalty={reg:.0f}; BPR k={bpr_best[1]}, "
          f"negatives {'uniform' if bpr_best[2] == 0 else 'popular'}")

    # ---- test week: train to day 23, test days 23 to 30, settings fixed above
    weight, used = user_item_scores(events, users, 23 * DAY, n_users, n_items)
    pos = test_positives(events, users, 23 * DAY, 30 * DAY, used, video)
    rows = np.array(sorted(pos))
    engaged = (weight > 0).sum(axis=0)
    ok = video & (engaged >= MIN_BUYERS)
    history = (weight > 0).sum(axis=1)
    hist = np.flatnonzero(history > 0)
    v = weight[weight > 0]
    print(f"\nTest-week training data: {len(v):,} positive pairs from {len(hist):,} users "
          f"(median {np.median(history[hist]):.0f} items) on {(engaged > 0).sum():,} items; weights: median {np.median(v):.2f}, "
          f"90th percentile {np.percentile(v, 90):.2f}, largest {v.max():.2f}, sum {v.sum():,.0f}")
    X, Y = ials(weight, k, reg, alpha)
    P, Q, b = bpr(weight, bpr_best[1], reg=0.01, neg_power=bpr_best[2])
    Pp, Qp, bp = bpr(weight, bpr_best[1], reg=0.01, neg_power=1.0 - bpr_best[2])   # the other sampler
    lang_pop = group_popularity(weight, users["language"], len(LANGUAGES))[users["language"]]
    cf = cf_scores(weight, item_neighbours(weight, k=50, strength=0.0))           # Chapter 3's choice
    ials_s = eligible_and_scored(X @ Y.T, weight, ok)
    match = users["vec"].astype(np.float64) @ items["vec"].T
    other = "popular" if bpr_best[2] == 0 else "uniform"
    scores = {"popular in your language": lang_pop,
              "CF, then popular in language": (cf, lang_pop),
              "iALS": ials_s,
              "iALS, then popular in language": (ials_s, lang_pop),   # a chain, as in Chapter 3
              f"BPR, {'uniform' if bpr_best[2] == 0 else 'popular'} negatives":
                  eligible_and_scored(P @ Q.T + b, weight, ok),
              f"BPR, {other} negatives": eligible_and_scored(Pp @ Qp.T + bp, weight, ok),
              "hidden taste (simulation only)": match,
              "taste + popularity (simulation)": 5 * match + np.log(items["popularity"])}
    taste = {"iALS": lambda u, i: np.einsum("nk,nk->n", X[u], Y[i]),
             f"BPR, {'uniform' if bpr_best[2] == 0 else 'popular'} negatives":
                 lambda u, i: np.einsum("nk,nk->n", P[u], Q[i]) + b[i],
             f"BPR, {other} negatives": lambda u, i: np.einsum("nk,nk->n", Pp[u], Qp[i]) + bp[i]}
    tcs, _ = taste_correlations(taste, users, items, hist, ok)        # models that score every pair
    tcf, common = taste_correlations({"CF, then popular in language": lambda u, i: cf[u, i], **taste},
                                     users, items, hist, ok)            # on the pairs CF can score
    groups = {"all": rows, "0-4 items": rows[history[rows] <= 4], "5+ items": rows[history[rows] >= 5]}
    print(f"\nTest week (train to day 23, test days 23-30), videos only, hit rate@10")
    print(f"{'':<34}" + "".join(f"{g:>11}" for g in groups) + f"{'full lists':>12}{'taste, all':>12}"
          f"{'taste, CF':>11}")
    print(f"{'users':<34}" + "".join(f"{len(v):>11,}" for v in groups.values()))
    tops = {}
    for name, s in scores.items():
        top = tops[name] = lists(s, rows, used, video)
        tc = (f"{tcs[name][1]:>12.2f}" if name in tcs else f"{'':>12}") + \
             (f"{tcf[name][1]:>11.2f}" if name in tcf else "")
        print(f"{name:<34}" + "".join(f"{hit_rate(top, rows, pos, v):>11.3f}" for v in groups.values())
              + f"{(top[:, -1] >= 0).mean():>12.0%}" + tc)
    hit = {n: {u for u, t in zip(rows.tolist(), tops[n]) if pos[u] & set(t.tolist())}
           for n in ("iALS, then popular in language", "CF, then popular in language")}
    a_only = len(hit["iALS, then popular in language"] - hit["CF, then popular in language"])
    b_only = len(hit["CF, then popular in language"] - hit["iALS, then popular in language"])
    print(f"All users: hit only with iALS chain {a_only}, only with CF chain {b_only}; "
          f"exact McNemar p = {mcnemar_exact(a_only, b_only):.2g}")
    print(f"Taste correlation = pooled Pearson between score and hidden match, on 100,000 random pairs "
          f"(users with history, eligible videos). 'all': every sampled pair; 'CF': only the "
          f"{common:.1%} of pairs that Chapter 3's CF can score ({round(common * 100_000):,} pairs; the CF row uses CF alone).")

    # ---- section 4.6: "Because you watched" neighbours, for every eligible video
    queries = np.flatnonzero(ok)
    table = item_neighbours(weight, k=50, strength=0.0)       # Chapter 3's shipped table
    hidden = items["vec"].astype(np.float64)
    cat = items["category"]
    print(f"\nUp to 10 neighbours for each of {len(queries):,} eligible videos (eligible videos only, never itself):")
    print(f"  {'neighbours from':<30}{'videos with any':>16}{'mean list length':>18}{'same category':>15}")
    got = {}
    for name, neigh in (("iALS item vectors (cosine)", lambda i: cosine_neighbours(Y, i, 10, ok)),
                        ("Chapter 3 item-based CF", lambda i: table_neighbours(table, i, 10, ok)),
                        ("hidden vectors (truth)", lambda i: cosine_neighbours(hidden, i, 10, ok))):
        g = got[name] = [neigh(i) for i in queries]
        same = np.concatenate([cat[n] == cat[i] for i, n in zip(queries, g)])
        print(f"  {name:<30}{np.mean([len(n) > 0 for n in g]):>16.0%}{np.mean([len(n) for n in g]):>18.1f}"
              f"{same.mean():>15.0%}")
    chance = np.mean([(cat[queries] == cat[i]).sum() - 1 for i in queries]) / (len(queries) - 1)
    print(f"  {'random eligible videos':<30}{'':>16}{'':>18}{chance:>15.0%}")
    covered = [r for r, n in enumerate(got["Chapter 3 item-based CF"]) if len(n) > 0]
    print(f"  (mean list length is over all {len(queries):,} videos, empty lists included)")
    for name in got:
        lists_c = [got[name][r] for r in covered]
        same = np.concatenate([cat[n] == cat[queries[r]] for r, n in zip(covered, lists_c)])
        print(f"  on the {len(covered)} videos where CF has links: {name:<28} "
              f"{sum(len(n) for n in lists_c):>5,} neighbours, mean {np.mean([len(n) for n in lists_c]):.1f}, "
              f"same category {same.mean():.0%}")
    # dot product (vector length counts) against cosine (direction only) for the user's list
    unit = Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
    cos_s = eligible_and_scored(X @ unit.T, weight, ok)
    print(f"  Item vector length vs log(1 + engaged users), correlation: "
          f"{np.corrcoef(np.linalg.norm(Y[ok], axis=1), np.log1p(engaged[ok]))[0, 1]:.2f}")
    print(f"  User's list by dot product: hit rate@10 {hit_rate(tops['iALS'], rows, pos):.3f}; "
          f"by cosine (item vectors set to length 1): {hit_rate(lists(cos_s, rows, used, video), rows, pos):.3f}")
    anchor = queries[0]
    print(f"  Because you watched video {anchor} ({CATEGORIES[items['category'][anchor]]}): "
          + ", ".join(f"{j} ({CATEGORIES[items['category'][j]]})" for j in cosine_neighbours(Y, anchor, 5, ok)))

    # ---- section 4.7: onboarding - fold-in from a user's first videos, for users the item
    # model never saw. A first video is one with a positive action (a click, a completion or a
    # like); each is taken at the moment of that action, using only what was known then.
    d = prepare(events, users, 23 * DAY)
    positive = [e for e, w in EVENT_WEIGHTS.items() if w > 0]
    good = np.isin(d["event"], positive) & video[d["item"]]
    firsts = {}
    for t, u, i, e in sorted(zip((-d["age_days"][good]).tolist(), d["user"][good].tolist(),
                                 d["item"][good].tolist(), d["event"][good].tolist())):
        f = firsts.setdefault(u, {})
        if i not in f and len(f) < 5:                         # distinct videos, in time order
            f[i] = float(np.log1p(EVENT_WEIGHTS[e]))          # the score of that one fresh action
    onboard = np.array([u for u in rows if len(firsts.get(u, {})) >= 5])
    w_train = weight.copy()
    w_train[onboard] = 0.0                                    # held out of item-model training
    _, Yh = ials(w_train, k, reg, alpha)
    YtY = Yh.T @ Yh                                           # once per item-model version
    ok_h = video & ((w_train > 0).sum(axis=0) >= MIN_BUYERS)  # eligibility from training users only
    lang_h = group_popularity(w_train, users["language"], len(LANGUAGES))[users["language"]]
    print(f"\nOnboarding: {len(onboard):,} test users with 5+ videos with a positive action before day 23, "
          f"held out of item-model training ({(w_train > 0).any(axis=1).sum():,} training users)")
    print(f"  {'videos known':<34}{'fold-in':>9}{'popular in your language':>26}")
    for n in (1, 3, 5, None):
        V = np.zeros((len(onboard), k))
        seen = np.zeros((len(onboard), n_items), dtype=bool)
        for r, u in enumerate(onboard):
            if n is None:                                     # reference: full history at day 23
                idx = np.flatnonzero(weight[u] > 0)
                V[r] = fold_in(Yh, YtY, idx, weight[u, idx], reg, alpha)
                seen[r] = used[u]
            else:
                idx, wts = list(firsts[u])[:n], list(firsts[u].values())[:n]
                V[r] = fold_in(Yh, YtY, idx, wts, reg, alpha)
                seen[r, idx] = True                           # exclude only what is known
        fold_s = np.where(ok_h[None, :] & V.any(axis=1)[:, None], V @ Yh.T, NO)
        hits = []
        for top in (top_k(fold_s, seen, video), top_k(lang_h[onboard], seen, video)):
            hits.append({u for u, t in zip(onboard.tolist(), top) if pos[u] & set(t.tolist())})
        a_only, b_only = len(hits[0] - hits[1]), len(hits[1] - hits[0])
        label = "full history at day 23 (reference)" if n is None else f"first {n}, in time order"
        print(f"  {label:<34}{len(hits[0]) / len(onboard):>9.3f}{len(hits[1]) / len(onboard):>26.3f}"
              f"   only fold-in {a_only}, only chart {b_only}, McNemar p = {mcnemar_exact(a_only, b_only):.2g}")

    # ---- section 4.7: videos with hidden history get vectors from content
    new = (np.random.default_rng(cfg.seed + 303).random(n_items) < 0.10) & video   # as in Chapter 3
    w_old = np.where(new[None, :], 0.0, weight)
    Xo, Yo = ials(w_old, k, reg, alpha)
    feats = item_features(items, n_items)
    known = ~new & (engaged >= MIN_BUYERS)
    Yhat, _ = content_to_factors(feats, Yo, known)
    pos_new = {u: s & set(np.flatnonzero(new).tolist()) for u, s in pos.items()}
    pos_new = {u: s for u, s in pos_new.items() if s}
    rn = np.array(sorted(pos_new))
    same_lang = (users["language"][rn, None] == items["language"][None, :]).astype(np.float64)
    content, _ = content_scores(w_old[rn], feats)
    pred = Xo[rn] @ Yhat.T
    pred01 = (pred - pred.min(axis=1, keepdims=True)) / (np.ptp(pred, axis=1, keepdims=True) + 1e-12)
    print(f"\n'New for you' (Chapter 3's history-masking test): {new.sum()} videos, {len(rn):,} users")
    rng = np.random.default_rng(1)                          # the same draws as Chapter 3
    for name, base in (("random order", 0.0), ("in your language, random order", same_lang),
                       ("in your language, by content (Chapter 3)", same_lang + 0.5 * content),
                       ("predicted vectors", pred01),
                       ("in your language, by predicted vectors", same_lang + 0.5 * pred01)):
        runs = [hit_rate(top_k(base + 1e-3 * rng.random((len(rn), n_items)), used[rn], new), rn, pos_new)
                for _ in range(20)]
        print(f"  {name:<42} hit rate@10 {np.mean(runs):.3f}")
