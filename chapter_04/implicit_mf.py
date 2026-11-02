"""Listings 4.2 to 4.6 - Matrix factorisation for implicit feedback (sections 4.4 to 4.8).

Section 4.4: iALS (Hu, Koren and Volinsky, 2008), trained by alternating least
             squares, and BPR (Rendle and colleagues, 2009), trained by SGD on
             pairs, with uniform or popularity-weighted negative samples.
Section 4.5: what the embeddings learn, checked against the hidden tastes.
Section 4.6: "Because you watched" from item vectors.
Section 4.7: cold start - fold-in for a user with a few actions, and item
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
from similarity_recs import (MIN_BUYERS, NO, content_scores, group_popularity,  # noqa: E402
                             hit_rate, item_features, item_neighbours, lists, mcnemar_exact,
                             test_positives, top_k, user_item_scores, cf_scores)


# ---------------------------------------------------------------- section 4.4: iALS
def ials_solve(R, Y, reg, alpha):
    """One half-step of iALS: the best vector for every row of R, with Y fixed.

    Row u minimises  sum over all items i of c_ui (p_ui - x_u . y_i)^2 + reg |x_u|^2,
    with p_ui = 1 and c_ui = 1 + alpha * r_ui if r_ui > 0, else p_ui = 0 and c_ui = 1.
    Solution: x_u = (Y'Y + Y_u' (C_u - I) Y_u + reg I)^-1  Y_u' c_u.
    Y'Y is shared by all rows, so each row only touches its own few items.
    """
    k = Y.shape[1]
    YtY = Y.T @ Y
    X = np.zeros((R.shape[0], k))
    rows, cols = np.nonzero(R)
    start = np.searchsorted(rows, np.arange(R.shape[0] + 1))
    for u in range(R.shape[0]):
        idx = cols[start[u]:start[u + 1]]
        if len(idx) == 0:
            continue                                         # no history: no vector
        Yu, c = Y[idx], 1.0 + alpha * R[u, idx]
        A = YtY + (Yu.T * (c - 1.0)) @ Yu + reg * np.eye(k)
        X[u] = np.linalg.solve(A, Yu.T @ c)
    return X


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
    ln sigmoid(score(u, i) - score(u, j)) minus an L2 penalty. Negatives are drawn
    in proportion to (engaged users + 1) ** neg_power: 0 is uniform, 1 follows
    popularity.
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
            pu, d = P[u].copy(), Q[i] - Q[j]
            g = 1.0 / (1.0 + np.exp(b[i] - b[j] + np.einsum("nk,nk->n", pu, d)))  # sigmoid(-x)
            np.add.at(P, u, lr * (g[:, None] * d - reg * pu))
            np.add.at(Q, i, lr * (g[:, None] * pu - reg * Q[i]))
            np.add.at(Q, j, lr * (-g[:, None] * pu - reg * Q[j]))
            np.add.at(b, i, lr * g)
            np.add.at(b, j, -lr * g)
    return P, Q, b


# ---------------------------------------------------------------- sections 4.5 to 4.7
def taste_correlation(score, users, items, history_users, eligible_items, n=20_000, seed=3):
    """Correlation between a model's score and the hidden match, on random
    (user, item) pairs. Possible only in a simulation."""
    rng = np.random.default_rng(seed)
    u = rng.choice(history_users, n)
    i = rng.choice(np.flatnonzero(eligible_items), n)
    true = np.einsum("nd,nd->n", users["vec"][u], items["vec"][i])
    s = score(u, i)
    ok = np.isfinite(s)                                      # skip pairs a method may not score
    return float(np.corrcoef(s[ok], true[ok])[0, 1])


def cosine_neighbours(Y, item, k=10, allowed=None):
    """The k items whose vectors point most nearly the same way as item's."""
    unit = Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
    sim = np.round(unit @ unit[item], 9)
    sim[item] = -np.inf
    if allowed is not None:
        sim[~allowed] = -np.inf
    return np.argsort(-sim, kind="stable")[:k]


def fold_in(Y, items_used, weights, reg, alpha):
    """A vector for a new or changed user, with item vectors fixed: one iALS step."""
    R = np.zeros((1, Y.shape[0]))
    R[0, items_used] = weights
    return ials_solve(R, Y, reg, alpha)[0]


def content_to_factors(feats, Y, known, ridge=1.0):
    """Ridge regression from content features to item vectors, fitted on items
    with history; returns predicted vectors for every item."""
    F = feats[known]
    W = np.linalg.solve(F.T @ F + ridge * np.eye(F.shape[1]), F.T @ Y[known])
    return feats @ W


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
    bi2, bj2 = bi + lr * g, bj - lr * g
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
                tc = taste_correlation(lambda u, i: np.einsum("nk,nk->n", X[u], Y[i]),
                                       users, items, hist16, ok16)
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
            tc = taste_correlation(lambda u, i: np.einsum("nk,nk->n", P[u], Q[i]) + b[i],
                                   users, items, hist16, ok16)
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
          f"(median {np.median(history[hist]):.0f} items); weights: median {np.median(v):.2f}, "
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
    taste = {"popular in your language": lambda u, i: lang_pop[u, i],
             "iALS": lambda u, i: np.einsum("nk,nk->n", X[u], Y[i]),
             f"BPR, {'uniform' if bpr_best[2] == 0 else 'popular'} negatives":
                 lambda u, i: np.einsum("nk,nk->n", P[u], Q[i]) + b[i],
             f"BPR, {other} negatives": lambda u, i: np.einsum("nk,nk->n", Pp[u], Qp[i]) + bp[i]}
    groups = {"all": rows, "0-4 items": rows[history[rows] <= 4], "5+ items": rows[history[rows] >= 5]}
    print(f"\nTest week (train to day 23, test days 23-30), videos only, hit rate@10")
    print(f"{'':<34}" + "".join(f"{g:>11}" for g in groups) + f"{'full lists':>12}{'taste corr':>12}")
    print(f"{'users':<34}" + "".join(f"{len(v):>11,}" for v in groups.values()))
    tops = {}
    for name, s in scores.items():
        top = tops[name] = lists(s, rows, used, video)
        tc = f"{taste_correlation(taste[name], users, items, hist, ok):>12.2f}" if name in taste else ""
        print(f"{name:<34}" + "".join(f"{hit_rate(top, rows, pos, v):>11.3f}" for v in groups.values())
              + f"{(top[:, -1] >= 0).mean():>12.0%}" + tc)
    hit = {n: {u for u, t in zip(rows.tolist(), tops[n]) if pos[u] & set(t.tolist())}
           for n in ("iALS, then popular in language", "CF, then popular in language")}
    a_only = len(hit["iALS, then popular in language"] - hit["CF, then popular in language"])
    b_only = len(hit["CF, then popular in language"] - hit["iALS, then popular in language"])
    print(f"All users: hit only with iALS chain {a_only}, only with CF chain {b_only}; "
          f"exact McNemar p = {mcnemar_exact(a_only, b_only):.2g}")
    print(f"Taste correlation of Chapter 3 CF on the pairs it can score: "
          f"{taste_correlation(lambda u, i: cf[u, i], users, items, hist, ok):.2f}")

    # ---- section 4.6: neighbours from item vectors
    sample = np.flatnonzero(ok)[::25]
    def same_category(neigh):
        return np.mean([np.mean(items["category"][neigh(i)] == items["category"][i]) for i in sample])
    sim = item_neighbours(weight, k=10, strength=0.0)
    hidden = items["vec"].astype(np.float64)
    print(f"\nShare of each video's 10 nearest videos in the same category ({len(sample)} videos):")
    print(f"  iALS item vectors (cosine): {same_category(lambda i: cosine_neighbours(Y, i, 10, ok)):.0%}")
    print(f"  Chapter 3 item-based CF:    "
          f"{same_category(lambda i: np.argsort(-np.where(ok, sim[i], -np.inf), kind='stable')[:10]):.0%}")
    print(f"  hidden vectors (truth):     {same_category(lambda i: cosine_neighbours(hidden, i, 10, ok)):.0%}")
    print(f"  random videos (chance):     {np.mean([np.mean(items['category'][ok] == items['category'][i]) for i in sample]):.0%}")
    # dot product (vector length counts) against cosine (direction only) for the user's list
    unit = Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
    cos_s = eligible_and_scored(X @ unit.T, weight, ok)
    print(f"  Item vector length vs log(1 + engaged users), correlation: "
          f"{np.corrcoef(np.linalg.norm(Y[ok], axis=1), np.log1p(engaged[ok]))[0, 1]:.2f}")
    print(f"  User's list by dot product: hit rate@10 {hit_rate(tops['iALS'], rows, pos):.3f}; "
          f"by cosine (item vectors set to length 1): {hit_rate(lists(cos_s, rows, used, video), rows, pos):.3f}")
    anchor = sample[0]
    print(f"  Because you watched video {anchor} ({CATEGORIES[items['category'][anchor]]}): "
          + ", ".join(f"{j} ({CATEGORIES[items['category'][j]]})" for j in cosine_neighbours(Y, anchor, 5, ok)))

    # ---- section 4.7: fold-in from a user's first n engaged items
    many = rows[history[rows] >= 5]
    print(f"\nFold-in for users with 5+ items ({len(many):,}), using only n of their items, chosen at random:")
    for n in (1, 3, 5):
        V = np.zeros((len(many), k))
        pick = np.random.default_rng(4)
        for r, u in enumerate(many):
            idx = np.sort(pick.choice(np.flatnonzero(weight[u] > 0), n, replace=False))
            V[r] = fold_in(Y, idx, weight[u, idx], reg, alpha)
        top = top_k(np.where(ok[None, :], V @ Y.T, NO), used[many], video)
        print(f"  n = {n}: hit rate@10 {hit_rate(top, many, pos):.3f}")
    print(f"  all items (the trained vectors): "
          f"{hit_rate(top_k(np.where(ok[None, :], X[many] @ Y.T, NO), used[many], video), many, pos):.3f}")

    # ---- section 4.7: videos with hidden history get vectors from content
    new = (np.random.default_rng(cfg.seed + 303).random(n_items) < 0.10) & video   # as in Chapter 3
    w_old = np.where(new[None, :], 0.0, weight)
    Xo, Yo = ials(w_old, k, reg, alpha)
    feats = item_features(items, n_items)
    known = ~new & (engaged >= MIN_BUYERS)
    Yhat = content_to_factors(feats, Yo, known)
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
