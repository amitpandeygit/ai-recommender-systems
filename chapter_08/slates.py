"""Listings 8.5 and 8.6 - Building the row: diversity and business rules (sections 8.6 and 8.7).

The heavy ranker scores each video on its own. The row of 10 is built from the 50
candidates the shipped pre-ranker keeps, in one of these ways:

  score order     - the top 10 by expected satisfied plays (Chapter 7)
  MMR             - maximal marginal relevance (Carbonell and Goldstein, 1998): each next
                    video maximises lam x relevance - (1 - lam) x its highest cosine
                    similarity to a video already chosen, with iALS video vectors
  whole row       - a greedy builder for one stated row objective: each next video
                    maximises relevance x beta ** (videos of its category already in the row)
  creator limit   - at most `cap` videos per creator
  new-video slot  - one slot holds the best "New for you" video (Chapter 6's source for
                    videos with too little history to be in the pool)
  ad slots        - sponsored shop products at fixed slots, or at slot 3 only when the
                    ad's expected revenue is worth the satisfied plays it pushes out

Chapter 2's simulator scores every slot on its own, so a row of similar videos loses
nothing there. Section 8.6's planted test (slate_data.BOREDOM) changes that one
thing: each earlier video of the same category in the row multiplies a video's
chance of satisfying by 0.75. Results are given in both worlds. The bids and the
sponsored products are synthetic (seed cfg.seed + 803).

Needs Python 3.12 or later and PyTorch. Run from the repository root (about three minutes):
    python chapter_08/slates.py
"""
import os
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "2"              # NumPy's linear algebra: the same thread count on every run
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from slate_data import (BOREDOM, DAY, KEPT, ROW, order_by, online, paired, row_metrics, rules, slate_outcomes,  # noqa: E402
                        split, world)
from metrics import diversity, ndcg_at_k  # noqa: E402
from ranking_data import PRIOR, impressions  # noqa: E402
from implicit_scores import prepare  # noqa: E402
from similarity_recs import NO, content_scores, item_features, top_k  # noqa: E402

LAMBDAS = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5)
BETAS = (1.0, 0.9, 0.75, 0.5)


def greedy(kept, rel, step_score, k=ROW):
    """Pick k videos one at a time; step_score(rel, chosen_mask, picked_so_far) gives
    each candidate's value given what is already in the row."""
    n, m = kept.shape
    chosen = np.zeros((n, m), dtype=bool)
    out = np.full((n, k), -1)
    rows = np.arange(n)
    for slot in range(k):
        s = np.where((kept >= 0) & ~chosen & np.isfinite(rel), step_score(chosen), -np.inf)
        best = np.argmax(s, axis=1)
        ok = np.isfinite(s[rows, best])
        out[ok, slot] = kept[rows[ok], best[ok]]
        chosen[rows[ok], best[ok]] = True
    return out


def mmr(kept, rel, vec, lam, k=ROW):
    """Listing 8.5. Maximal marginal relevance over each user's kept candidates.
    Relevance is the heavy ranker's score divided by the user's highest one."""
    r = rel / np.max(np.where(np.isfinite(rel), rel, -np.inf), axis=1, keepdims=True)
    v = vec[np.maximum(kept, 0)]
    v = v / np.maximum(np.linalg.norm(v, axis=2, keepdims=True), 1e-12)
    sim = np.einsum("nad,nbd->nab", v, v)                         # cosine between candidates

    def step(chosen):
        most = np.where(chosen[:, None, :], sim, -np.inf).max(axis=2)
        most = np.where(chosen.any(axis=1, keepdims=True), most, 0.0)   # the first pick has nothing to repeat
        return lam * r - (1 - lam) * most
    return greedy(kept, r, step, k)


def whole_row(kept, rel, category, beta, k=ROW):
    """A greedy builder for the row objective sum of relevance x beta ** repeats."""
    cat = category[np.maximum(kept, 0)]

    def step(chosen):
        same = ((cat[:, :, None] == cat[:, None, :]) & chosen[:, None, :]).sum(axis=2)
        return rel * beta ** same
    return greedy(kept, rel, step, k)


def creator_limit(kept, rel, creator, cap, k=ROW):
    """Listing 8.6 (first part). Score order, but at most `cap` videos per creator."""
    cr = creator[np.maximum(kept, 0)]

    def step(chosen):
        count = ((cr[:, :, None] == cr[:, None, :]) & chosen[:, None, :]).sum(axis=2)
        return np.where(count < cap, rel, -np.inf)
    return greedy(kept, rel, step, k)


def insert_at(top, extra, slot):
    """Listing 8.6 (second part). Put extra[u] at `slot` (1 to 10) of user u's row and
    push the videos below it down one place; the last one leaves the row."""
    out = top.copy()
    has = extra >= 0
    s = slot - 1
    out[has, s + 1:] = top[has, s:-1]
    out[has, s] = extra[has]
    return out


def look_curve(W):
    """Each slot's click chance relative to slot 1, measured on exploratory slots
    (random videos) of days 0 to 23: the system's own estimate of attention."""
    early = impressions(W["events"], W["served"], W["users"], W["items"], 0, 23 * DAY, 23 * DAY)
    ex = early["explored"]
    ctr = np.array([early["click"][ex & (early["position"] == k)].mean() for k in range(1, ROW + 1)])
    return ctr / ctr[0]


def ad_setup(W, cfg, R, explored_only=True):
    """Sponsored products (20%, synthetic bids in coins per click) and Tesselo's ad
    click model: a logistic regression on the product's smoothed click rate (logit)
    and the iALS score, with one term per slot. It is fitted on the product
    impressions of days 9 to 23, each week with the features of the snapshot before
    it; by default only on exploratory slots, where the product was drawn at random."""
    items, users = W["items"], W["users"]
    rng = np.random.default_rng(cfg.seed + 803)
    product = items["is_product"]
    sponsored = product & (rng.random(len(product)) < 0.20)
    bid = np.where(sponsored, rng.lognormal(0.0, 0.5, len(product)), 0.0)
    flipped = dict(items, is_product=~product)                  # impressions() keeps videos; flipped, it keeps products
    Xs, ys = [], []
    for start, day in ((9, 9), (16, 16)):
        P = impressions(W["events"], W["served"], users, flipped, start * DAY, (start + 7) * DAY, (start + 7) * DAY)
        keep = P["explored"] if explored_only else np.ones(len(P["item"]), dtype=bool)
        S = W["snaps"][day]
        u, i = P["user"][keep], P["item"][keep]
        X = np.stack([np.ones(len(i)), logit(product_ctr(W, day)[i]), np.einsum("nk,nk->n", S["X"][u], S["Y"][i])],
                     axis=1)
        Xs.append(np.concatenate([X, (P["position"][keep][:, None] == np.arange(2, ROW + 1)[None, :])], axis=1))
        ys.append(P["click"][keep].astype(float))
    X, y = np.concatenate(Xs), np.concatenate(ys)
    coef = fit_logistic(X, y)
    ok = sponsored & ~R["out_of_stock"]                          # section 8.2: never advertise what cannot be bought
    return {"bid": bid, "ok": ok, "sponsored": sponsored, "coef": coef, "rows": len(y), "clicks": int(y.sum())}


def logit(p):
    return np.log(p / (1 - p))


def product_ctr(W, day):
    """Each product's click rate from the logs before `day`, smoothed towards the mean."""
    product = W["items"]["is_product"]
    d = prepare(W["events"], W["users"], day * DAY)
    views = np.bincount(d["item"][d["event"] == "viewable"], minlength=len(product)).astype(float)
    clicks = np.bincount(d["item"][d["event"] == "click"], minlength=len(product)).astype(float)
    mean = clicks[product].sum() / views[product].sum()
    return (clicks + PRIOR * mean) / (views + PRIOR)


def fit_logistic(X, y, ridge=1e-3, iters=25):
    """Logistic regression by Newton's method with a small ridge penalty."""
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ w)))
        H = (X * (p * (1 - p))[:, None]).T @ X + ridge * np.eye(X.shape[1])
        w -= np.linalg.solve(H, X.T @ (p - y) + ridge * w)
    return w


def best_ads(W, S, rows, ads, n=2):
    """Each user's n best ads by estimated value at slot 1, P(click) x bid, among the
    sponsored, in-stock products they have not used."""
    c = ads["coef"]
    lc = logit(product_ctr(W, 23))
    z = c[0] + c[1] * lc[None, :] + c[2] * (S["X"][rows] @ S["Y"].T)
    value = np.where(ads["ok"][None, :] & ~S["used"][rows], 1 / (1 + np.exp(-z)) * ads["bid"][None, :], -np.inf)
    best = np.argsort(-value, axis=1, kind="stable")[:, :n]
    v = np.take_along_axis(value, best, axis=1)
    return np.where(np.isfinite(v), best, -1), np.where(np.isfinite(v), v, 0.0)


def revenue(top, rows, W, bid):
    """Expected ad revenue per row: P(click) on each product slot x its bid (truth)."""
    click, _ = slate_outcomes(top, rows, W["users"], W["items"], W["cfg"])
    i = np.maximum(top, 0)
    return np.where((top >= 0) & W["items"]["is_product"][i], click * bid[i], 0.0).sum(axis=1)


def new_for_you(W, S, rows, R):
    """Chapter 6's "New for you" source: videos below the history threshold, in the
    user's language first, then by content similarity to their history; the rules
    of section 8.2 applied."""
    items, users = W["items"], W["users"]
    video = W["video"]
    feats = item_features(items, len(items["vec"]))
    new_ok = video & ~S["ok"] & ~R["removed"]
    same = (users["language"][rows, None] == items["language"][None, :]).astype(np.float64)
    score = np.where(new_ok[None, :], same + 0.5 * content_scores(S["weight"][rows], feats)[0], NO)
    for k, u in enumerate(rows):                                 # the user-specific rules
        ban = np.isin(items["creator"], R["blocked_creators"][u]) | np.isin(np.arange(len(video)),
                                                                            R["unlicensed"][R["region"][u]])
        if R["teen"][u]:
            ban |= R["adult"]
        score[k, ban] = NO
    return top_k(score, S["used"][rows], video, 1)[:, 0], new_ok


def build(W, sp, who, method, x):
    """The row for every user of sp[who], built from the kept candidates."""
    items = W["items"]
    kept, rel = sp[f"kept_{who}"], np.where(sp[f"kept_{who}"] >= 0, sp[f"heavy_{who}"], -np.inf)
    if method == "MMR":
        return mmr(kept, rel, sp["S"]["Y"], x)
    if method == "whole row":
        return whole_row(kept, rel, items["category"], x)
    return creator_limit(kept, rel, items["creator"], x)


def choose_diversity(W, val, verbose=False):
    """The rule written before the run, on the validation week (offline only): for
    each method, the most diverse setting whose NDCG@10 (test users) is within 5% of
    score order; then the method whose rows are more varied (all consenting users)."""
    items = W["items"]
    if verbose:
        print("Validation week (offline only): NDCG@10 of the test users' rows, category variety of all "
              "consenting users' rows")
    nd0, picks = None, {}
    for method, xs in (("MMR", LAMBDAS), ("whole row", BETAS)):
        for x in xs:
            nd = np.mean([ndcg_at_k(t.tolist(), {i: 1.0 for i in val["pos"][u]})
                          for u, t in zip(val["rows"], build(W, val, "rows", method, x))])
            var = np.nanmean([diversity(t.tolist(), items["category"]) for t in build(W, val, "aud", method, x)])
            nd0 = nd if nd0 is None else nd0
            ok = nd >= 0.95 * nd0
            if ok:
                picks[method] = (x, var)
            if verbose:
                print(f"  {method} {x:<6}NDCG@10 {nd:.4f}  variety {var:.3f}" + ("" if ok else "   (more than 5% below)"))
    method = max(picks, key=lambda m: picks[m][1])
    if verbose:
        print(f"Rule written before the run: per method, the most diverse setting whose validation NDCG@10 is within "
              f"5% of score order (MMR lam = {picks['MMR'][0]}, whole row beta = {picks['whole row'][0]}); then the "
              f"more varied of the two: {method}, {picks[method][0]}")
    return method, picks[method][0]


def ad_allocation(s_top, look, value, e):
    """Listing 8.6 (third part). For each user, the slot k that maximises
    e x (expected ad revenue at k) - (satisfied plays the ad pushes out), or no ad if
    nothing is positive. Pushing out: every video from slot k down moves one slot
    lower, so it loses (w_j - w_(j+1)) x its score, with w_11 = 0."""
    w = np.r_[look, 0.0]
    drop = (w[:-1] - w[1:])[None, :] * s_top                     # loss of moving the video in slot j down one
    loss = np.cumsum(drop[:, ::-1], axis=1)[:, ::-1]             # an ad at slot k moves slots k..10
    gain = e * look[None, :] * value[:, None] - loss
    best = np.argmax(gain, axis=1)
    return np.where(gain[np.arange(len(gain)), best] > 0, best + 1, 0)   # 0: no ad


def table_row(label, res, base=None, base_b=None, first=False):
    if first:
        print(f"  {'':<28}{'NDCG@10':>8}{'variety':>8}{'coverage':>9}{'top-10% cr.':>12}{'online':>8}"
              f"{'against':>17}{'boredom':>9}{'against':>17}")
    d = "" if base is None else "{:+.4f} ({:.4f})".format(*paired(res["online"], base))
    db = "" if base_b is None else "{:+.4f} ({:.4f})".format(*paired(res["online, boredom"], base_b))
    print(f"  {label:<28}{res['NDCG@10']:>8.4f}{res['variety']:>8.3f}{res['coverage']:>9.1%}"
          f"{res['top-10% creators']:>12.1%}{res['online'].mean():>8.4f}{d:>17}{res['online, boredom'].mean():>9.4f}"
          f"{db:>17}")


if __name__ == "__main__":
    t0 = time.time()
    W = world()
    cfg, users, items = W["cfg"], W["users"], W["items"]
    val, test = split(W, "validation"), split(W, "test")
    R = rules(cfg, users, items)
    print(f"Shipped pipeline rebuilt  [{time.time() - t0:.0f} s]")

    # ---- section 8.6: diversity, validation week (offline only) and test week
    print("\nSection 8.6: the planted boredom test multiplies a video's chance of satisfying by "
          f"{BOREDOM} for each earlier video of its category in the row")
    picks = choose_diversity(W, val, verbose=True)
    print(f"\nTest week: rows built from the {KEPT} kept candidates; offline on {len(test['rows']):,} test users, "
          f"the rest on {len(test['aud']):,} consenting users; online in Chapter 2's world and in the boredom test")
    base = row_metrics(order_by(test["kept_rows"], test["heavy_rows"]),
                       order_by(test["kept_aud"], test["heavy_aud"]), test, W)
    table_row("score order", base, first=True)
    for method, xs in (("MMR", LAMBDAS[1:]), ("whole row", BETAS[1:])):
        for x in xs:
            res = row_metrics(build(W, test, "rows", method, x), build(W, test, "aud", method, x), test, W)
            table_row(f"{method}, {'lam' if method == 'MMR' else 'beta'} = {x}", res, base["online"],
                      base["online, boredom"])
    print(f"  (against: paired difference from score order over the same users, standard error in brackets)  "
          f"[{time.time() - t0:.0f} s]")

    # ---- section 8.7: business rules
    print("\nSection 8.7: business rules on the test week")
    print("Creator limit")
    table_row("no limit (score order)", base, first=True)
    for cap in (2, 1):
        res = row_metrics(build(W, test, "rows", "cap", cap), build(W, test, "aud", "cap", cap), test, W)
        table_row(f"at most {cap} per creator", res, base["online"], base["online, boredom"])
    n_creator = np.mean([len(set(items["creator"][t[t >= 0]].tolist())) for t in
                         order_by(test["kept_aud"], test["heavy_aud"])])
    print(f"  score order shows {n_creator:.2f} different creators per row on average")

    print("\nNew-video slot: the best 'New for you' video at one slot, the rest in score order")
    S = test["S"]
    new_a, new_ok = new_for_you(W, S, test["aud"], R)
    new_r, _ = new_for_you(W, S, test["rows"], R)
    top_a0 = order_by(test["kept_aud"], test["heavy_aud"])
    top_r0 = order_by(test["kept_rows"], test["heavy_rows"])
    print(f"  videos below the history threshold: {new_ok.sum():,} of {W['video'].sum():,}; users with a candidate: "
          f"{np.mean(new_a >= 0):.1%}")
    print(f"  {'':<28}{'online':>8}{'against':>17}{'from the new video':>20}{'new videos shown':>18}")
    for slot in (2, 4, 8):
        ta = insert_at(top_a0, new_a, slot)
        on = online(ta, test["aud"], W)
        _, play = slate_outcomes(ta, test["aud"], users, items, cfg)
        from_new = play[:, slot - 1].mean()
        d, se = paired(on, base["online"])
        print(f"  {'slot ' + str(slot):<28}{on.mean():>8.4f}{f'{d:+.4f} ({se:.4f})':>17}{from_new:>20.4f}"
              f"{len(set(new_a[new_a >= 0].tolist())):>18,}")

    print("\nAd slots: sponsored products (synthetic bids), each user's in-stock ad with the highest estimated value")
    look = look_curve(W)
    ads = ad_setup(W, cfg, R)
    print("  attention by slot, relative to slot 1, from exploratory slots: " + " ".join(f"{x:.3f}" for x in look))
    print(f"  sponsored products: {ads['sponsored'].sum():,}, in stock: {ads['ok'].sum():,}; bids: median "
          f"{np.median(ads['bid'][ads['sponsored']]):.2f} coins per click")
    for label, only in (("every product impression", False), ("exploratory slots only", True)):
        m = ad_setup(W, cfg, R, explored_only=only)
        t2, v2 = best_ads(W, S, test["aud"], m)
        top = insert_at(order_by(test["kept_aud"], test["heavy_aud"]), t2[:, 0], 3)
        print(f"  ad click model fitted on {label} ({m['rows']:,} rows, {m['clicks']:,} clicks): "
              f"{m['coef'][0]:+.3f} {m['coef'][1]:+.3f} x logit(click rate) {m['coef'][2]:+.3f} x iALS score; "
              f"an ad at slot 3: estimated revenue {(look[2] * v2[:, 0]).mean():.4f}, realised "
              f"{revenue(top, test['aud'], W, m['bid']).mean():.4f} coins per row")
    two_ads, two_values = best_ads(W, S, test["aud"], ads)
    ad, ad_value = two_ads[:, 0], two_values[:, 0]
    rel = order_by(test["kept_aud"], test["heavy_aud"])
    s_top = np.take_along_axis(np.where(test["kept_aud"] >= 0, test["heavy_aud"], -np.inf),
                               np.argsort(-np.where(test["kept_aud"] >= 0, test["heavy_aud"], -np.inf), axis=1,
                                          kind="stable")[:, :ROW], axis=1)
    s_top = np.where(np.isfinite(s_top), s_top, 0.0)
    w = np.r_[look, 0.0]
    loss3 = ((w[2:ROW] - w[3:])[None, :] * s_top[:, 2:]).sum(axis=1)   # what an ad at slot 3 pushes out
    print(f"  estimates for an ad at slot 3, mean over users: revenue {look[2] * ad_value.mean():.4f} coins, "
          f"satisfied plays pushed out {loss3.mean():.4f}")
    print(f"  {'':<34}{'rows with an ad':>16}{'mean slot':>10}{'online':>8}{'against':>17}{'revenue / row':>15}")
    policies = [("no ads", None, None), ("fixed: slot 3", (3,), None), ("fixed: slots 3 and 8", (3, 8), None)]
    policies += [(f"best slot, exchange rate {e}", None, e) for e in (0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.3)]
    for label, slots, e in policies:
        top = rel.copy()
        if e is not None:
            slot = ad_allocation(s_top, look, ad_value, e)
            for k in range(1, ROW + 1):
                m = (slot == k) & (ad >= 0)
                top[m] = insert_at(top[m], ad[m], k)
            shown, mean_slot = (slot > 0) & (ad >= 0), slot[(slot > 0) & (ad >= 0)]
        elif slots is not None:
            for k, s_ in enumerate(slots):                            # the best ad at slot 3, the next at slot 8
                top = insert_at(top, two_ads[:, k], s_)
            shown, mean_slot = ad >= 0, np.full((ad >= 0).sum(), np.mean(slots))
        else:
            shown, mean_slot = np.zeros(len(top), dtype=bool), np.zeros(1)
        on = online(top, test["aud"], W)
        d, se = paired(on, base["online"])
        print(f"  {label:<34}{shown.mean():>16.1%}{(np.mean(mean_slot) if len(mean_slot) else 0):>10.2f}{on.mean():>8.4f}"
              f"{f'{d:+.4f} ({se:.4f})':>17}{revenue(top, test['aud'], W, ads['bid']).mean():>15.4f}")
    print(f"[{time.time() - t0:.0f} s]")
