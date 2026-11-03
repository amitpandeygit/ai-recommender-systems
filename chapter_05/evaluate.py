"""Listings 5.4 to 5.6 - Evaluating Tesselo's recommenders (sections 5.2 to 5.8).

  5.2  business metrics: the conversion paths in Tesselo's logs, and cross-sell in the shop
  5.3  ranking metrics for every baseline on the "Because you watched" video row,
       per language, and calibration of iALS scores with Platt scaling
  5.4  beyond accuracy: diversity, novelty, serendipity, coverage, concentration
  5.5  the baselines every model must beat
  5.6  three ways to split the data, and the sampled-metrics trap
  5.7  a simulated online test, using the hidden behaviour model of Chapter 2
  5.8  a simulated A/B test and the sample size it needs

The recommenders are Chapter 3's and Chapter 4's, with the settings chosen there.
Every number is a toy setting chosen to make effects visible, not a measurement.

Needs Python 3.12 or later. Run from the repository root (about ten minutes):
    python chapter_05/evaluate.py
"""
import sys
from math import erf, sqrt
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for folder in ("chapter_03", "chapter_04"):
    sys.path.insert(0, str(ROOT / folder))
from tesselo_shop import CATEGORIES, DAY, LANGUAGES, Config, build  # noqa: E402
from tesselo_data import sigmoid  # noqa: E402
from similarity_recs import (MIN_BUYERS, NO, cf_scores, group_popularity, item_neighbours,  # noqa: E402
                             test_positives, top_k, top_k_chain, user_item_scores)
from implicit_scores import prepare  # noqa: E402
from implicit_mf import bpr, eligible_and_scored, ials  # noqa: E402
from metrics import (auc, average_precision, brier, diversity, ece, exposure, fit_platt,  # noqa: E402
                     gini, hit_at_k, log_loss, ndcg_at_k, novelty, precision_at_k, recall_at_k,
                     reciprocal_rank, reliability_by_rank, serendipity)

K, ALPHA, REG = 16, 10.0, 1.0            # Chapter 4's iALS settings
HITS = ("complete", "like")


# ---------------------------------------------------------------- shared helpers
def ranked(score_fn, rows, used, eligible, k=10):
    """Lists for the given rows. score_fn(rows) returns a score matrix for those
    rows, or a tuple of them for a fallback chain (Chapter 3's top_k_chain).
    Scores are built only for the rows asked for, to keep memory small."""
    s = score_fn(rows)
    if isinstance(s, tuple):
        return top_k_chain(list(s), used[rows], eligible, k)
    return top_k(s, used[rows], eligible, k)


def ranking_table(top, rows, pos, used, video):
    """Mean of every section 5.3 metric over users. top holds each user's full
    ordering (best first); the first 10 entries are the list the user would see."""
    out = {m: [] for m in ("hit@10", "P@10", "R@10", "MRR", "MAP@10", "NDCG@10", "AUC")}
    for u, t in zip(rows, top):
        rel, t = pos[u], t.tolist()
        out["hit@10"].append(hit_at_k(t, rel))
        out["P@10"].append(precision_at_k(t, rel))
        out["R@10"].append(recall_at_k(t, rel))
        out["MRR"].append(reciprocal_rank(t, rel))
        out["MAP@10"].append(average_precision(t, rel))
        out["NDCG@10"].append(ndcg_at_k(t, {i: 1.0 for i in rel}))
        out["AUC"].append(auc(t, rel, np.flatnonzero(video & ~used[u]).tolist()))
    return {m: float(np.nanmean(v)) for m, v in out.items()}


def expected_outcomes(top, rows, users, items, cfg):
    """Section 5.7: expected clicks and satisfied plays for each user's list,
    from Chapter 2's hidden behaviour model (simulation only).

    A user scrolls on from one slot to the next with probability scroll_keep, so
    slot r is on screen with probability keep^(r-1) and is the slot just glimpsed
    past with probability keep^(r-2) (1 - keep). P(look) is then divided by r^decay."""
    r = np.arange(1, top.shape[1] + 1)
    keep = cfg.scroll_keep
    on = keep ** (r - 1)
    glimpse = np.where(r >= 2, keep ** np.maximum(r - 2, 0) * (1 - keep), 0.0)
    p_look = (on + cfg.glimpse_look * glimpse) / r ** cfg.look_decay
    filled = top >= 0
    i = np.where(filled, top, 0)
    match = np.einsum("nd,nkd->nk", users["vec"][rows].astype(np.float64), items["vec"][i].astype(np.float64))
    bait = items["clickbait"][i]
    attract = sigmoid(5 * match - 3.5 + bait)
    satisfy = sigmoid(5 * match - 2.0 - 2 * bait)
    click = np.where(filled, p_look * attract, 0.0)
    sat = click * satisfy
    return click, sat


def simulate_week(top, rows, users, items, cfg, rng, impressions=7):
    """Section 5.8: draw a week of behaviour for each user, who sees the same list
    `impressions` times. Returns satisfied plays per user (a random outcome)."""
    total = np.zeros(len(rows))
    filled = top >= 0
    i = np.where(filled, top, 0)
    match = np.einsum("nd,nkd->nk", users["vec"][rows].astype(np.float64), items["vec"][i].astype(np.float64))
    bait = items["clickbait"][i]
    attract, satisfy = sigmoid(5 * match - 3.5 + bait), sigmoid(5 * match - 2.0 - 2 * bait)
    r = np.arange(1, top.shape[1] + 1)
    for _ in range(impressions):
        depth = 1 + np.minimum(rng.geometric(1 - cfg.scroll_keep, len(rows)) - 1, top.shape[1] - 1)
        on = r[None, :] <= depth[:, None]
        glimpse = r[None, :] == depth[:, None] + 1
        p_look = np.where(on, 1.0, np.where(glimpse, cfg.glimpse_look, 0.0)) / r ** cfg.look_decay
        u = rng.random(top.shape)
        looked = u < p_look
        clicked = looked & (rng.random(top.shape) < attract)
        satisfied = clicked & (rng.random(top.shape) < satisfy) & filled
        total += satisfied.sum(axis=1)
    return total


def first_hit_time(events, users, cutoff, video):
    """(user, video) -> age in days of the first completion or like before cutoff."""
    d = prepare(events, users, cutoff)
    m = np.isin(d["event"], HITS) & video[d["item"]]
    first = {}
    for u, i, a in zip(d["user"][m].tolist(), d["item"][m].tolist(), d["age_days"][m].tolist()):
        first[(u, i)] = max(first.get((u, i), a), a)
    return first


def split_scores(weight, users, k=K):
    """The three split-test methods, trained on one training matrix."""
    pop = group_popularity(weight, np.zeros(len(weight), dtype=np.int64), 1)[0]
    by_lang = group_popularity(weight, users["language"], len(LANGUAGES))
    ok = (weight > 0).sum(axis=0) >= MIN_BUYERS
    X, Y = ials(weight, k, REG, ALPHA)
    lang = lambda r: by_lang[users["language"][r]]                          # noqa: E731
    return {"popular worldwide": lambda r: np.broadcast_to(pop, (len(r), len(pop))),
            "popular in your language": lang,
            "iALS, then popular in language": lambda r: (eligible_and_scored(X[r] @ Y.T, weight[r], ok), lang(r))}


def print_rank_table(title, table, metrics):
    print(title)
    print(f"  {'':<34}" + "".join(f"{m:>9}" for m in metrics))
    for name, t in table.items():
        print(f"  {name:<34}" + "".join(f"{t[m]:>9.3f}" for m in metrics))


if __name__ == "__main__":
    cfg = Config()
    users, items, served, events, orders, truth = build(cfg)
    n_users, n_items = cfg.num_users, cfg.num_items
    video = ~items["is_product"]

    # ---- section 5.2: business metrics from the logs (consented, de-duplicated events)
    d = prepare(events, users, 30 * DAY)
    prod = items["is_product"][d["item"]]
    count = lambda name, mask: int(((d["event"] == name) & mask).sum())   # noqa: E731
    print("Section 5.2: conversion paths over 30 days (events of users who allowed personalisation)")
    v = {s: count(s, ~prod) for s in ("viewable", "click", "complete", "like")}
    p = {s: count(s, prod) for s in ("viewable", "click", "add_to_cart", "purchase")}
    print("  videos:   " + " -> ".join(f"{s} {n:,}" for s, n in v.items()))
    print("  products: " + " -> ".join(f"{s} {n:,}" for s, n in p.items()))
    print(f"  product conversion: {p['purchase'] / p['viewable']:.2%} of viewable impressions, "
          f"{p['purchase'] / p['click']:.1%} of clicks, {p['purchase'] / p['add_to_cart']:.0%} of carts")
    order_ids, first_idx = np.unique(orders["order_id"], return_index=True)
    first_item = orders["item_id"][first_idx]
    anchored = [o for o, a in zip(order_ids, first_item) if int(a) in truth["complements"]]
    basket = {}
    for o, i in zip(orders["order_id"].tolist(), orders["item_id"].tolist()):
        basket.setdefault(o, set()).add(i)
    cross = np.mean([len(set(truth["complements"][int(a)]) & basket[o]) > 0
                     for o, a in zip(order_ids, first_item) if int(a) in truth["complements"]])
    multi = np.mean([len({items["category"][i] for i in basket[o]}) > 1 for o in order_ids])
    print(f"  shop orders: {len(order_ids):,}; with items from 2+ categories: {multi:.1%}; "
          f"orders whose first item is an anchor product: {len(anchored):,}, "
          f"of which {cross:.1%} also contain one of its complements")

    # ---- the test week of Chapters 3 and 4: train to day 23, test days 23 to 30
    weight, used = user_item_scores(events, users, 23 * DAY, n_users, n_items)
    pos = test_positives(events, users, 23 * DAY, 30 * DAY, used, video)
    rows = np.array(sorted(pos))
    engaged = (weight > 0).sum(axis=0)
    ok = video & (engaged >= MIN_BUYERS)
    hist = (weight > 0).any(axis=1)
    pop = group_popularity(weight, np.zeros(n_users, dtype=np.int64), 1)[0]
    by_lang = group_popularity(weight, users["language"], len(LANGUAGES))
    sim = item_neighbours(weight, k=50, strength=0.0)                         # Chapter 3's choice
    X, Y = ials(weight, K, REG, ALPHA)
    Pb, Qb, bb = bpr(weight, 32, reg=0.01, neg_power=1.0)                      # Chapter 4's sampler
    lang_pop = lambda r: by_lang[users["language"][r]]                        # noqa: E731
    # most clicked: click-through rate per eligible video, with a prior of 20 views at the mean rate
    d23 = prepare(events, users, 23 * DAY)
    views = np.bincount(d23["item"][d23["event"] == "viewable"], minlength=n_items).astype(float)
    clicks = np.bincount(d23["item"][d23["event"] == "click"], minlength=n_items).astype(float)
    ctr = (clicks + 20 * clicks.sum() / views.sum()) / (views + 20)
    match = lambda r: users["vec"][r].astype(np.float64) @ items["vec"].T.astype(np.float64)  # noqa: E731
    row = lambda v: (lambda r: np.broadcast_to(v, (len(r), n_items)))        # noqa: E731
    shuffle = lambda r: np.array([np.random.default_rng([cfg.seed, 504, int(u)]).random(n_items)  # noqa: E731
                                  for u in r])                               # a fixed random order per user
    methods = {"random (eligible videos)": lambda r: np.where(ok, shuffle(r), NO),
               "most clicked (click-through rate)": row(np.where(ok, ctr, NO)),
               "popular worldwide": row(pop),
               "popular in your language": lang_pop,
               "CF, then popular in language": lambda r: (cf_scores(weight[r], sim), lang_pop(r)),
               "iALS, then popular in language":
                   lambda r: (eligible_and_scored(X[r] @ Y.T, weight[r], ok), lang_pop(r)),
               "BPR popular negatives, then language":
                   lambda r: (eligible_and_scored(Pb[r] @ Qb.T + bb, weight[r], ok), lang_pop(r))}
    sims = {"hidden taste (simulation only)": match,
            "taste + popularity (simulation only)": lambda r: 5 * match(r) + np.log(items["popularity"])}
    n_video = int(video.sum())

    # ---- sections 5.3 and 5.5: ranking metrics for every baseline
    full = {name: ranked(s, rows, used, video, k=n_video) for name, s in {**methods, **sims}.items()}
    table = {name: ranking_table(t, rows, pos, used, video) for name, t in full.items()}
    print(f"\nSections 5.3 and 5.5: test week, video row, {len(rows):,} users with a hit "
          f"(mean {np.mean([len(pos[u]) for u in rows]):.2f} relevant videos each)")
    print_rank_table("", table, ("hit@10", "P@10", "R@10", "MRR", "MAP@10", "NDCG@10", "AUC"))

    # per language, NDCG@10 (user-side fairness); report languages with 100+ test users
    lang_rows = users["language"][rows]
    print("\n  NDCG@10 by user language:")
    print(f"  {'':<34}" + "".join(f"{LANGUAGES[g][:10]:>11}" for g in range(len(LANGUAGES))))
    print(f"  {'test users':<34}" + "".join(f"{(lang_rows == g).sum():>11,}" for g in range(len(LANGUAGES))))
    for name in ("popular in your language", "CF, then popular in language", "iALS, then popular in language"):
        vals = []
        for g in range(len(LANGUAGES)):
            m = lang_rows == g
            vals.append(np.mean([ndcg_at_k(t.tolist(), {i: 1.0 for i in pos[u]})
                                 for u, t in zip(rows[m], full[name][m])]) if m.sum() >= 100 else np.nan)
        print(f"  {name:<34}" + "".join(f"{x:>11.3f}" for x in vals))

    # ---- section 5.3: calibration of iALS scores, Platt scaling fitted on the validation week
    w16, used16 = user_item_scores(events, users, 16 * DAY, n_users, n_items)
    pos16 = test_positives(events, users, 16 * DAY, 23 * DAY, used16, video)
    r16 = np.array(sorted(pos16))
    ok16 = video & ((w16 > 0).sum(axis=0) >= MIN_BUYERS)
    X16, Y16 = ials(w16, K, REG, ALPHA)

    def candidate_pairs(Xm, Ym, w, u_rows, usedm, okm, positives, n=50):
        s = eligible_and_scored(Xm[u_rows] @ Ym.T, w[u_rows], okm)
        s = np.where(usedm[u_rows], NO, s)
        top = np.argsort(-s, axis=1, kind="stable")[:, :n]
        sc = np.take_along_axis(s, top, axis=1)
        y = np.array([[i in positives[u] for i in t] for u, t in zip(u_rows, top)])
        keep = np.isfinite(sc)
        return sc[keep], y[keep].astype(float)

    s_val, y_val = candidate_pairs(X16, Y16, w16, r16, used16, ok16, pos16)
    s_te, y_te = candidate_pairs(X, Y, weight, rows, used, ok, pos)
    a, b = fit_platt(s_val, y_val)
    raw = np.clip(s_te, 0, 1)
    platt = 1 / (1 + np.exp(-(a * s_te + b)))
    print(f"\nCalibration: iALS top-50 candidates; Platt scaling fitted on {len(s_val):,} validation pairs "
          f"(a = {a:.2f}, b = {b:.2f}); tested on {len(s_te):,} test pairs, observed rate {y_te.mean():.4f}")
    print(f"  {'':<34}{'mean p':>9}{'ECE':>9}{'log loss':>10}{'Brier':>9}")
    for name, pr in (("raw score, clipped to 0-1", raw), ("Platt scaling", platt)):
        print(f"  {name:<34}{pr.mean():>9.4f}{ece(pr, y_te):>9.4f}{log_loss(pr, y_te):>10.4f}{brier(pr, y_te):>9.4f}")
    print("  ten equal-size groups of test pairs, sorted by prediction (mean prediction : observed rate):")
    for name, pr in (("raw", raw), ("platt", platt)):
        print(f"  {name:<6}" + "  ".join(f"{m:.3f}:{o:.3f}" for m, o, _ in reliability_by_rank(pr, y_te)))

    # ---- section 5.4: beyond accuracy, on the lists the test users would see
    hist_share = (engaged + 1) / (hist.sum() + 1)
    chart = ranked(lang_pop, rows, used, video)                               # the obvious list
    creators = items["creator"]
    cpop = np.bincount(creators, weights=engaged.astype(float), minlength=creators.max() + 1)
    top_creators = set(np.argsort(-cpop)[: len(cpop) // 10].tolist())
    print(f"\nSection 5.4: beyond accuracy, top-10 lists of the {len(rows):,} test users "
          f"({ok.sum():,} eligible videos)")
    print(f"  {'':<34}{'diversity':>10}{'novelty':>9}{'serendip.':>10}{'coverage':>10}{'Gini':>7}"
          f"{'top-10% creators':>18}")
    for name, t in full.items():
        top10 = t[:, :10]
        exp = np.zeros(n_items)
        for lst in top10:
            for i, e in exposure(lst.tolist()).items():
                exp[i] += e
        div = np.nanmean([diversity(lst.tolist(), items["category"]) for lst in top10])
        nov = np.nanmean([novelty(lst.tolist(), hist_share) for lst in top10])
        ser = np.mean([serendipity(lst.tolist(), pos[u], set(c.tolist())) for u, lst, c in zip(rows, top10, chart)])
        cov = np.mean(exp[ok] > 0)
        top_share = sum(exp[i] for i in np.flatnonzero(exp) if creators[i] in top_creators) / exp.sum()
        print(f"  {name:<34}{div:>10.2f}{nov:>9.2f}{ser:>10.4f}{cov:>10.1%}{gini(exp[ok]):>7.2f}{top_share:>18.1%}")
    print(f"  (the top 10% of creators own {np.isin(creators[ok], list(top_creators)).mean():.1%} "
          f"of eligible videos)")

    # ---- section 5.6: three splits, the same methods
    first = first_hit_time(events, users, 30 * DAY, video)
    w30, used30 = user_item_scores(events, users, 30 * DAY, n_users, n_items)
    pairs = sorted(first)
    print("\nSection 5.6: the same three methods under three splits (hit@10 and NDCG@10)")
    split_rng = np.random.default_rng(cfg.seed + 501)
    results = {}
    # (a) random split: 20% of all (user, video) hits, from any day, held out
    test_mask = split_rng.random(len(pairs)) < 0.2
    held = {}
    for (u, i), t in zip(pairs, test_mask):
        if t:
            held.setdefault(u, set()).add(i)
    # (b) leave last out: each user's last new hit (by first-hit time), for users with 2+ hits
    by_user = {}
    for (u, i), age in first.items():
        by_user.setdefault(u, []).append((age, i))
    last = {u: {min(v)[1]} for u, v in by_user.items() if len(v) >= 2}   # smallest age = latest
    for name, test in (("random split, 20% of hits", held), ("leave last hit out", last)):
        w = w30.copy()
        uu = used30.copy()
        for u, s in test.items():
            for i in s:
                w[u, i] = 0.0
                uu[u, i] = False
        r = np.array(sorted(test))
        sc = split_scores(w, users)
        results[name] = (len(r), {m: ranking_table(ranked(s, r, uu, video, k=n_video), r, test, uu, video)
                                   for m, s in sc.items()})
        del w, uu, sc
    results["time split (train to day 23)"] = (len(rows), {m: table[m] for m in
                                                           ("popular worldwide", "popular in your language",
                                                            "iALS, then popular in language")})
    targets = {"random split, 20% of hits": held, "leave last hit out": last}
    for name, (n, res) in results.items():
        tg = targets.get(name, pos)
        print(f"  {name} ({n:,} users, {np.mean([len(tg[u]) for u in tg]):.2f} target videos each)")
        for m, t in res.items():
            print(f"    {m:<34} hit@10 {t['hit@10']:.3f}  NDCG@10 {t['NDCG@10']:.3f}")

    # ---- section 5.6: sampled metrics - rank each user's hits among 100 random videos
    samp_rng = np.random.default_rng(cfg.seed + 502)
    print("\nSampled metrics: each test user's relevant videos ranked among 100 random unused videos")
    print(f"  {'':<34}{'full hit@10':>12}{'sampled hit@10':>15}{'full NDCG@10':>13}{'sampled NDCG@10':>16}")
    samples = {}
    for u in rows:
        pool = np.flatnonzero(video & ~used[u])
        pool = pool[~np.isin(pool, list(pos[u]))]
        samples[u] = set(samp_rng.choice(pool, 100, replace=False).tolist()) | pos[u]
    for name in methods:
        t_full = full[name]
        hs, ns = [], []
        for u, t in zip(rows, t_full):
            sub = [i for i in t.tolist() if i in samples[u]]
            hs.append(hit_at_k(sub, pos[u]))
            ns.append(ndcg_at_k(sub, {i: 1.0 for i in pos[u]}))
        print(f"  {name:<34}{table[name]['hit@10']:>12.3f}{np.mean(hs):>15.3f}"
              f"{table[name]['NDCG@10']:>13.3f}{np.mean(ns):>16.3f}")

    # ---- section 5.7: a simulated online test - every consenting user sees one top-10 list
    aud = np.flatnonzero(users["current_consent"])
    print(f"\nSection 5.7: simulated online test, {len(aud):,} users who allow personalisation, one list each")
    r = np.arange(1, 11)
    on = cfg.scroll_keep ** (r - 1)
    gl = np.where(r >= 2, cfg.scroll_keep ** np.maximum(r - 2, 0) * (1 - cfg.scroll_keep), 0.0)
    print("  P(look) by slot 1-10: " + " ".join(f"{x:.3f}" for x in (on + cfg.glimpse_look * gl) / r ** cfg.look_decay))
    print(f"  {'':<34}{'NDCG@10 (offline)':>18}{'clicks':>8}{'satisfied':>10}{'per click':>10}{'clickbait':>10}")
    online = {}
    for name, s in {**methods, **sims}.items():
        top = ranked(s, aud, used, video)
        click, sat = expected_outcomes(top, aud, users, items, cfg)
        online[name] = (top, sat.sum(axis=1))
        bait = np.mean(items["clickbait"][top[top >= 0]] > 0)
        print(f"  {name:<34}{table[name]['NDCG@10']:>18.3f}{click.sum(axis=1).mean():>8.3f}"
              f"{sat.sum(axis=1).mean():>10.3f}{sat.sum() / click.sum():>10.3f}{bait:>10.1%}")
    base, new = online["CF, then popular in language"][1], online["iALS, then popular in language"][1]
    print(f"  lift in expected satisfied plays, iALS chain over CF chain: "
          f"({new.mean():.4f} - {base.mean():.4f}) / {base.mean():.4f} = {(new.mean() - base.mean()) / base.mean():.1%}")

    # ---- section 5.8: a simulated A/B test, one week, users split at random into two halves
    ab_rng = np.random.default_rng(cfg.seed + 503)
    arm = ab_rng.random(len(aud)) < 0.5
    out = np.zeros(len(aud))
    for name, m in (("CF, then popular in language", ~arm), ("iALS, then popular in language", arm)):
        top = online[name][0][m]
        out[m] = simulate_week(top, aud[m], users, items, cfg, ab_rng)
    ya, yb = out[~arm], out[arm]

    diff = yb.mean() - ya.mean()
    se = np.sqrt(ya.var(ddof=1) / len(ya) + yb.var(ddof=1) / len(yb))
    z = diff / se
    p_two = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
    print(f"\nSection 5.8: simulated A/B test, one week, 7 views of the row per user")
    n_exp = len(aud) / 2
    chi2 = ((len(ya) - n_exp) ** 2 + (len(yb) - n_exp) ** 2) / n_exp
    print(f"  sample ratio check: {len(ya):,} and {len(yb):,} users, expected {n_exp:,.0f} each; "
          f"chi-square {chi2:.3f}, p = {1 - erf(sqrt(chi2 / 2)):.2f}")
    print(f"  control (CF chain): {len(ya):,} users, mean {ya.mean():.4f} satisfied plays, sd {ya.std(ddof=1):.4f}")
    print(f"  treatment (iALS chain): {len(yb):,} users, mean {yb.mean():.4f}, sd {yb.std(ddof=1):.4f}")
    print(f"  difference {diff:.4f} ({diff / ya.mean():.1%}), standard error {se:.4f}, "
          f"95% interval {diff - 1.96 * se:.4f} to {diff + 1.96 * se:.4f}, z = {z:.2f}, "
          f"p {'< 1e-10' if p_two < 1e-10 else f'= {p_two:.2g}'}")
    sd = ya.std(ddof=1)
    for rel in (0.02, 0.05, diff / ya.mean()):
        delta = rel * ya.mean()
        print(f"  users per arm to detect a {rel:.1%} change (alpha 0.05, power 80%): "
              f"16 x {sd:.4f}^2 / {delta:.4f}^2 = {16 * sd ** 2 / delta ** 2:,.0f}")
