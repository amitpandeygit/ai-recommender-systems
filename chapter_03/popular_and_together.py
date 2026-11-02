"""Listings 3.1 and 3.2 - Non-personalised rows and "frequently bought together".

Section 3.2: top charts per language, a damped mean for few ratings, and
trending with smoothing.
Section 3.3: support, confidence and lift on baskets, a significance score,
and the heavy-buyer correction Amazon described in 2017.

Everything that learns from behaviour uses only data the user allowed for
personalisation, both at the time and now (Chapter 2's consent rule).

Needs Python 3.12 or later. Run from the repository root:
    python chapter_03/popular_and_together.py
"""
import numpy as np

from tesselo_shop import CATEGORIES, DAY, LANGUAGES, Config, build
from implicit_scores import pair_sums, prepare

MIN_BUYERS = 5            # never publish a chart entry or a pair seen with fewer buyers


# ---------------------------------------------------------------- section 3.2
def damped_mean(total, count, prior_mean, prior_weight):
    """Average that starts at prior_mean and moves to the item's own average as
    evidence grows: (total + prior_weight * prior_mean) / (count + prior_weight)."""
    return (total + prior_weight * prior_mean) / (count + prior_weight)


def engaged_users(data, n_items):
    """Per item: users whose decayed implicit score (Chapter 2) is positive."""
    pairs, raw, _ = pair_sums(data["user"], data["item"], data["event"], data["age_days"])
    good = pairs[raw > 0]
    return good, np.bincount(good[:, 1], minlength=n_items)


def top_chart(good, group_of_user, n_groups, n_items, k=10):
    """Top k items per group (for example, per app language) by engaged users."""
    counts = np.zeros((n_groups, n_items), dtype=np.int64)
    np.add.at(counts, (group_of_user[good[:, 0]], good[:, 1]), 1)
    counts[counts < MIN_BUYERS] = 0                       # small counts are never shown
    return np.argsort(-counts, axis=1, kind="stable")[:, :k], counts


def trending(item, time, now, n_items, recent_days=3, base_days=14, prior=5.0):
    """Smoothed ratio of recent sales to the sales the item's own baseline predicts.

    expected = baseline sales * recent_days / base_days. Adding `prior` to both sides
    stops an item going from 1 sale to 3 from looking like a 3x trend.
    """
    recent = (time >= now - recent_days * DAY) & (time < now)
    base = (time >= now - (recent_days + base_days) * DAY) & ~recent & (time < now)
    r = np.bincount(item[recent], minlength=n_items).astype(float)
    b = np.bincount(item[base], minlength=n_items).astype(float)
    expected = b * recent_days / base_days
    return (r + prior) / (expected + prior), r, expected


# ---------------------------------------------------------------- section 3.3
def basket_matrix(order_id, item, n_items):
    """Baskets x items, 1 if the basket contains the item."""
    ids, row = np.unique(order_id, return_inverse=True)
    x = np.zeros((len(ids), n_items), dtype=np.float32)
    x[row, item] = 1.0
    return x


def pair_statistics(x):
    """Support, confidence and lift for every ordered pair (A -> B), plus a
    significance score. x is baskets x items (0/1)."""
    n = x.shape[0]
    n_a = x.sum(axis=0)                                   # baskets with A
    n_ab = x.T @ x                                        # baskets with both A and B
    np.fill_diagonal(n_ab, 0)
    support = n_ab / n
    with np.errstate(divide="ignore", invalid="ignore"):
        confidence = n_ab / n_a[:, None]                  # P(B | A); row A, column B
        expected = np.outer(n_a, n_a) / n                 # baskets with both if independent
        lift = n_ab / expected
        score = (n_ab - expected) / np.sqrt(expected)     # how many standard errors above chance
    for m in (confidence, lift, score):
        m[~np.isfinite(m)] = 0.0
    return {"n_a": n_a, "n_ab": n_ab, "support": support, "confidence": confidence,
            "lift": lift, "score": score, "expected": expected}


def heavy_buyer_expected(user, item, n_items):
    """Expected number of X-buyers who also buy Y if X and Y are unrelated.

    Naive: |X buyers| * P(a customer buys Y).
    Corrected (Smith and Linden, 2017): each X-buyer c gets |c| chances to buy Y,
    where |c| is c's number of non-X purchases, each succeeding with
    P_Y = Y purchases / all purchases: sum over c of 1 - (1 - P_Y) ** |c|.
    Returns observed counts, naive and corrected expectations (items x items).
    """
    custs, cu = np.unique(user, return_inverse=True)
    buys = np.zeros((len(custs), n_items), dtype=np.float32)   # purchases per customer and item
    np.add.at(buys, (cu, item), 1.0)
    bought = (buys > 0).astype(np.float32)
    n_obs = bought.T @ bought                                   # customers who bought both
    p_y_cust = bought.mean(axis=0)
    naive = bought.sum(axis=0)[:, None] * p_y_cust[None, :]
    p_y = buys.sum(axis=0) / buys.sum()
    total = buys.sum(axis=1)
    corrected = np.zeros((n_items, n_items))
    for x in np.flatnonzero(bought.sum(axis=0) > 0):
        c = np.flatnonzero(bought[:, x])
        m, h = np.unique(total[c] - buys[c, x], return_counts=True)   # |c| and how many customers
        corrected[x] = (h[:, None] * (1 - (1 - p_y[None, :]) ** m[:, None])).sum(axis=0)
    return n_obs, naive, corrected


def worked_example():
    print("Damped mean (prior 4.1 stars, weight 20):")
    for name, total, count in (("Earbuds A", 10.0, 2), ("Earbuds B", 2_208.0, 480)):
        print(f"  {name}: plain mean {total / count:.2f}, damped "
              f"{damped_mean(total, count, 4.1, 20):.3f}")
    names = ["phone", "case", "charger", "top-up", "lipstick"]
    baskets = [[0, 1, 3], [0, 1], [0, 2], [0, 1, 2, 3], [1], [1, 3], [3], [4, 3], [4], [0, 1, 3]]
    x = np.zeros((len(baskets), len(names)), dtype=np.float32)
    for r, b in enumerate(baskets):
        x[r, b] = 1
    s = pair_statistics(x)
    print("\nRule                support  confidence   lift")
    for a, b in ((0, 1), (1, 0), (0, 3), (0, 2)):
        print(f"{names[a]:>8} -> {names[b]:<8}  {s['support'][a, b]:.2f}      "
              f"{s['confidence'][a, b]:.2f}      {s['lift'][a, b]:.3f}")


if __name__ == "__main__":
    worked_example()
    cfg = Config()
    users, items, served, events, orders, truth = build(cfg)
    n_items, now = cfg.num_items, cfg.days * DAY

    # Charts per app language from Chapter 2's implicit scores, at the end of day 30
    data = prepare(events, users, now)
    good, engaged = engaged_users(data, n_items)
    top, counts = top_chart(good, users["language"], len(LANGUAGES), n_items)
    glob = np.argsort(-engaged, kind="stable")[:10]
    print("\nTop 5 by engaged users, worldwide and in two app languages:")
    print("  worldwide:", [f"{i}({CATEGORIES[items['category'][i]]})" for i in glob[:5]])
    for g in (1, 4):
        print(f"  {LANGUAGES[g]}:", [f"{i}({CATEGORIES[items['category'][i]]})" for i in top[g, :5]])
    overlap = np.mean([len(set(top[g]) & set(glob)) for g in range(len(LANGUAGES))])
    same_lang = np.mean([(items["language"][top[g]] == g).mean() for g in range(len(LANGUAGES))])
    print(f"  Average overlap of a language's top 10 with the worldwide top 10: {overlap:.1f} items;"
          f" share of a language's top 10 in that language: {same_lang:.0%}")

    # Consent: only orders placed while personalisation was on, by users who still allow it
    ok = (orders["time"] < users["consent_off_from"][orders["user_id"]]) \
        & users["current_consent"][orders["user_id"]]
    o = {key: val[ok] for key, val in orders.items()}
    print(f"\nOrder lines usable for learning: {ok.sum():,} of {len(ok):,}")

    # Trending at the end of day 30
    tr, r, e = trending(o["item_id"], o["time"], now, n_items)
    raw_ratio = np.where(e > 0, r / np.maximum(e, 1e-9), np.where(r > 0, np.inf, 0))
    for label, s in (("raw ratio (recent / expected)", raw_ratio), ("smoothed, prior 5", tr)):
        best = np.argsort(-s, kind="stable")[:5]
        print(f"Trending by {label}: planted trends in top 5: {np.isin(best, truth['trend']).sum()}; "
              f"recent sales of the top 5: {r[best].astype(int).tolist()}")

    # Frequently bought together, at the basket level
    products = np.flatnonzero(items["is_product"])          # work on products only: smaller matrices
    col = np.full(n_items, -1)
    col[products] = np.arange(len(products))
    x = basket_matrix(o["order_id"], col[o["item_id"]], len(products))
    s = pair_statistics(x)
    comp = {col[a]: col[b].tolist() for a, b in truth["complements"].items()}
    t = col[truth["topup"]]
    anchors = [a for a in comp if s["n_a"][a] >= 20]
    print(f"\nBaskets: {x.shape[0]:,}. Anchors with at least 20 baskets: {len(anchors)}")
    print("Share of each anchor's top 2 that are its planted complements, and how often")
    print("the data top-up is in the top 2:")
    for label, m in (("confidence", s["confidence"]), ("lift", s["lift"]),
                     ("significance score", s["score"])):
        hit = top_up = 0
        for a in anchors:
            best = np.argsort(-m[a], kind="stable")[:2]
            hit += np.isin(best, comp[a]).sum()
            top_up += t in best
        print(f"  {label:<20} complements {hit / (2 * len(anchors)):.0%}   top-up {top_up / len(anchors):.0%}")
    shown = (s["score"] >= 3) & (s["n_ab"] >= MIN_BUYERS)     # Tesselo's rule for showing a pair
    n_shown = [min(shown[a].sum(), 2) for a in anchors]
    good = sum(np.isin(np.argsort(-np.where(shown[a], s["score"][a], -np.inf))[:k], comp[a]).sum()
               for a, k in zip(anchors, n_shown))
    print(f"  With the rule (score >= 3, >= {MIN_BUYERS} baskets): {np.mean(n_shown):.2f} pairs shown per "
          f"anchor, {good / max(sum(n_shown), 1):.0%} of them planted complements")

    # Customers who bought X also bought Y: the heavy-buyer correction
    n_obs, naive, corrected = heavy_buyer_expected(o["user_id"], col[o["item_id"]], len(products))
    xs = np.flatnonzero(n_obs.diagonal() >= 30)
    xs = xs[xs != t]
    print(f"\nItems with at least 30 buyers: {len(xs)}. Customers who bought X and the top-up,"
          f" observed / expected:")
    print(f"  naive expectation:     {np.mean(n_obs[xs, t] / naive[xs, t]):.2f}")
    print(f"  heavy-buyer corrected: {np.mean(n_obs[xs, t] / corrected[xs, t]):.2f}")
