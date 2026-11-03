"""Listings 5.1 and 5.2 - Metrics for recommendation lists (sections 5.3 and 5.4).

Every function here scores ONE user's list, or one set of predictions, so that
each step can be checked by hand. evaluate.py averages them over users.

A list is a sequence of item ids, best first. An id of -1 is an empty slot (the
method had nothing more to recommend). "relevant" is the set of items the user
went on to enjoy in the test period.

The ranking metrics share one input contract, checked by check_list(): item ids
in a list are unique, k is at least 1, and the user has at least one relevant
item. Recall, AP, NDCG and AUC are not defined for a user with nothing relevant,
so a caller leaves such users out of the average and reports how many there were.

Needs Python 3.12 or later. Run from the repository root to see the worked example:
    python chapter_05/metrics.py
"""
import math

import numpy as np


# ---------------------------------------------------------------- section 5.3: ranking
def check_list(ranked, relevant, k=None):
    """The input contract of every ranking metric. Raises ValueError if it is broken."""
    if k is not None and k < 1:
        raise ValueError("k must be at least 1")
    ids = [i for i in (ranked[:k] if k else ranked) if i >= 0]   # -1 (empty slot) may repeat
    if len(ids) != len(set(ids)):
        raise ValueError("the list repeats an item id")
    if not relevant:
        raise ValueError("no relevant items: leave this user out of the average and count them")


def hit_at_k(ranked, relevant, k=10):
    """1 if any of the top k is relevant, else 0."""
    check_list(ranked, relevant, k)
    return float(any(i in relevant for i in ranked[:k]))


def precision_at_k(ranked, relevant, k=10):
    """Relevant items in the top k, divided by k (empty slots count as misses)."""
    check_list(ranked, relevant, k)
    return sum(i in relevant for i in ranked[:k]) / k


def recall_at_k(ranked, relevant, k=10):
    """Relevant items in the top k, divided by all relevant items."""
    check_list(ranked, relevant, k)
    return sum(i in relevant for i in ranked[:k]) / len(relevant)


def reciprocal_rank(ranked, relevant, k=None):
    """1 / position of the first relevant item (positions start at 1); 0 if none."""
    check_list(ranked, relevant, k)
    for pos, i in enumerate(ranked[:k] if k else ranked, start=1):
        if i in relevant:
            return 1.0 / pos
    return 0.0


def average_precision(ranked, relevant, k=10):
    """Mean of precision@r over the positions r of relevant items in the top k,
    divided by min(|relevant|, k), the most hits a list of length k can hold."""
    check_list(ranked, relevant, k)
    hits, total = 0, 0.0
    for pos, i in enumerate(ranked[:k], start=1):
        if i in relevant:
            hits += 1
            total += hits / pos
    return total / min(len(relevant), k)


def ndcg_at_k(ranked, gains, k=10):
    """Normalised discounted cumulative gain. gains maps item -> gain (relevance);
    items not in gains have gain 0. Position r is discounted by 1 / log2(r + 1).
    The ideal list puts the largest gains first."""
    check_list(ranked, {i for i, g in gains.items() if g > 0}, k)
    dcg = sum(gains.get(i, 0.0) / math.log2(pos + 1) for pos, i in enumerate(ranked[:k], start=1))
    best = sorted(gains.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(pos + 1) for pos, g in enumerate(best, start=1))
    return dcg / idcg


def auc(ranked, relevant, candidates):
    """Share of (relevant, non-relevant) candidate pairs in which the relevant item
    is ranked higher. Candidates missing from the list are tied at the bottom;
    a tie counts as half. 0.5 is a random order, 1.0 a perfect one."""
    check_list(ranked, relevant)
    pos_of = {i: p for p, i in enumerate(ranked) if i >= 0}
    bottom = len(ranked) + 1
    rank = lambda i: pos_of.get(i, bottom)                  # noqa: E731  (lower is better)
    rel = [rank(i) for i in candidates if i in relevant]
    non = np.sort([rank(i) for i in candidates if i not in relevant])
    if not rel or len(non) == 0:
        raise ValueError("AUC needs at least one relevant and one non-relevant candidate")
    wins = sum(len(non) - np.searchsorted(non, r, side="right") for r in rel)
    ties = sum(np.searchsorted(non, r, side="right") - np.searchsorted(non, r, side="left") for r in rel)
    return float((wins + 0.5 * ties) / (len(rel) * len(non)))


# ---------------------------------------------------------------- section 5.3: calibration
def reliability(p, y, bins=10):
    """Group predictions into equal-width bins of predicted probability. Returns,
    per non-empty bin: mean prediction, observed rate and count."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    out = []
    for b in range(bins):
        m = idx == b
        if m.any():
            out.append((float(p[m].mean()), float(y[m].mean()), int(m.sum())))
    return out


def reliability_by_rank(p, y, groups=10):
    """Reliability with equal-count groups: sort by prediction, cut into groups of
    (nearly) equal size. Useful when most predictions are small."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    order = np.argsort(p, kind="stable")
    return [(float(p[g].mean()), float(y[g].mean()), len(g)) for g in np.array_split(order, groups)]


def ece(p, y, bins=10):
    """Expected calibration error: the count-weighted mean gap between the mean
    prediction and the observed rate, over the bins of reliability()."""
    n = len(p)
    return sum(c / n * abs(obs - pred) for pred, obs, c in reliability(p, y, bins))


def log_loss(p, y, eps=1e-12):
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(p, y):
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def fit_platt(s, y, iters=50):
    """Platt scaling: p = sigmoid(a * s + b), fitted by Newton's method on log loss.
    The slope a is not constrained: the map keeps the order of scores only if a > 0."""
    s, y = np.asarray(s, float), np.asarray(y, float)
    a, b = 1.0, 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(a * s + b)))
        g = np.array([np.sum((p - y) * s), np.sum(p - y)])
        w = p * (1 - p)
        h = np.array([[np.sum(w * s * s), np.sum(w * s)], [np.sum(w * s), np.sum(w)]]) + 1e-9 * np.eye(2)
        a, b = np.array([a, b]) - np.linalg.solve(h, g)
    return float(a), float(b)


# ---------------------------------------------------------------- section 5.4: beyond accuracy
def diversity(ranked, category):
    """Intra-list diversity by category: share of pairs of listed items whose
    categories differ. 0 = all one category."""
    c = [category[i] for i in ranked if i >= 0]
    pairs = len(c) * (len(c) - 1) / 2
    if pairs == 0:
        return float("nan")
    same = sum(c[a] == c[b] for a in range(len(c)) for b in range(a + 1, len(c)))
    return 1 - same / pairs


def novelty(ranked, share):
    """Mean self-information, -log2(share of users who engaged with the item), in
    bits. A hit everyone has seen carries little news; a rare item carries more."""
    s = [share[i] for i in ranked if i >= 0]
    return float(np.mean([-math.log2(x) for x in s])) if s else float("nan")


def serendipity(ranked, relevant, expected, k=10):
    """Share of the top k that is both relevant and unexpected, where "expected"
    is what an obvious recommender (here, the chart) would have shown."""
    return sum(i in relevant and i not in expected for i in ranked[:k]) / k


def gini(x):
    """Gini coefficient of non-negative amounts: 0 = equal, near 1 = concentrated."""
    x = np.sort(np.asarray(x, float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return float("nan")
    cum = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def exposure(ranked):
    """Position-weighted exposure: slot r gets 1 / log2(r + 1), as in NDCG."""
    return {i: 1 / math.log2(pos + 1) for pos, i in enumerate(ranked, start=1) if i >= 0}


def worked_example():
    """Section 5.3: one list, every metric, by hand."""
    ranked = [12, 7, 33, 5, 21, 40, 8, 19, 2, 27]
    relevant = {7, 21, 50}                                   # 50 is not in the list
    gains = {i: 1.0 for i in relevant}
    print("Worked example: list", ranked, "relevant", sorted(relevant))
    print(f"  hit@10 {hit_at_k(ranked, relevant):.0f}; precision@10 {precision_at_k(ranked, relevant):.2f}; "
          f"recall@10 {recall_at_k(ranked, relevant):.4f}; reciprocal rank {reciprocal_rank(ranked, relevant):.2f}")
    print(f"  AP@10 {average_precision(ranked, relevant):.4f}; NDCG@10 {ndcg_at_k(ranked, gains):.4f}")
    graded = {7: 1.0, 21: 3.0, 50: 3.0}                      # gain 2^g - 1: complete g = 1, like g = 2
    print(f"  NDCG@10 with graded gains {graded}: {ndcg_at_k(ranked, graded):.4f}")
    cands = ranked + [50, 60, 61, 62, 63]                    # 15 candidates, 3 relevant
    print(f"  AUC over 15 candidates (50 and 60-63 unranked): {auc(ranked, relevant, cands):.4f}")
    for bad, why in ((lambda: average_precision([1, 1], {1}, k=2), "repeated id"),
                     (lambda: recall_at_k(ranked, set()), "no relevant items")):
        try:
            bad()
        except ValueError as e:
            print(f"  input contract ({why}): ValueError: {e}")


if __name__ == "__main__":
    worked_example()
