"""Listings 3.3 to 3.7 - Similarity, content-based filtering, item-based
collaborative filtering, k-means segments, and an honest offline test.

Section 3.4: four similarity measures on one small example, and shrinkage.
Section 3.5: content-based filtering with TF-IDF item vectors and user profiles
             built from Chapter 2's implicit scores.
Section 3.6: item-based collaborative filtering with a precomputed neighbour table.
Section 3.7: k-means user segments, used for users with little history.
Section 3.8: the first Tesselo recommender, tested on a time split. Training uses
             only what was known at the end of day 16 (validation) or day 23
             (test), through Chapter 2's prepare(). Settings are chosen on the
             validation week, never on the test week.

This is a laptop-scale, in-memory implementation for teaching (dense NumPy
matrices). Chapter 6 shows how neighbour search runs at scale.

Needs Python 3.12 or later. Run from the repository root:
    python chapter_03/similarity_recs.py
"""
import numpy as np

from tesselo_shop import DAY, LANGUAGES, Config, build, vocabulary
from implicit_scores import active_blocks, pair_sums, prepare

STRONG = {"complete", "like", "add_to_cart", "purchase"}   # what counts as a hit in the test week


# ---------------------------------------------------------------- section 3.4
def jaccard(a, b):
    """Users who engaged with both / users who engaged with either (0 means no engagement)."""
    a, b = a != 0, b != 0
    union = (a | b).sum()
    return (a & b).sum() / union if union else 0.0


def cosine(a, b):
    """Angle between the two full columns; a missing score counts as 0."""
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return a @ b / norm if norm else 0.0


def pearson(a, b):
    """Correlation over users who scored both items, each centred on the item's mean."""
    both = (a != 0) & (b != 0)
    if both.sum() < 2:
        return 0.0                                           # no evidence either way
    x, y = a[both] - a[both].mean(), b[both] - b[both].mean()
    norm = np.sqrt((x @ x) * (y @ y))
    return x @ y / norm if norm else 0.0


def adjusted_cosine(a, b, user_mean):
    """Cosine over users who scored both, each score centred on that user's own mean."""
    both = (a != 0) & (b != 0)
    x, y = a[both] - user_mean[both], b[both] - user_mean[both]
    norm = np.sqrt((x @ x) * (y @ y))
    return x @ y / norm if norm else 0.0


def shrink(similarity, n_common, strength):
    """Pull a similarity towards 0 when few users support it: n / (n + strength)."""
    return similarity * n_common / (n_common + strength)


def worked_example():
    x = np.array([6.0, 2.0, 8.0, 0.0, 4.0])          # five users' scores on item X (0 = none)
    y = np.array([5.0, 1.0, 0.0, 3.0, 4.0])          # and on item Y
    user_mean = np.array([4.0, 3.0, 5.0, 2.0, 2.0])  # each user's mean over all their items
    p = pearson(x, y)
    print(f"Jaccard {jaccard(x, y):.3f}  cosine {cosine(x, y):.4f}  Pearson {p:.4f}  "
          f"adjusted cosine {adjusted_cosine(x, y, user_mean):.4f}")
    print(f"Pearson shrunk with 3 common users and strength 10: {shrink(p, 3, 10):.4f}")


# ---------------------------------------------------------------- data for 3.5 to 3.8
def user_item_scores(events, users, cutoff, n_users, n_items):
    """Chapter 2's implicit scores at a cutoff, as dense users x items matrices."""
    d = prepare(events, users, cutoff)
    pairs, raw, blocked = pair_sums(d["user"], d["item"], d["event"], d["age_days"])
    blocks = active_blocks(events, users, cutoff)
    blocked |= np.array([(int(u), int(i)) in blocks for u, i in pairs], dtype=bool)
    used = np.zeros((n_users, n_items), dtype=bool)          # any scored event: do not recommend again
    used[pairs[:, 0], pairs[:, 1]] = True
    for u, i in blocks:
        used[u, i] = True                                    # blocked items are never shown
    weight = np.zeros((n_users, n_items), dtype=np.float32)  # how much each positive pair counts
    ok = (raw > 0) & ~blocked
    weight[pairs[ok, 0], pairs[ok, 1]] = np.log1p(raw[ok])
    return weight, used


def test_positives(events, users, start, end, used):
    """(user, item) pairs with a strong positive in [start, end), new to the user."""
    d = prepare(events, users, end, window_days=(end - start) / DAY)
    strong = np.isin(d["event"], list(STRONG))
    pairs = {(int(u), int(i)) for u, i in zip(d["user"][strong], d["item"][strong])
             if not used[u, i]}
    by_user = {}
    for u, i in pairs:
        by_user.setdefault(u, set()).add(i)
    return by_user


# ---------------------------------------------------------------- section 3.5
def item_features(items, n_items):
    """TF-IDF over title words, plus one token each for category and language."""
    words = vocabulary()
    col = {w: k for k, w in enumerate(words)}
    n_cat = items["category"].max() + 1
    f = np.zeros((n_items, len(words) + n_cat + len(LANGUAGES)), dtype=np.float32)
    for i, title in enumerate(items["title"]):
        for w in title:
            f[i, col[w]] += 1.0
    f[np.arange(n_items), len(words) + items["category"]] = 1.0
    f[np.arange(n_items), len(words) + n_cat + items["language"]] = 1.0
    df = (f > 0).sum(axis=0)
    f *= np.log((1 + n_items) / (1 + df)) + 1.0             # rare tokens count more
    return f / np.linalg.norm(f, axis=1, keepdims=True)


def content_scores(weight, feats):
    """Profile = score-weighted sum of the user's item vectors; score = cosine."""
    profile = weight @ feats
    norm = np.linalg.norm(profile, axis=1, keepdims=True)
    profile = np.divide(profile, norm, out=np.zeros_like(profile), where=norm > 0)
    return profile @ feats.T, profile


# ---------------------------------------------------------------- section 3.6
def item_neighbours(weight, k=50, strength=20.0, min_common=3):
    """Offline job: each item's k most similar items (cosine on engaged users,
    shrunk by n / (n + strength), ignoring pairs with fewer than min_common users)."""
    b = (weight > 0).astype(np.float32)
    common = b.T @ b                                         # users who engaged with both
    n = np.diag(common).copy()
    denom = np.sqrt(np.outer(n, n)) * (common + strength) / np.maximum(common, 1e-12)
    sim = np.divide(common, denom, out=np.zeros_like(common), where=common > 0)
    sim[common < min_common] = 0.0
    np.fill_diagonal(sim, 0.0)
    drop = np.argpartition(-sim, k, axis=1)[:, k:]           # keep each row's top k
    np.put_along_axis(sim, drop, 0.0, axis=1)
    return sim


def cf_scores(weight, sim):
    """Online: score(u, j) = sum over the user's items i of weight(u, i) * sim(i, j)."""
    return weight @ sim


# ---------------------------------------------------------------- section 3.7
def kmeans(x, k, rng, iters=30):
    """Lloyd's algorithm with k-means++ seeding. Returns centres, labels and inertia."""
    centres = [x[rng.integers(len(x))]]
    for _ in range(k - 1):                                   # k-means++: far points more likely
        d = np.min([((x - c) ** 2).sum(axis=1) for c in centres], axis=0)
        centres.append(x[rng.choice(len(x), p=d / d.sum())])
    centres = np.array(centres)
    for _ in range(iters):
        dist = ((x[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
        labels = dist.argmin(axis=1)
        for c in range(k):
            if (labels == c).any():
                centres[c] = x[labels == c].mean(axis=0)
    inertia = ((x - centres[labels]) ** 2).sum()
    return centres, labels, inertia


def group_popularity(weight, group, n_groups):
    """Engaged users per item within each group (language, or k-means segment)."""
    counts = np.zeros((n_groups, weight.shape[1]), dtype=np.float32)
    for g in range(n_groups):
        counts[g] = (weight[group == g] > 0).sum(axis=0)
    return counts


# ---------------------------------------------------------------- section 3.8
def top_k(scores, used, k=10):
    """Best k unused items per user, best first; -1 where fewer than k are left."""
    s = np.where(used, -np.inf, scores)
    best = np.argpartition(-s, k, axis=1)[:, :k]
    rows = np.arange(len(s))[:, None]
    best = best[rows, np.argsort(-s[rows, best], axis=1, kind="stable")]
    return np.where(np.isfinite(s[rows, best]), best, -1)


def hit_rate(top, positives, users_):
    """Share of users with at least one test positive in their list."""
    return np.mean([len(positives[u] & set(top[u].tolist())) > 0 for u in users_])


def segment_scores(weight, profile, fallback, n_segments, seed=0):
    """Popular in the user's k-means segment; users with no history get the fallback."""
    has = (weight > 0).any(axis=1)
    _, seg, _ = kmeans(profile[has], n_segments, np.random.default_rng(seed))
    counts = group_popularity(weight[has], seg, n_segments)
    out = fallback.copy()
    out[np.flatnonzero(has)] = counts[seg]
    return out


def recommenders(weight, feats, language, item_language, n_segments, k, strength):
    """Score matrices for every method (higher is better)."""
    has = (weight > 0).any(axis=1)[:, None]
    same = (language[:, None] == item_language[None, :]).astype(np.float32)
    pop = (weight > 0).sum(axis=0).astype(np.float32)
    lang_pop = group_popularity(weight, language, len(LANGUAGES))[language]
    content, profile = content_scores(weight, feats)
    cf = cf_scores(weight, item_neighbours(weight, k=k, strength=strength))
    tie = lang_pop / (lang_pop.max(axis=1, keepdims=True) + 1.0)   # in [0, 1): breaks ties only
    return {"popular worldwide": np.broadcast_to(pop, weight.shape),
            "popular in your language": lang_pop,
            "popular in your segment": segment_scores(weight, profile, lang_pop, n_segments),
            "content, in your language": np.where(has, same + 0.5 * content, lang_pop),
            "item-based CF": cf,
            "CF, then popular in language": np.where(cf > 0, cf + 1.0, tie)}


if __name__ == "__main__":
    worked_example()
    cfg = Config()
    users, items, served, events, orders, truth = build(cfg)
    n_users, n_items = cfg.num_users, cfg.num_items
    feats = item_features(items, n_items)

    # Choose settings on the validation week: train to day 16, test on days 16 to 23
    w16, used16 = user_item_scores(events, users, 16 * DAY, n_users, n_items)
    pos16 = test_positives(events, users, 16 * DAY, 23 * DAY, used16)
    print(f"\nValidation week (train to day 16, test days 16-23), {len(pos16):,} users, hit rate@10:")
    best = None
    for k in (20, 50, 200):
        for strength in (0.0, 5.0, 20.0):
            cf = cf_scores(w16, item_neighbours(w16, k=k, strength=strength))
            hr = hit_rate(top_k(cf, used16), pos16, list(pos16))
            print(f"  item-based CF, {k:>3} neighbours, shrinkage strength {strength:>4.0f}: {hr:.3f}")
            best = max(best or (hr, k, strength), (hr, k, strength))
    lang16 = group_popularity(w16, users["language"], len(LANGUAGES))[users["language"]]
    _, profile16 = content_scores(w16, feats)
    seg_best = None
    for n_seg in (4, 8, 16, 32):
        hr = hit_rate(top_k(segment_scores(w16, profile16, lang16, n_seg), used16), pos16, list(pos16))
        print(f"  popular in your segment, {n_seg:>2} segments: {hr:.3f}")
        seg_best = max(seg_best or (hr, n_seg), (hr, n_seg))
    print(f"Chosen: {best[1]} neighbours, strength {best[2]:.0f}, {seg_best[1]} segments")

    # Final test: train to day 23, test on days 23 to 30, settings fixed above
    weight, used = user_item_scores(events, users, 23 * DAY, n_users, n_items)
    pos = test_positives(events, users, 23 * DAY, 30 * DAY, used)
    scores = recommenders(weight, feats, users["language"], items["language"],
                          seg_best[1], best[1], best[2])
    scores["hidden taste (simulation only)"] = users["vec"] @ items["vec"].T
    history = (weight > 0).sum(axis=1)
    groups = {"all": list(pos), "0-4 items": [u for u in pos if history[u] <= 4],
              "5+ items": [u for u in pos if history[u] >= 5]}
    print(f"\nTest week (train to day 23, test days 23-30), hit rate@10")
    print(f"{'':<32}" + "".join(f"{g:>11}" for g in groups))
    print(f"{'users':<32}" + "".join(f"{len(v):>11,}" for v in groups.values()))
    for name, s in scores.items():
        top = top_k(s, used)
        print(f"{name:<32}" + "".join(f"{hit_rate(top, pos, v):>11.3f}" for v in groups.values()))

    # "New for you": items uploaded on day 23 have no history; rank only them
    new = np.random.default_rng(cfg.seed + 303).random(n_items) < 0.10
    w_old = np.where(new[None, :], 0.0, weight).astype(np.float32)   # as if new items did not exist
    pos_new = {u: {i for i in s if new[i]} for u, s in pos.items()}
    pos_new = {u: s for u, s in pos_new.items() if s}
    rows = np.array(sorted(pos_new))                                  # only users we can test
    content, _ = content_scores(w_old[rows], feats)
    same_lang = (users["language"][rows, None] == items["language"][None, :]).astype(np.float32)
    hide = used[rows] | ~new[None, :]                                 # only new items compete
    rng = np.random.default_rng(1)
    print(f"\n'New for you' row: {new.sum()} new items, {len(pos_new):,} users with a positive on one")
    for name, base in (("random order", 0.0), ("in your language, random order", same_lang),
                       ("in your language, by content", same_lang + 0.5 * content)):
        runs = []
        for _ in range(20):                                           # random tie-breaks: average 20 draws
            top = top_k(base + 1e-3 * rng.random(hide.shape), hide)
            runs.append(np.mean([len(pos_new[u] & set(t.tolist())) > 0 for u, t in zip(rows, top)]))
        print(f"  {name:<32} hit rate@10 {np.mean(runs):.3f}")
