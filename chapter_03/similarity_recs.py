"""Listings 3.3 to 3.7 - Similarity, content-based filtering, item-based
collaborative filtering, k-means segments, and an honest offline test.

Section 3.4: four similarity measures on one small example, and shrinkage.
Section 3.5: content-based filtering with TF-IDF item vectors and user profiles
             built from Chapter 2's implicit scores.
Section 3.6: item-based collaborative filtering with a precomputed neighbour table.
Section 3.7: k-means user segments, used for users with little history.
Section 3.8: the first Tesselo recommender for the "Because you watched" video
             row, tested on a time split. Training uses only what was known at
             the end of day 16 (validation) or day 23 (test), through Chapter 2's
             prepare(). Settings are chosen on the validation week, never on the
             test week. Every method has its own eligibility rule: an item a
             method has no evidence for, or that fewer than MIN_BUYERS users
             stand behind, is never recommended, so a list can be shorter than 10.

This is a laptop-scale, in-memory implementation for teaching (dense NumPy
matrices). Chapter 6 shows how neighbour search runs at scale.

Needs Python 3.12 or later. Run from the repository root:
    python chapter_03/similarity_recs.py
"""
import numpy as np

from tesselo_shop import DAY, LANGUAGES, Config, build, vocabulary
from implicit_scores import EVENT_WEIGHTS, active_blocks, pair_sums, prepare
from popular_and_together import MIN_BUYERS

HITS = {"complete", "like"}     # a hit for "Because you watched": a completed or liked video
NO = -np.inf                    # score of an item a method may not recommend
DECIMALS = 9                    # scores are rounded, then ties go to the lower item id


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
    """Chapter 2's implicit scores at a cutoff, as dense users x items matrices.

    weight: log(1 + decayed score) for positive, unblocked pairs (no rarity weight).
    used:   pairs with any scored event, or an active block. These are never
            recommended again. Unscored telemetry (viewable, watch_end) does not
            count: an item the user only scrolled past can still be recommended.
    """
    d = prepare(events, users, cutoff)
    scored = np.isin(d["event"], list(EVENT_WEIGHTS))
    pairs, raw, blocked = pair_sums(d["user"][scored], d["item"][scored], d["event"][scored],
                                    d["age_days"][scored])
    blocks = active_blocks(events, users, cutoff)
    blocked |= np.array([(int(u), int(i)) in blocks for u, i in pairs], dtype=bool)
    used = np.zeros((n_users, n_items), dtype=bool)
    used[pairs[:, 0], pairs[:, 1]] = True
    for u, i in blocks:
        used[u, i] = True
    weight = np.zeros((n_users, n_items))
    ok = (raw > 0) & ~blocked
    weight[pairs[ok, 0], pairs[ok, 1]] = np.log1p(raw[ok])
    return weight, used


def test_positives(events, users, start, end, used, eligible_items):
    """Users -> items with a completed or liked video in [start, end), not used before."""
    d = prepare(events, users, end, window_days=(end - start) / DAY)
    hit = np.isin(d["event"], list(HITS)) & eligible_items[d["item"]]
    by_user = {}
    for u, i in zip(d["user"][hit].tolist(), d["item"][hit].tolist()):
        if not used[u, i]:
            by_user.setdefault(u, set()).add(i)
    return by_user


# ---------------------------------------------------------------- section 3.5
def item_features(items, n_items):
    """TF-IDF over title words, plus one token each for category and language."""
    words = vocabulary()
    col = {w: k for k, w in enumerate(words)}
    n_cat = items["category"].max() + 1
    f = np.zeros((n_items, len(words) + n_cat + len(LANGUAGES)))
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
def item_neighbours(weight, k=50, strength=20.0, min_common=MIN_BUYERS):
    """Offline job: each item's k most similar items (cosine on engaged users,
    shrunk by n / (n + strength), ignoring pairs with fewer than min_common users)."""
    b = (weight > 0).astype(np.float64)
    common = b.T @ b                                         # users who engaged with both
    n = np.diag(common).copy()
    denom = np.sqrt(np.outer(n, n)) * (common + strength) / np.maximum(common, 1e-12)
    sim = np.divide(common, denom, out=np.zeros_like(common), where=common > 0)
    sim[common < min_common] = 0.0
    np.fill_diagonal(sim, 0.0)
    order = np.argsort(-np.round(sim, DECIMALS), axis=1, kind="stable")
    np.put_along_axis(sim, order[:, k:], 0.0, axis=1)        # keep each row's top k
    return sim


def cf_scores(weight, sim):
    """Online: score(u, j) = sum over the user's items i of weight(u, i) * sim(i, j).
    Items with no supporting neighbour get no score at all, not a zero."""
    s = weight @ sim
    return np.where(s > 0, s, NO)


# ---------------------------------------------------------------- section 3.7
def assign(x, centres):
    return ((x[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)


def kmeans(x, k, rng, iters=30):
    """Lloyd's algorithm with k-means++ seeding. Returns centres, labels and inertia.
    Asks for at most as many centres as there are distinct points."""
    if len(x) == 0:
        raise ValueError("No profiles to cluster")
    k = min(k, len(np.unique(x, axis=0)))
    centres = [x[rng.integers(len(x))]]
    while len(centres) < k:                                  # k-means++: far points more likely
        d = np.min([((x - c) ** 2).sum(axis=1) for c in centres], axis=0)
        if d.sum() == 0:
            break                                            # every point already sits on a centre
        centres.append(x[rng.choice(len(x), p=d / d.sum())])
    centres = np.array(centres)
    for _ in range(iters):
        labels = assign(x, centres)
        new = np.array([x[labels == c].mean(axis=0) if (labels == c).any() else centres[c]
                        for c in range(len(centres))])
        if np.allclose(new, centres):
            break
        centres = new
    labels = assign(x, centres)                              # labels match the final centres
    inertia = ((x - centres[labels]) ** 2).sum()
    return centres, labels, inertia


def group_popularity(weight, group, n_groups):
    """Engaged users per item within each group; fewer than MIN_BUYERS -> not eligible."""
    counts = np.zeros((n_groups, weight.shape[1]))
    for g in range(n_groups):
        counts[g] = (weight[group == g] > 0).sum(axis=0)
    return np.where(counts >= MIN_BUYERS, counts, NO)


def segment_scores(weight, profile, fallback, n_segments, seed=0):
    """Popular in the user's k-means segment; users with no history get the fallback."""
    has = (weight > 0).any(axis=1)
    centres, seg, _ = kmeans(profile[has], n_segments, np.random.default_rng(seed))
    out = fallback.copy()
    out[np.flatnonzero(has)] = group_popularity(weight[has], seg, len(centres))[seg]
    return out


# ---------------------------------------------------------------- section 3.8
def top_k(scores, used, eligible_items, k=10):
    """Best k items per row that are eligible, unused and scored: highest rounded
    score first, ties to the lower item id. -1 fills slots with no such item."""
    s = np.where(used | ~eligible_items[None, :], NO, np.round(scores, DECIMALS))
    best = np.argsort(-s, axis=1, kind="stable")[:, :k]
    return np.where(np.isfinite(np.take_along_axis(s, best, axis=1)), best, -1)


def hit_rate(top, rows, positives, members=None):
    """Share of users (in members, default all rows) with a test positive in their list."""
    members = set(rows) if members is None else set(members)
    return np.mean([len(positives[u] & set(t.tolist())) > 0
                    for u, t in zip(rows, top) if u in members])


def recommenders(weight, feats, language, item_language, n_segments, k, strength):
    """Score matrices for every method; NO marks items a method may not recommend."""
    has = (weight > 0).any(axis=1)[:, None]
    same = (language[:, None] == item_language[None, :]).astype(np.float64)
    pop = group_popularity(weight, np.zeros(len(weight), dtype=np.int64), 1)[0]
    lang_pop = group_popularity(weight, language, len(LANGUAGES))[language]
    content, profile = content_scores(weight, feats)
    cf = cf_scores(weight, item_neighbours(weight, k=k, strength=strength))
    return {"popular worldwide": np.broadcast_to(pop, weight.shape),
            "popular in your language": lang_pop,
            "popular in your segment": segment_scores(weight, profile, lang_pop, n_segments),
            "content, in your language": np.where(has, np.where(content > 0, same + 0.5 * content, NO),
                                                  lang_pop),
            "item-based CF": cf,
            "CF, then popular in language": np.where(np.isfinite(cf), 1e6 + cf, lang_pop)}


if __name__ == "__main__":
    worked_example()
    cfg = Config()
    users, items, served, events, orders, truth = build(cfg)
    n_users, n_items = cfg.num_users, cfg.num_items
    feats = item_features(items, n_items)
    video = ~items["is_product"]                              # the "Because you watched" row

    # Choose settings on the validation week: train to day 16, test on days 16 to 23
    w16, used16 = user_item_scores(events, users, 16 * DAY, n_users, n_items)
    pos16 = test_positives(events, users, 16 * DAY, 23 * DAY, used16, video)
    r16 = np.array(sorted(pos16))
    print(f"\nValidation week (train to day 16, test days 16-23), {len(r16):,} users, hit rate@10:")
    best = None
    for k in (20, 50, 200):
        for strength in (0.0, 5.0, 20.0):
            cf = cf_scores(w16[r16], item_neighbours(w16, k=k, strength=strength))
            hr = hit_rate(top_k(cf, used16[r16], video), r16, pos16)
            print(f"  item-based CF, {k:>3} neighbours, shrinkage strength {strength:>4.0f}: {hr:.3f}"
                  f" ({round(hr * len(r16))} users)")
            if best is None or hr > best[0]:                  # ties keep the smaller setting
                best = (hr, k, strength)
    lang16 = group_popularity(w16, users["language"], len(LANGUAGES))[users["language"]]
    _, profile16 = content_scores(w16, feats)
    seg_best = None
    for n_seg in (4, 8, 16, 32):
        seg = segment_scores(w16, profile16, lang16, n_seg)[r16]
        hr = hit_rate(top_k(seg, used16[r16], video), r16, pos16)
        print(f"  popular in your segment, {n_seg:>2} segments: {hr:.3f} ({round(hr * len(r16))} users)")
        if seg_best is None or hr > seg_best[0]:
            seg_best = (hr, n_seg)
    print(f"Chosen: {best[1]} neighbours, strength {best[2]:.0f}, {seg_best[1]} segments")

    # Final test: train to day 23, test on days 23 to 30, settings fixed above
    weight, used = user_item_scores(events, users, 23 * DAY, n_users, n_items)
    pos = test_positives(events, users, 23 * DAY, 30 * DAY, used, video)
    rows = np.array(sorted(pos))
    history = (weight > 0).sum(axis=1)
    print(f"\nAt day 23: {(history > 0).sum():,} users with history, {history[history > 0].mean():.1f} "
          f"engaged items each on average; {used.sum():,} (user, item) pairs excluded as used")
    scores = recommenders(weight, feats, users["language"], items["language"],
                          seg_best[1], best[1], best[2])
    match = users["vec"].astype(np.float64) @ items["vec"].T        # hidden truth, simulation only
    scores["hidden taste (simulation only)"] = match
    scores["taste + popularity (simulation)"] = 5 * match + np.log(items["popularity"])
    groups = {"all": rows, "0-4 items": rows[history[rows] <= 4], "5+ items": rows[history[rows] >= 5]}
    print(f"Test week (train to day 23, test days 23-30), videos only, hit rate@10")
    print(f"{'':<32}" + "".join(f"{g:>11}" for g in groups) + f"{'full lists':>12}")
    print(f"{'users':<32}" + "".join(f"{len(v):>11,}" for v in groups.values()))
    for name, s in scores.items():
        top = top_k(s[rows], used[rows], video)
        print(f"{name:<32}" + "".join(f"{hit_rate(top, rows, pos, v):>11.3f}" for v in groups.values())
              + f"{(top[:, -1] >= 0).mean():>12.0%}")

    # "New for you", a history-masking test: hide the history of 10% of videos,
    # as if they had just arrived, and rank only those videos
    new = (np.random.default_rng(cfg.seed + 303).random(n_items) < 0.10) & video
    w_old = np.where(new[None, :], 0.0, weight)
    pos_new = {u: s & set(np.flatnonzero(new).tolist()) for u, s in pos.items()}
    pos_new = {u: s for u, s in pos_new.items() if s}
    rn = np.array(sorted(pos_new))
    content, _ = content_scores(w_old[rn], feats)
    same_lang = (users["language"][rn, None] == items["language"][None, :]).astype(np.float64)
    rng = np.random.default_rng(1)
    print(f"\n'New for you' row: {new.sum()} videos with hidden history, {len(rn):,} users with a hit on one")
    for name, base in (("random order", 0.0), ("in your language, random order", same_lang),
                       ("in your language, by content", same_lang + 0.5 * content)):
        runs = [hit_rate(top_k(base + 1e-3 * rng.random((len(rn), n_items)), used[rn], new), rn, pos_new)
                for _ in range(20)]                           # random tie-breaks: average 20 draws
        print(f"  {name:<32} hit rate@10 {np.mean(runs):.3f}")
