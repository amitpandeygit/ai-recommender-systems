"""Listing 2.2 - Turning logged behaviour into implicit scores (Tesselo, Chapter 2).

Three steps, applied in this order:
  1. event weights  - a purchase says more than a click; "not interested" is negative
  2. time decay     - each event counts half as much every HALF_LIFE_DAYS days
  3. rarity weight  - positive scores on items few people engage with count more
                      (the IDF idea from text search: weight = ln(N_users / users_engaged))

The first part reproduces the worked example in section 2.5 step by step.
The second part applies the same function to the logs from tesselo_data.py, after
stitching logged-out visits to their user through the device ID.

Needs Python 3.12 or later. Tested with Python 3.13 and 3.14, NumPy 2.5.4. Run from the repository root:
    pip install -r requirements.txt -c constraints.txt
    python chapter_02/implicit_scores.py
Repository: https://github.com/amitpandeygit/ai-recommender-systems (folder chapter_02)
"""
import numpy as np

from tesselo_data import Config, simulate

EVENT_WEIGHTS = {"click": 1, "complete": 3, "like": 4, "add_to_cart": 3,
                 "purchase": 6, "skip": -2, "not_interested": -5}   # watch_seconds: not scored
HALF_LIFE_DAYS = 14.0


def implicit_scores(user, item, event, age_days, n_users):
    """Return (pairs, scores): one score per distinct (user, item) pair."""
    if len(event) == 0:
        return np.empty((0, 2), dtype=np.int64), np.empty(0)
    weight = np.array([EVENT_WEIGHTS.get(e, 0) for e in event], dtype=float)
    decayed = weight * 0.5 ** (age_days / HALF_LIFE_DAYS)            # step 2: time decay
    pairs, inverse = np.unique(np.stack([user, item], axis=1), axis=0, return_inverse=True)
    raw = np.bincount(inverse.ravel(), weights=decayed)               # step 1: sum per pair
    positive = raw > 0
    users_engaged = np.bincount(pairs[positive, 1], minlength=item.max() + 1)
    rarity = np.log(n_users / np.maximum(users_engaged[pairs[:, 1]], 1))
    scores = np.where(positive, raw * rarity, raw)                    # step 3: positives only
    return pairs, scores


def worked_example():
    """Section 2.5: Maya's four items, with every intermediate number printed."""
    n_users = 100_000
    rows = [  # (item, event, age in days)
        ("A cooking video", "click", 2), ("A cooking video", "complete", 2), ("A cooking video", "like", 2),
        ("B phone case", "click", 20), ("B phone case", "add_to_cart", 20), ("B phone case", "purchase", 20),
        ("C clickbait video", "click", 1), ("C clickbait video", "skip", 1),
        ("D song video", "not_interested", 5),
    ]
    engaged = {"A cooking video": 500, "B phone case": 20_000,
               "C clickbait video": 8_000, "D song video": 2_000}
    print("Item               raw  age  decay   decayed  users  rarity   score")
    for name in engaged:
        evs = [(e, age) for n, e, age in rows if n == name]
        raw = sum(EVENT_WEIGHTS[e] for e, _ in evs)
        age = evs[0][1]
        decay = 0.5 ** (age / HALF_LIFE_DAYS)
        rarity = np.log(n_users / engaged[name])
        score = raw * decay * rarity if raw > 0 else raw * decay
        print(f"{name:<17} {raw:>4} {age:>4}  {decay:.4f} {raw * decay:>8.3f} {engaged[name]:>6}"
              f"  {rarity:>6.4f} {score:>7.2f}")


def stitch(events):
    """Give logged-out events the user ID last seen on the same device."""
    user = events["user_id"].copy()
    known = user >= 0
    device_owner = dict(zip(events["device_id"][known], user[known]))
    missing = np.flatnonzero(~known)
    user[missing] = [device_owner.get(d, -1) for d in events["device_id"][missing]]
    return user


if __name__ == "__main__":
    worked_example()

    cfg = Config()
    users, items, imp, events = simulate(cfg)
    user = stitch(events)
    print(f"\nEvents without a user ID: {(events['user_id'] < 0).mean():.1%} before stitching, "
          f"{(user < 0).mean():.1%} after")

    keep = (user >= 0) & users["consent"][np.maximum(user, 0)]    # no profiling without consent
    print(f"Events dropped because the user turned personalisation off: "
          f"{((user >= 0) & ~keep).mean():.1%}")
    now_s = events["time_s"].max()
    age_days = (now_s - events["time_s"][keep]) / 86_400
    pairs, scores = implicit_scores(user[keep], events["item_id"][keep],
                                    events["event"][keep], age_days, cfg.num_users)

    # Compare each user's top 10 by click count with their top 10 by implicit score
    clicks = events["event"][keep] == "click"
    cpairs, ccount = np.unique(np.stack([user[keep][clicks], events["item_id"][keep][clicks]], axis=1),
                               axis=0, return_counts=True)
    def bait_share(p, s):
        share = []
        for u in np.unique(p[:, 0]):
            rows = np.flatnonzero(p[:, 0] == u)
            top = p[rows[np.argsort(-s[rows], kind="stable")[:10]], 1]
            share.append((items["clickbait"][top] > 0).mean())
        return np.mean(share)
    print(f"Clickbait share of each user's top 10: {bait_share(cpairs, ccount):.1%} by clicks, "
          f"{bait_share(pairs, scores):.1%} by implicit score "
          f"(catalogue share {np.mean(items['clickbait'] > 0):.1%})")
