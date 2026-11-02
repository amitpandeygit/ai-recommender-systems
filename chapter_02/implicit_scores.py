"""Listings 2.1 and 2.2 - From raw logs to implicit scores (Tesselo, Chapter 2).

Two jobs, kept apart:

1. prepare(): turn raw logs into eligible training events. Remove duplicates,
   keep only events that had reached the server by the training cutoff (before
   deriving anything from them), estimate action times, link identities with a
   careful same-visit rule, and check consent at the time and now. The generator
   already supplies impression IDs and sequence numbers; joining actions to
   impressions and validating times are covered in Chapter 28.
2. Score the eligible events in three steps:
     step 1  event weights  - a purchase says more than a click
     step 2  time decay     - each event counts half as much every HALF_LIFE_DAYS days
     step 3  rarity weight  - optional; positive scores on items fewer users engage
                              with count more: 1 + ln((1 + N) / (1 + max(n, min(MIN_USERS, N))))
   "Not interested" also sets a separate block flag. Blocked items are filtered
   out before any ranking, whatever their score.

Item statistics (N and n) are computed once, from a defined population and
window, and passed in. The worked example in section 2.5 runs through the same
functions.

This is a laptop-scale, in-memory implementation for teaching. Chapter 28 shows
how the same steps run on a data platform.

Needs Python 3.12 or later. Tested with Python 3.13 and 3.14, NumPy 2.5.4. Run from the repository root:
    pip install -r requirements.txt -c constraints.txt
    python chapter_02/implicit_scores.py
Repository: https://github.com/amitpandeygit/ai-recommender-systems (folder chapter_02)
"""
import numpy as np

from tesselo_data import Config, corrected_time, simulate

EVENT_WEIGHTS = {"click": 1, "complete": 3, "like": 4, "add_to_cart": 3,
                 "purchase": 6, "skip": -2, "not_interested": -5}
UNSCORED = {"viewable", "watch_end", "login"}      # logged on purpose, but not part of the score
BLOCK_EVENTS = {"not_interested"}
HALF_LIFE_DAYS = 14.0
MIN_USERS = 5                                      # rarity treats fewer engaged users as 5


def check_event_names(event):
    """Refuse unknown event names instead of silently scoring them as zero."""
    unknown = set(np.unique(event)) - EVENT_WEIGHTS.keys() - UNSCORED
    if unknown:
        raise ValueError(f"Unknown event names: {sorted(unknown)}")


def link_same_visit(events):
    """Link logged-out events to a user only if, later in the same visit, exactly one
    account logged in, from the same app install.

    This is a careful rule, not proof: it assumes one person used the phone for the
    whole visit. It is retrospective, so it is used only to build training data. At
    serving time, before the login, the request was anonymous. Visits with logins
    from more than one account are left anonymous, and so is everything else.
    """
    user = events["user_id"].copy()
    login = np.flatnonzero(events["event"] == "login")
    owners = {}
    for s, u, a in zip(events["session_id"][login], events["user_id"][login],
                       events["anonymous_id"][login]):
        owners.setdefault(s, set()).add((u, a))
    anon = np.flatnonzero(user < 0)
    for j in anon:
        found = owners.get(events["session_id"][j])
        if found and len(found) == 1:                       # exactly one account logged in
            u, a = next(iter(found))
            if a == events["anonymous_id"][j]:              # same app install
                user[j] = u
    return user


def prepare(events, users, cutoff):
    """Raw logs -> eligible training events (user, item, event, age in days)."""
    first = np.unique(events["event_id"], return_index=True)[1]          # 1. remove duplicates
    e = {key: val[first] for key, val in events.items()}
    arrived = e["server_time"] < cutoff                                  # 2. only what had arrived
    e = {key: val[arrived] for key, val in e.items()}                    #    by the cutoff, before
    check_event_names(e["event"])                                        #    anything is derived
    t = corrected_time(e)                                                # 3. estimated action time
    user = link_same_visit(e)                                            # 4. identity
    keep = ((t < cutoff) & (user >= 0) & (e["item_id"] >= 0)
            & e["consent_at_time"]                                       # 5. consent then...
            & users["current_consent"][np.maximum(user, 0)])             #    ...and now
    return {"user": user[keep], "item": e["item_id"][keep], "event": e["event"][keep],
            "age_days": (cutoff - t[keep]) / 86_400}


def pair_sums(user_idx, item_idx, event, age_days):
    """Steps 1 and 2: one decayed sum, and one block flag, per (user, item) pair."""
    weight = np.array([EVENT_WEIGHTS.get(e, 0) for e in event], dtype=float)   # UNSCORED -> 0
    decayed = weight * 0.5 ** (age_days / HALF_LIFE_DAYS)
    pairs, inverse = np.unique(np.stack([user_idx, item_idx], axis=1), axis=0, return_inverse=True)
    raw = np.bincount(inverse.ravel(), weights=decayed, minlength=len(pairs))
    is_block = np.isin(event, list(BLOCK_EVENTS)).astype(float)
    blocked = np.bincount(inverse.ravel(), weights=is_block, minlength=len(pairs)) > 0
    return pairs, raw, blocked


def rarity_weights(users_engaged, n_users):
    """Step 3: smoothed rarity per item, from population statistics passed in.

    users_engaged[i] = number of users in the population whose pair with item i has a
    positive decayed sum and is not blocked. The weight is always at least 1.
    """
    users_engaged = np.asarray(users_engaged)
    if n_users < 1:
        raise ValueError("The scoring population is empty")
    if (users_engaged < 0).any() or (users_engaged > n_users).any():
        raise ValueError("Each engaged count must be between 0 and the population size")
    floor = min(MIN_USERS, n_users)                     # the floor can never exceed N
    return 1.0 + np.log((1 + n_users) / (1 + np.maximum(users_engaged, floor)))


def implicit_scores(pairs, raw, rarity=None):
    """Final score per pair. rarity=None turns step 3 off."""
    if rarity is None:
        return raw
    return np.where(raw > 0, raw * rarity[pairs[:, 1]], raw)


def worked_example():
    """Section 2.5: Maya's four items, run through the same functions."""
    names = ["A cooking video", "B phone case", "C clickbait video", "D song video"]
    rows = [(0, "click", 2), (0, "complete", 2), (0, "like", 2),
            (1, "click", 20), (1, "add_to_cart", 20), (1, "purchase", 20),
            (2, "click", 1), (2, "skip", 1),
            (3, "not_interested", 5)]
    item = np.array([r[0] for r in rows])
    event = np.array([r[1] for r in rows], dtype=object)
    age = np.array([r[2] for r in rows], dtype=float)
    check_event_names(event)
    pairs, raw, blocked = pair_sums(np.zeros(len(rows), dtype=int), item, event, age)
    engaged, n_users = np.array([500, 20_000, 8_000, 2_000]), 100_000   # population statistics
    rarity = rarity_weights(engaged, n_users)
    score = implicit_scores(pairs, raw, rarity)
    print("Item               decayed sum  users  rarity    score  blocked")
    for p, r, b, s in zip(pairs, raw, blocked, score):
        print(f"{names[p[1]]:<18} {r:>11.3f} {engaged[p[1]]:>6}  {rarity[p[1]]:.4f} {s:>8.2f}  {b}")


def clickbait_share_in_top10(pairs, score, blocked, clickbait):
    """Rerank each user's already-used items; skip blocked items and scores <= 0."""
    ok = ~blocked & (score > 0)
    p, s = pairs[ok], score[ok]
    order = np.lexsort((-s, p[:, 0]))                       # by user, then best score first
    p = p[order]
    start = np.r_[0, np.flatnonzero(np.diff(p[:, 0])) + 1]
    rank = np.arange(len(p)) - np.repeat(start, np.diff(np.r_[start, len(p)]))
    top = p[rank < 10]
    n = np.bincount(top[:, 0])
    has = n > 0
    return (np.bincount(top[:, 0], weights=clickbait[top[:, 1]] > 0)[has] / n[has]).mean()


if __name__ == "__main__":
    worked_example()

    cfg = Config()
    users, items, served, events = simulate(cfg)
    cutoff = cfg.days * 86_400                                # train at the end of day 30

    # Identity: same-visit linking versus "last user seen on this phone"
    first = np.unique(events["event_id"], return_index=True)[1]
    e = {key: val[first] for key, val in events.items()}
    anon = e["user_id"] < 0
    safe = link_same_visit(e)
    known = e["user_id"] >= 0
    last_user = dict(zip(e["device_id"][known], e["user_id"][known]))   # whole month, incl. the future
    naive = np.array([last_user.get(d, -1) for d in e["device_id"]])
    for label, linked in (("same-visit login", safe), ("last user on phone", naive)):
        got = anon & (linked >= 0)
        wrong = got & (linked != e["_true_user"])
        print(f"\nLogged-out events linked by {label:<19}: {got.sum() / anon.sum():.1%}, "
              f"wrongly linked: {wrong.sum() / max(got.sum(), 1):.1%}", end="")
    print()

    # Training data at the cutoff
    data = prepare(events, users, cutoff)
    print(f"Eligible training events: {len(data['event']):,}")
    u_keys, u_idx = np.unique(data["user"], return_inverse=True)   # dense indices for any ID type
    i_keys, i_idx = np.unique(data["item"], return_inverse=True)
    pairs, raw, blocked = pair_sums(u_idx, i_idx, data["event"], data["age_days"])
    n_users = len(u_keys)                                       # the scoring population
    engaged = np.bincount(pairs[(raw > 0) & ~blocked, 1], minlength=len(i_keys))
    rarity = rarity_weights(engaged, n_users)
    print(f"Scoring population: {n_users:,} users with eligible events; "
          f"blocked (user, item) pairs: {blocked.sum():,}")

    _, inverse = np.unique(np.stack([u_idx, i_idx], axis=1), axis=0, return_inverse=True)
    clicks = np.bincount(inverse.ravel(), weights=data["event"] == "click", minlength=len(pairs))
    bait = items["clickbait"][i_keys]
    print(f"Clickbait share of each user's top 10 already-used items "
          f"(catalogue share {np.mean(items['clickbait'] > 0):.1%}):")
    for label, s in (("ranked by clicks", clicks),
                     ("implicit score, no rarity", implicit_scores(pairs, raw)),
                     ("implicit score, with rarity", implicit_scores(pairs, raw, rarity))):
        print(f"  {label:<28} {clickbait_share_in_top10(pairs, s, blocked, bait):.1%}")
