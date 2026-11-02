"""Tesselo synthetic data generator (Chapter 2, full version of Listing 2.1).

Builds a small Tesselo world (users, items, creators) and simulates 30 days of
home-feed traffic. A simple "current production system" picks each slate, so the
logs contain the same biases real logs do:

  * popularity bias  - candidates are sampled in proportion to item popularity
  * exposure bias    - items never sampled are never shown, so they get no feedback
  * position bias    - users look at lower slots less often (position-based click model)
  * clickbait        - some items attract clicks but do not satisfy

A small share of slots (5 percent) is filled with a uniformly random item and
flagged, in the spirit of Kuaishou's KuaiRand dataset. Those slots let us measure
the biases (see the report printed by main()).

Every number here is a toy setting chosen to make the effects visible, not a
measurement from a real platform.

Needs Python 3.12 or later. Tested with Python 3.13 and 3.14, NumPy 2.5.4. Run from the repository root:
    pip install -r requirements.txt -c constraints.txt
    python chapter_02/tesselo_data.py
Repository: https://github.com/amitpandeygit/ai-recommender-systems (folder chapter_02)
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class Config:
    num_users: int = 2_000
    num_items: int = 5_000
    num_creators: int = 300
    dim: int = 16                   # size of the hidden taste vectors
    days: int = 30
    requests_per_user: float = 30.0  # average home-feed visits per user in 30 days
    slate_size: int = 10
    candidates: int = 100           # items the current system considers per request
    random_slot_rate: float = 0.05  # share of slots given to a uniformly random item
    exam_decay: float = 0.8         # chance of looking at slot k is 1 / k**exam_decay
    clickbait_share: float = 0.10
    product_share: float = 0.30     # the rest of the catalogue is video
    anonymous_share: float = 0.20   # share of visits made while logged out
    no_consent_share: float = 0.10  # users who turned personalisation off
    seed: int = 7


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def make_world(cfg, rng):
    """Users, items and creators, with hidden tastes the system cannot see."""
    def unit(n):
        v = rng.normal(size=(n, cfg.dim)).astype(np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    users = {
        "vec": unit(cfg.num_users),
        "device_id": np.arange(cfg.num_users) + 900_000,   # one main phone per user
        "consent": rng.random(cfg.num_users) > cfg.no_consent_share,
    }
    rank = rng.permutation(cfg.num_items) + 1
    popularity = 1.0 / rank                                 # Zipf-like: a few items are very popular
    items = {
        "vec": unit(cfg.num_items),
        "creator": rng.integers(0, cfg.num_creators, cfg.num_items),
        "is_product": rng.random(cfg.num_items) < cfg.product_share,
        "clickbait": np.where(rng.random(cfg.num_items) < cfg.clickbait_share, 1.5, 0.0),
        "popularity": popularity / popularity.sum(),
        "duration_s": rng.integers(15, 600, cfg.num_items),  # video length in seconds
    }
    return users, items


def choose_slates(cfg, rng, users, items, req_user):
    """The current production system: sample popular candidates, rank by a noisy match.

    Users without personalisation consent get the most popular candidates instead
    (a non-profiling feed, as the EU Digital Services Act requires very large
    platforms to offer).
    """
    n_req, k = len(req_user), cfg.slate_size
    slates = np.empty((n_req, k), dtype=np.int64)
    for start in range(0, n_req, 5_000):                    # batches keep memory small
        u = req_user[start:start + 5_000]
        cands = rng.choice(cfg.num_items, size=(len(u), cfg.candidates), p=items["popularity"])
        match = np.einsum("rd,rcd->rc", users["vec"][u], items["vec"][cands])
        score = match + rng.normal(0, 0.1, match.shape)
        score[~users["consent"][u]] = items["popularity"][cands[~users["consent"][u]]]
        order = np.argsort(-score, axis=1)
        ranked = np.take_along_axis(cands, order, axis=1)
        for r in range(len(u)):                             # top k, no duplicates
            _, first = np.unique(ranked[r], return_index=True)
            slates[start + r] = ranked[r][np.sort(first)][:k]
    random_slot = rng.random((n_req, k)) < cfg.random_slot_rate
    slates[random_slot] = rng.integers(0, cfg.num_items, random_slot.sum())
    return slates, random_slot


def simulate(cfg=Config()):
    """Return the world plus an impression log and an event log (dicts of arrays)."""
    rng = np.random.default_rng(cfg.seed)
    users, items = make_world(cfg, rng)

    n_req = int(cfg.num_users * cfg.requests_per_user)
    req_user = rng.integers(0, cfg.num_users, n_req)
    req_time = np.sort(rng.random(n_req) * cfg.days * 86_400)   # seconds since day 0
    logged_in = rng.random(n_req) > cfg.anonymous_share
    slates, random_slot = choose_slates(cfg, rng, users, items, req_user)

    # Flatten to one row per impression (one item shown in one slot of one request)
    k = cfg.slate_size
    imp = {
        "request_id": np.repeat(np.arange(n_req), k),
        "user_id": np.repeat(np.where(logged_in, req_user, -1), k),   # -1 = logged out
        "device_id": np.repeat(users["device_id"][req_user], k),
        "true_user": np.repeat(req_user, k),       # hidden truth, for teaching only
        "time_s": np.repeat(req_time, k),
        "item_id": slates.ravel(),
        "position": np.tile(np.arange(1, k + 1), n_req),
        "random_slot": random_slot.ravel(),
    }

    # Listing 2.1 (the key lines): the position-based click model
    u, i = imp["true_user"], imp["item_id"]
    match = np.einsum("nd,nd->n", users["vec"][u], items["vec"][i])
    p_look = 1.0 / imp["position"] ** cfg.exam_decay             # position bias
    p_attract = sigmoid(5 * match - 3.5 + items["clickbait"][i])  # clickbait lifts clicks...
    p_satisfy = sigmoid(5 * match - 2.0 - 2 * items["clickbait"][i])  # ...but not satisfaction
    looked = rng.random(len(i)) < p_look
    clicked = looked & (rng.random(len(i)) < p_attract)
    satisfied = clicked & (rng.random(len(i)) < p_satisfy)
    imp["clicked"] = clicked

    # What happens after a click
    is_video = ~items["is_product"][i]
    watch_frac = np.where(satisfied, rng.beta(5, 2, len(i)), rng.beta(1, 6, len(i)))
    rows = []                                                  # (impression index, event, value)
    def add(mask, name, value=None):
        idx = np.flatnonzero(mask)
        vals = np.ones(len(idx)) if value is None else value[idx]
        rows.append((idx, np.full(len(idx), name, dtype=object), vals))
    add(clicked, "click")
    v = clicked & is_video
    add(v, "watch_seconds", watch_frac * items["duration_s"][i])
    add(v & (watch_frac >= 0.8), "complete")
    add(v & (watch_frac < 0.1), "skip")
    add(satisfied & is_video & (rng.random(len(i)) < 0.25), "like")
    add(clicked & ~satisfied & (rng.random(len(i)) < 0.05), "not_interested")
    cart = satisfied & ~is_video & (rng.random(len(i)) < 0.30)
    add(cart, "add_to_cart")
    add(cart & (rng.random(len(i)) < 0.50), "purchase")

    idx = np.concatenate([r[0] for r in rows])
    order = np.argsort(idx, kind="stable")
    idx = idx[order]
    events = {
        "request_id": imp["request_id"][idx],
        "user_id": imp["user_id"][idx],
        "device_id": imp["device_id"][idx],
        "item_id": imp["item_id"][idx],
        "time_s": imp["time_s"][idx] + rng.uniform(1, 120, len(idx)),
        "event": np.concatenate([r[1] for r in rows])[order],
        "value": np.concatenate([r[2] for r in rows])[order],
    }
    return users, items, imp, events


def report(cfg, items, imp, events):
    """Measure the biases the generator planted, using the random slots."""
    k = cfg.slate_size
    pos, rnd, clk = imp["position"], imp["random_slot"], imp["clicked"]
    print(f"Impressions: {len(pos):,}   events: {len(events['event']):,}   "
          f"random slots: {rnd.mean():.1%}")
    ctr_normal = np.array([clk[(pos == p) & ~rnd].mean() for p in range(1, k + 1)])
    ctr_random = np.array([clk[(pos == p) & rnd].mean() for p in range(1, k + 1)])
    true_look = 1.0 / np.arange(1, k + 1) ** cfg.exam_decay
    print("\nPosition  CTR(normal)  CTR(random)  random/random@1  true look rate")
    for p in range(k):
        print(f"{p + 1:>8}  {ctr_normal[p]:>11.3f}  {ctr_random[p]:>11.3f}"
              f"  {ctr_random[p] / ctr_random[0]:>15.2f}  {true_look[p]:>14.2f}")

    shown = np.bincount(imp["item_id"][~rnd], minlength=cfg.num_items)
    top1 = np.argsort(-items["popularity"])[: cfg.num_items // 100]
    print(f"\nTop 1% most popular items get {shown[top1].sum() / shown.sum():.1%} of normal impressions"
          f" and {np.isin(imp['item_id'][rnd], top1).mean():.1%} of random ones.")
    bottom = np.argsort(items["popularity"])[: cfg.num_items // 2]
    print(f"The less popular half of the catalogue gets {shown[bottom].sum() / shown.sum():.1%} of normal"
          f" impressions and {np.isin(imp['item_id'][rnd], bottom).mean():.1%} of random ones.")

    bait = items["clickbait"][imp["item_id"]] > 0
    video = ~items["is_product"][imp["item_id"]]
    ev_watch = events["event"] == "watch_seconds"
    wf = events["value"][ev_watch] / items["duration_s"][events["item_id"][ev_watch]]
    bait_ev = items["clickbait"][events["item_id"][ev_watch]] > 0
    print(f"\nClickbait videos: CTR {clk[bait & video].mean():.3f}, "
          f"average share watched {wf[bait_ev].mean():.0%}")
    print(f"Other videos:     CTR {clk[~bait & video].mean():.3f}, "
          f"average share watched {wf[~bait_ev].mean():.0%}")


if __name__ == "__main__":
    cfg = Config()
    users, items, imp, events = simulate(cfg)
    report(cfg, items, imp, events)
