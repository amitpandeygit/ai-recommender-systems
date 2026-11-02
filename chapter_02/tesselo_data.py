"""Tesselo synthetic data generator (Chapter 2, full version of Listing 2.3).

Builds a small Tesselo world and simulates 30 days of home-feed visits. The logs
it writes have the same structure, and the same problems, as real logs:

  * popularity and exposure bias - the current system samples popular candidates
  * position bias                - users scroll away and look less at lower slots
  * clickbait                    - some items attract clicks but do not satisfy
  * identity gaps                - logged-out sessions and shared household phones
  * time problems                - offline phones upload late, some phone clocks are wrong
  * duplicates                   - weak networks resend some events

About 10% of slots are exploratory (more than a real platform would use, so
that this small simulation has enough data): the item is drawn uniformly from the
catalogue items not already in that slate, and the slot is flagged with its
probabilities. These slots let us measure the biases (see report()).

Every number here is a toy setting chosen to make the effects visible, not a
measurement from a real platform. Fields whose names start with "_" are the
hidden truth of the simulation. A real log never has them; we keep them only to
check whether a method recovers the truth.

Needs Python 3.12 or later. Tested with Python 3.13 and 3.14, NumPy 2.5.4. Run from the repository root:
    pip install -r requirements.txt -c constraints.txt
    python chapter_02/tesselo_data.py
Repository: https://github.com/amitpandeygit/ai-recommender-systems (folder chapter_02)

Tables returned by simulate()
-----------------------------
users  : one row per user.  current_consent, consent_off_from (time personalisation was turned off)
items  : one row per item.  creator, is_product, duration_s, popularity, clickbait (hidden)
served : one row per item the server sent (a served slot).
         impression_id, request_id, session_id, position, item_id, served_time,
         user_id (-1 if logged out at serve time), anonymous_id, device_id,
         consent_at_time, explored, explore_rate, explore_item_prob
events : one row per event the app or server logged.
         event_id, event, impression_id (-1 for login), request_id, session_id,
         user_id, anonymous_id, device_id, item_id, value,
         client_time (phone clock), client_send_time (phone clock at upload),
         server_time (when the server received it), seq (per-device counter),
         consent_at_time
All times are seconds since the start of day 0.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class Config:
    num_users: int = 4_000
    num_items: int = 5_000
    num_creators: int = 300
    dim: int = 16                    # size of the hidden taste vectors
    days: int = 30
    sessions_per_user: float = 10.0  # average visits per user in 30 days
    max_requests: int = 4            # a visit loads 1 to 4 slates
    slate_size: int = 10
    candidates: int = 100            # items the current system considers per request
    explore_rate: float = 0.10       # chance that a slot is exploratory (high, so the toy has enough data)
    scroll_keep: float = 0.92        # chance the user scrolls on from one slot to the next
    look_decay: float = 0.5          # P(look | slot k on screen) = 1 / k**look_decay
    glimpse_look: float = 0.3        # a slot passed in under a second is looked at 30% as often
    clickbait_share: float = 0.10
    product_share: float = 0.30      # the rest of the catalogue is video
    logged_out_share: float = 0.25   # visits that start logged out
    login_midway_share: float = 0.30  # of those, visits where the user logs in partway through
    shared_device_share: float = 0.10  # users with no phone of their own, sharing a household phone
    offline_share: float = 0.15      # visits uploaded later, in one batch
    skewed_clock_share: float = 0.03  # phones whose clock is wrong
    duplicate_share: float = 0.01    # events sent twice
    consent_off_share: float = 0.10  # users with personalisation off from the start
    consent_withdraw_share: float = 0.05  # users who turn it off during the month
    seed: int = 7


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def make_world(cfg, rng):
    """Users, devices and items, with hidden tastes the system cannot see."""
    def unit(n):
        v = rng.normal(size=(n, cfg.dim)).astype(np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    n = cfg.num_users
    device = np.arange(n)                               # each user's own phone...
    shared = rng.random(n) < cfg.shared_device_share
    device[shared] = rng.integers(0, n, shared.sum())   # ...or a household phone shared with another user
    off_from = np.full(n, np.inf)
    off_from[rng.random(n) < cfg.consent_off_share] = 0.0
    withdraw = (off_from == np.inf) & (rng.random(n) < cfg.consent_withdraw_share)
    off_from[withdraw] = rng.random(withdraw.sum()) * cfg.days * 86_400
    users = {"vec": unit(n), "device": device, "consent_off_from": off_from,
             "current_consent": off_from == np.inf}
    skew = np.where(rng.random(n) < cfg.skewed_clock_share, rng.normal(0, 3 * 3_600, n), 0.0)

    rank = rng.permutation(cfg.num_items) + 1
    popularity = 1.0 / rank                             # Zipf-like: a few items are very popular
    items = {
        "vec": unit(cfg.num_items),
        "creator": rng.integers(0, cfg.num_creators, cfg.num_items),
        "is_product": rng.random(cfg.num_items) < cfg.product_share,
        "duration_s": rng.integers(15, 600, cfg.num_items),
        "popularity": popularity / popularity.sum(),
        "clickbait": np.where(rng.random(cfg.num_items) < cfg.clickbait_share, 1.5, 0.0),
    }
    return users, items, skew


def make_requests(cfg, rng, users):
    """Visits (sessions) and the requests inside them, with login state."""
    n_sess = rng.poisson(cfg.sessions_per_user, cfg.num_users)
    s_user = np.repeat(np.arange(cfg.num_users), n_sess)
    s_start = rng.random(len(s_user)) * (cfg.days * 86_400 - 3_600)
    s_nreq = rng.integers(1, cfg.max_requests + 1, len(s_user))
    starts_out = rng.random(len(s_user)) < cfg.logged_out_share
    login_at = np.full(len(s_user), -1)                 # request index at which the user logs in
    midway = starts_out & (s_nreq >= 2) & (rng.random(len(s_user)) < cfg.login_midway_share)
    login_at[midway] = rng.integers(1, s_nreq[midway])  # 1 .. n_req - 1
    login_at[~starts_out] = 0
    r_sess = np.repeat(np.arange(len(s_user)), s_nreq)
    r_idx = np.concatenate([np.arange(k) for k in s_nreq])
    r_time = s_start[r_sess] + r_idx * rng.uniform(60, 240, len(r_sess))
    r_user = s_user[r_sess]
    r_logged = (login_at[r_sess] >= 0) & (r_idx >= login_at[r_sess])
    return {"sess_user": s_user, "sess_start": s_start, "login_at": login_at,
            "req_session": r_sess, "req_index": r_idx, "req_time": r_time,
            "req_user": r_user, "req_logged_in": r_logged}


def choose_slates(cfg, rng, users, items, req_user, req_consent):
    """The current production system, plus a small share of exploratory slots."""
    n_req, k, n_items = len(req_user), cfg.slate_size, cfg.num_items
    slates = np.empty((n_req, k), dtype=np.int64)
    for start in range(0, n_req, 5_000):                # batches keep memory small
        sl = slice(start, start + 5_000)
        u = req_user[sl]
        cands = rng.choice(n_items, size=(len(u), cfg.candidates), p=items["popularity"])
        score = np.einsum("rd,rcd->rc", users["vec"][u], items["vec"][cands])
        score += rng.normal(0, 0.1, score.shape)
        no_profile = ~req_consent[sl]                   # personalisation off: rank by popularity
        score[no_profile] = items["popularity"][cands[no_profile]]
        ranked = np.take_along_axis(cands, np.argsort(-score, axis=1), axis=1)
        for r in range(len(u)):                         # top k, no duplicates
            _, first = np.unique(ranked[r], return_index=True)
            slates[start + r] = ranked[r][np.sort(first)][:k]

    # Exploration: replace the slot with an item drawn uniformly from the items
    # not already in this slate (so no slate repeats an item).
    explored = rng.random((n_req, k)) < cfg.explore_rate
    for r, p in zip(*np.nonzero(explored)):
        while True:
            pick = rng.integers(0, n_items)
            if pick not in slates[r]:
                slates[r, p] = pick
                break
    explore_item_prob = np.where(explored, 1.0 / (n_items - k), np.nan)
    return slates, explored, explore_item_prob


def simulate(cfg=Config()):
    """Return (users, items, served, events). See the module docstring for the fields."""
    rng = np.random.default_rng(cfg.seed)
    users, items, skew = make_world(cfg, rng)
    rq = make_requests(cfg, rng, users)
    n_req, k = len(rq["req_user"]), cfg.slate_size
    consent = rq["req_time"] < users["consent_off_from"][rq["req_user"]]
    slates, explored, explore_item_prob = choose_slates(cfg, rng, users, items, rq["req_user"], consent)

    # ---- served slots: one row per item sent ------------------------------------
    rep = lambda a: np.repeat(a, k)
    true_user = rep(rq["req_user"])
    served = {
        "impression_id": np.arange(n_req * k),
        "request_id": rep(np.arange(n_req)),
        "session_id": rep(rq["req_session"]),
        "position": np.tile(np.arange(1, k + 1), n_req),
        "item_id": slates.ravel(),
        "served_time": rep(rq["req_time"]),
        "user_id": np.where(rep(rq["req_logged_in"]), true_user, -1),
        "anonymous_id": users["device"][true_user] + 500_000,   # one app install per phone
        "device_id": users["device"][true_user] + 900_000,
        "consent_at_time": rep(consent),
        "explored": explored.ravel(),
        "explore_rate": np.full(n_req * k, cfg.explore_rate),
        "explore_item_prob": explore_item_prob.ravel(),
        "_true_user": true_user,
    }

    # ---- what the user saw and did (Listing 2.3, key lines) -----------------------
    pos, i, u = served["position"], served["item_id"], true_user
    depth = 1 + np.minimum(rng.geometric(1 - cfg.scroll_keep, n_req) - 1, k - 1)  # slots viewed
    depth = rep(depth)
    on_screen = pos <= depth                       # viewable: on screen for at least a second
    glimpse = pos == depth + 1                     # passed by in under a second
    p_look = np.where(on_screen, 1.0, np.where(glimpse, cfg.glimpse_look, 0.0)) / pos ** cfg.look_decay
    match = np.einsum("nd,nd->n", users["vec"][u], items["vec"][i])
    p_attract = sigmoid(5 * match - 3.5 + items["clickbait"][i])       # clickbait lifts clicks...
    p_satisfy = sigmoid(5 * match - 2.0 - 2 * items["clickbait"][i])  # ...but not satisfaction
    looked = rng.random(len(i)) < p_look
    clicked = looked & (rng.random(len(i)) < p_attract)
    satisfied = clicked & (rng.random(len(i)) < p_satisfy)
    served.update({"_viewable": on_screen, "_glimpsed": glimpse, "_looked": looked,
                   "_clicked": clicked})

    # ---- action times, in true behavioural order ----------------------------------
    t_screen = served["served_time"] + (pos - 1) * 2.5
    t_click = t_screen + np.where(glimpse, rng.uniform(0.2, 0.9, len(i)), rng.uniform(1, 6, len(i)))
    is_video = ~items["is_product"][i]
    watch_frac = np.where(satisfied, rng.beta(5, 2, len(i)), rng.beta(1, 6, len(i)))
    watch_s = watch_frac * items["duration_s"][i]
    t_end = t_click + watch_s
    t_cart = t_click + rng.uniform(5, 60, len(i))
    cart = satisfied & ~is_video & (rng.random(len(i)) < 0.30)
    rows = []

    def add(mask, name, t, value=None):
        idx = np.flatnonzero(mask)
        rows.append((idx, np.full(len(idx), name, dtype=object), t[idx],
                     np.ones(len(idx)) if value is None else value[idx]))

    add(on_screen, "viewable", t_screen + 1.0)
    add(clicked, "click", t_click)
    v = clicked & is_video
    add(v, "watch_end", t_end, watch_s)
    add(v & (watch_frac >= 0.8), "complete", t_end)
    add(v & (watch_frac < 0.1), "skip", t_end)
    add(satisfied & is_video & (rng.random(len(i)) < 0.25), "like", t_end + rng.uniform(0, 3, len(i)))
    add(clicked & ~satisfied & (rng.random(len(i)) < 0.05), "not_interested",
        np.where(is_video, t_end, t_click) + rng.uniform(1, 5, len(i)))
    add(cart, "add_to_cart", t_cart)
    add(cart & (rng.random(len(i)) < 0.50), "purchase", t_cart + rng.uniform(30, 600, len(i)))

    imp = np.concatenate([r[0] for r in rows])
    ev = {"event": np.concatenate([r[1] for r in rows]),
          "impression_id": imp, "request_id": served["request_id"][imp],
          "session_id": served["session_id"][imp], "item_id": served["item_id"][imp],
          "value": np.concatenate([r[3] for r in rows]),
          "_event_time": np.concatenate([r[2] for r in rows]),
          "_true_user": true_user[imp], "user_id": served["user_id"][imp],
          "consent_at_time": served["consent_at_time"][imp]}

    # Login events: a visit that started logged out and logs in partway through
    s_login = np.flatnonzero(rq["login_at"] > 0)
    first_req = np.searchsorted(rq["req_session"], s_login)
    login_req = first_req + rq["login_at"][s_login]
    t_login = rq["req_time"][login_req] - 5.0
    lu = rq["sess_user"][s_login]
    login_ev = {"event": np.full(len(s_login), "login", dtype=object),
                "impression_id": np.full(len(s_login), -1), "request_id": login_req,
                "session_id": s_login, "item_id": np.full(len(s_login), -1),
                "value": np.ones(len(s_login)), "_event_time": t_login, "_true_user": lu,
                "user_id": lu,
                "consent_at_time": t_login < users["consent_off_from"][lu]}
    ev = {key: np.concatenate([ev[key], login_ev[key]]) for key in ev}
    ev["anonymous_id"] = users["device"][ev["_true_user"]] + 500_000
    ev["device_id"] = users["device"][ev["_true_user"]] + 900_000

    # ---- delivery: separate from behaviour ------------------------------------------
    n_ev = len(ev["event"])
    sess = ev["session_id"]
    sess_end = np.full(len(rq["sess_user"]), -np.inf)
    np.maximum.at(sess_end, sess, ev["_event_time"])
    offline = rng.random(len(rq["sess_user"])) < cfg.offline_share
    reconnect = sess_end + rng.exponential(3 * 3_600, len(sess_end))   # one late batch per offline visit
    upload = np.where(offline[sess], reconnect[sess], ev["_event_time"] + rng.uniform(0.05, 2.0, n_ev))
    dev_skew = skew[users["device"][ev["_true_user"]]]
    ev["client_time"] = ev["_event_time"] + dev_skew                  # what the phone's clock said
    ev["client_send_time"] = upload + dev_skew
    ev["server_time"] = upload + rng.uniform(0.02, 0.3, n_ev)          # when the server received it
    order = np.lexsort((ev["_event_time"], ev["device_id"]))           # per-device counter, in true order
    seq = np.empty(n_ev, dtype=np.int64)
    dev_sorted = ev["device_id"][order]
    first = np.r_[0, np.flatnonzero(np.diff(dev_sorted)) + 1]
    seq[order] = np.arange(n_ev) - np.repeat(first, np.diff(np.r_[first, n_ev]))
    ev["seq"] = seq
    ev["event_id"] = np.arange(n_ev)

    # Weak networks resend some events: same event_id, later arrival
    dup = np.flatnonzero(rng.random(n_ev) < cfg.duplicate_share)
    events = {key: np.concatenate([val, val[dup]]) for key, val in ev.items()}
    events["server_time"][n_ev:] += rng.uniform(1, 60, len(dup))
    events["client_send_time"][n_ev:] += rng.uniform(1, 60, len(dup))
    order = np.argsort(events["server_time"], kind="stable")          # logs arrive in receipt order
    events = {key: val[order] for key, val in events.items()}
    return users, items, served, events


def corrected_time(events):
    """Action time on the server's clock: receipt time minus the phone's own delay.

    The phone reports when it sent the batch (client_send_time) and when each
    action happened (client_time), both on its own clock. The difference is the
    delay; it is correct even when the phone's clock is wrong.
    """
    return events["server_time"] - (events["client_send_time"] - events["client_time"])


def report(cfg, items, served, events):
    """Measure what the generator planted, using the exploratory slots."""
    k = cfg.slate_size
    pos, ex = served["position"], served["explored"]
    clk = served["_clicked"]
    view = np.zeros(len(pos), dtype=bool)
    v_ids = events["impression_id"][events["event"] == "viewable"]
    view[np.unique(v_ids)] = True                                       # from the log, not the truth
    n_dup = len(events["event_id"]) - len(np.unique(events["event_id"]))
    print(f"Served slots: {len(pos):,}   viewable: {view.mean():.1%}   "
          f"exploratory: {ex.mean():.1%}   events: {len(events['event']):,} "
          f"(of which {n_dup:,} duplicates)")

    keep, d = cfg.scroll_keep, cfg.look_decay
    kk = np.arange(1, k + 1)
    p_view = keep ** (kk - 1)
    p_glimpse = np.where(kk > 1, keep ** np.maximum(kk - 2, 0) * (1 - keep), 0.0)
    planted_served = (p_view + cfg.glimpse_look * p_glimpse) / kk ** d
    planted_view = 1.0 / kk ** d
    ctr_n = np.array([clk[(pos == p) & ~ex].mean() for p in kk])
    ctr_e = np.array([clk[(pos == p) & ex].mean() for p in kk])
    ctr_ev = np.array([clk[(pos == p) & ex & view].mean() for p in kk])
    print("\n          clicks per served slot              |  exploratory, clicks per viewable slot")
    print("Position  normal  explored  relative  planted |  CTR    relative  planted")
    for p in range(k):
        print(f"{p + 1:>8}  {ctr_n[p]:.3f}  {ctr_e[p]:>8.3f}  {ctr_e[p] / ctr_e[0]:>8.2f}"
              f"  {planted_served[p] / planted_served[0]:>7.2f} |  {ctr_ev[p]:.3f}"
              f"  {ctr_ev[p] / ctr_ev[0]:>8.2f}  {planted_view[p]:>7.2f}")
    print(f"\nClicks on slots that never qualified as viewable: "
          f"{(clk & ~view).sum():,} ({(clk & ~view).sum() / clk.sum():.1%} of all clicks)")

    shown = np.bincount(served["item_id"][~ex], minlength=cfg.num_items)
    top1 = np.argsort(-items["popularity"])[: cfg.num_items // 100]
    bottom = np.argsort(items["popularity"])[: cfg.num_items // 2]
    print(f"\nTop 1% most popular items get {shown[top1].sum() / shown.sum():.1%} of normal slots"
          f" and {np.isin(served['item_id'][ex], top1).mean():.1%} of exploratory ones.")
    print(f"The less popular half of the catalogue gets {shown[bottom].sum() / shown.sum():.1%} of normal"
          f" slots and {np.isin(served['item_id'][ex], bottom).mean():.1%} of exploratory ones.")

    bait = items["clickbait"][served["item_id"]] > 0
    video = ~items["is_product"][served["item_id"]]
    first = np.unique(events["event_id"], return_index=True)[1]          # drop duplicates
    e = {key: val[first] for key, val in events.items()}
    w = e["event"] == "watch_end"
    wf = e["value"][w] / items["duration_s"][e["item_id"][w]]
    wb = items["clickbait"][e["item_id"][w]] > 0
    print(f"\nClickbait videos: CTR {clk[bait & video].mean():.3f}, average share watched {wf[wb].mean():.0%}")
    print(f"Other videos:     CTR {clk[~bait & video].mean():.3f}, average share watched {wf[~wb].mean():.0%}")

    late = e["server_time"] - e["_event_time"] > 3_600
    print(f"\nEvents that reached the server more than an hour after the action: {late.mean():.1%}")
    ct = corrected_time(e)
    m_c, m_p = e["event"] == "click", e["event"] == "purchase"
    imp_c = dict(zip(e["impression_id"][m_c], np.flatnonzero(m_c)))
    pairs = np.array([(imp_c[i], j) for i, j in zip(e["impression_id"][m_p], np.flatnonzero(m_p))])
    for label, t in (("server time", e["server_time"]), ("client time", e["client_time"]),
                     ("corrected time", ct)):
        bad = (t[pairs[:, 1]] <= t[pairs[:, 0]]).sum()
        print(f"Purchases not after their click, by {label:<14}: {bad} of {len(pairs)}")
    big = np.abs(e["client_time"] - e["_event_time"]) > 600
    print(f"Events whose phone clock was off by more than 10 minutes: {big.mean():.1%}; "
          f"largest error after correction: {np.abs(ct - e['_event_time']).max():.2f} s")


if __name__ == "__main__":
    cfg = Config()
    users, items, served, events = simulate(cfg)
    report(cfg, items, served, events)
