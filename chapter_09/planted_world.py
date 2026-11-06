"""Listing 9.1 - Tesselo's planted sequence world (section 9.2).

In Chapter 2's simulator every user has one hidden taste vector, fixed for the
whole month, and whether a video satisfies depends only on that vector. The order
of what a user did carries no information, so no sequence model can beat a model
of the whole history there. That is a real result, and the chapter reports it.
To see what sequence models are for, this file plants two effects, and only these:

  * session intent - each visit has one topic, a video category drawn from the
                     user's own taste (P(category) grows with exp(3 x the user's
                     match to the category)). During the visit the user's taste
                     moves towards it: the vector is taste + 1.0 x the category's
                     direction, made unit length again.
  * slow drift     - each user's taste turns at a steady rate towards a hidden
                     direction of their own, 60 degrees over the 30 days.

Everything else is Chapter 2's: the same users, items, visits and requests, the
same old system (which still ranks with each user's day-0 taste, so it knows
neither the intent nor the drift), and the same random draws for looks and
clicks. The planted draws use their own generator (seed cfg.seed + 900). With
both effects set to 0 the logs are exactly Chapter 2's; the script checks this.
The numbers are toy settings chosen to make the effects visible, not
measurements. Hidden truth keeps a leading "_".

set_world(name) makes Chapter 3's build() (and so every later chapter's
pipeline) use one of the two worlds, without changing any earlier file.

Needs Python 3.12 or later. Run from the repository root (under a minute):
    python chapter_09/planted_world.py
"""
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _ch in ("chapter_02", "chapter_03"):
    sys.path.insert(0, str(ROOT / _ch))
import tesselo_data  # noqa: E402
import tesselo_shop  # noqa: E402
from tesselo_data import Config, choose_slates, make_requests, make_world, sigmoid  # noqa: E402,F401
from tesselo_shop import DAY, VIDEO_CATEGORIES, build  # noqa: E402

WORLDS = ("Chapter 2's world", "planted world")


@dataclass
class Planted:
    intent: float = 1.0              # weight of the visit's topic in the user's taste
    focus: float = 3.0               # how strongly the topic follows the user's own taste
    drift_degrees: float = 60.0      # how far each taste turns over the 30 days


def category_directions(cfg):
    """The 8 hidden directions that define the video categories. Chapter 3 draws them
    (seed cfg.seed + 300) after one integer draw; this repeats those two draws."""
    rng = np.random.default_rng(cfg.seed + 300)
    rng.integers(1 << 30)
    d = rng.normal(size=(len(VIDEO_CATEGORIES), cfg.dim))
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def plant(cfg, P, users, rq):
    """Draw each user's drift direction and each visit's intent (own generator)."""
    rng = np.random.default_rng(cfg.seed + 900)
    vec = users["vec"].astype(np.float64)
    w = rng.normal(size=vec.shape)
    w -= (w * vec).sum(axis=1, keepdims=True) * vec            # at right angles to the day-0 taste
    users["_drift"] = (w / np.linalg.norm(w, axis=1, keepdims=True)).astype(np.float32)
    logit = P.focus * vec[rq["sess_user"]] @ category_directions(cfg).T
    p = np.exp(logit - logit.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    u = rng.random(len(p))[:, None]
    return (u > np.cumsum(p, axis=1)).sum(axis=1)               # one category per visit


def taste(users, u, t, intent, P, cfg=Config()):
    """The hidden taste of user u at time t (seconds) during a visit with this intent
    (-1: no intent). Chapter 2's world is P = Planted(0, 0, 0)."""
    theta = np.deg2rad(P.drift_degrees) * np.asarray(t, dtype=np.float64) / (cfg.days * DAY)
    v = (np.cos(theta)[:, None] * users["vec"][u] + np.sin(theta)[:, None] * users["_drift"][u])
    if P.intent > 0:
        has = np.asarray(intent) >= 0
        v = v + has[:, None] * P.intent * category_directions(cfg)[np.maximum(intent, 0)]
    return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)


def simulate_planted(cfg=Config(), P=None):
    """Chapter 2's simulate(), with the planted effects of P (Listing 9.1). The lines
    that differ are marked PLANTED; every other line is Chapter 2's. The planted draws
    use their own generator, so with P = Planted(0, 0) the logs equal Chapter 2's."""
    P = Planted() if P is None else P
    rng = np.random.default_rng(cfg.seed)
    users, items, skew = make_world(cfg, rng)
    rq = make_requests(cfg, rng, users)
    intent = plant(cfg, P, users, rq)                                   # PLANTED
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
    eff = taste(users, u, served["served_time"], intent[served["session_id"]], P, cfg)   # PLANTED
    match = np.einsum("nd,nd->n", eff, items["vec"][i])                # PLANTED
    p_attract = sigmoid(5 * match - 3.5 + items["clickbait"][i])       # clickbait lifts clicks...
    p_satisfy = sigmoid(5 * match - 2.0 - 2 * items["clickbait"][i])  # ...but not satisfaction
    looked = rng.random(len(i)) < p_look
    clicked = looked & (rng.random(len(i)) < p_attract)
    satisfied = clicked & (rng.random(len(i)) < p_satisfy)
    served.update({"_viewable": on_screen, "_glimpsed": glimpse, "_looked": looked,
                   "_clicked": clicked, "_intent": intent[served["session_id"]]})   # PLANTED: kept as truth

    # ---- action times, in true behavioural order ------------------------------------
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
          "_true_user": true_user[imp],
          "consent_at_time": served["consent_at_time"][imp]}

    # The app attaches the identity that is logged in at the moment of each event
    sess_ev = ev["session_id"]
    logged = ev["_event_time"] >= rq["t_login"][sess_ev]
    ev["user_id"] = np.where(logged, rq["login_user"][sess_ev], -1)

    # Login events: a visit that started logged out and someone logs in partway through
    s_login = np.flatnonzero(rq["login_at"] > 0)
    first_req = np.searchsorted(rq["req_session"], s_login)
    login_req = first_req + rq["login_at"][s_login]
    t_login = rq["t_login"][s_login]
    lu = rq["login_user"][s_login]
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


def set_world(name, P=None):
    """Make Chapter 3's build() use Chapter 2's simulator or the planted one."""
    if name == WORLDS[0]:
        tesselo_shop.simulate = tesselo_data.simulate
    else:
        def planted(cfg=Config()):
            return simulate_planted(cfg, P)
        tesselo_shop.simulate = planted


def truth(W, name):
    """A function giving the hidden taste for (users, times, intents) in that world."""
    P = Planted(0.0, 0.0, 0.0) if name == WORLDS[0] else Planted()
    users = W["users"]
    if "_drift" not in users:                                    # Chapter 2's world has no drift field
        users["_drift"] = np.zeros_like(users["vec"])
    return lambda u, t, intent: taste(users, u, t, intent, P, W["cfg"])


if __name__ == "__main__":
    import platform
    print(f"Environment: Python {platform.python_version()}, NumPy {np.__version__}, {platform.system()} "
          f"{platform.machine()}, {os.cpu_count()} CPUs")
    cfg = Config()
    a = tesselo_data.simulate(cfg)
    b = simulate_planted(cfg, Planted(0.0, 0.0, 0.0))
    same = all(np.array_equal(a[3][k], b[3][k]) for k in a[3] if a[3][k].dtype != object) and \
        all((a[3][k] == b[3][k]).all() for k in a[3] if a[3][k].dtype == object)
    print(f"With both effects at 0 the planted simulator reproduces Chapter 2's logs exactly: {same}")
    set_world(WORLDS[1])
    users, items, served, events, orders, _ = build(cfg)
    dirs = category_directions(cfg)
    video = ~items["is_product"]
    agree = (np.argmax(items["vec"][video] @ dirs.T, axis=1) == items["category"][video]).mean()
    print(f"Video categories recovered from the directions: {agree:.1%} (Chapter 3 assigns 15% at random)")
    ses = served["session_id"]
    intent = served["_intent"]
    clicked = served["_clicked"] & video[served["item_id"]]
    cat = items["category"][served["item_id"]]
    print(f"Planted world: {clicked.sum():,} video clicks; share whose category is the visit's intent: "
          f"{(cat[clicked] == intent[clicked]).mean():.1%}; share of shown videos in that category: "
          f"{(cat[video[served['item_id']]] == intent[video[served['item_id']]]).mean():.1%}")
    aserved = a[2]
    aclicked = aserved["_clicked"] & video[aserved["item_id"]]
    acat = items["category"][aserved["item_id"]]
    print(f"Chapter 2's world: {aclicked.sum():,} video clicks; share in the (unused) intent category: "
          f"{(acat[aclicked] == intent[aclicked]).mean():.1%}")
