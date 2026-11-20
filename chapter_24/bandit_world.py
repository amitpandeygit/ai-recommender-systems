"""Shared pieces of Chapter 24: Chapter 2's world as a place where a bandit can act.

A bandit learns from the outcomes of its own choices, so it cannot be tested on a
fixed log alone: it must choose, see what happens, and choose again. Chapter 2's
generator already holds the rules that decide what a user does with a video in a
slot. This file lets a bandit use those rules, unchanged, for videos it picks:

  p_look(cfg)            - the chance that a user looks at each slot of a row of 10
                           (Chapter 5's formula: scrolling, glimpses, attention)
  expected_play(...)     - the exact chance of a satisfied play (a completion or a
                           like) for a video in a slot, from the hidden taste
  Draws                  - uniform random numbers drawn once per request and slot, so
                           that every policy meets the same users with the same luck
                           (common random numbers); outcome() turns them into events
                           exactly as Chapter 2's generator does
  visit_delays(...)      - when the outcome of a request reaches the server: two
                           minutes after the request, or, for a visit uploaded late by
                           a phone that was offline, when that phone reconnected in
                           Chapter 2's logs
  ranked_slates(cfg)     - the slates Chapter 2's old system ranked before it put
                           exploratory items into some slots (a replay of its seeded
                           random draws, checked against the logs)

Nothing here changes Chapter 2's logs: every new draw uses its own generator.
Fields and arguments whose names start with "_" hold the simulation's hidden truth;
policies never read them.

Needs Python 3.12 or later. Imported by the other Chapter 24 scripts.
"""
import os
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "2"              # NumPy's linear algebra: the same thread count on every run
import dataclasses
import sys
from pathlib import Path

import numpy as np
from scipy.stats import beta as beta_dist

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _ch in ("chapter_02", "chapter_03", "chapter_05"):
    sys.path.insert(0, str(ROOT / _ch))
from tesselo_data import Config, choose_slates, make_requests, make_world, sigmoid  # noqa: E402

DAY = 86_400
ROW = 10
EVENT_WEIGHTS = {"click": 1, "complete": 3, "like": 4, "skip": -2, "not_interested": -5}   # Chapter 2, Table 2.4
# What the app can log (Chapter 5, section 5.7): a satisfied user completes the video
# (share watched from Beta(5, 2) at 0.8 or more) or likes it (probability 0.25).
P_COMPLETE_SAT = 1 - (6 * 0.8 ** 5 - 5 * 0.8 ** 6)       # 0.34464
P_COMPLETE_UNSAT = 0.2 ** 6                               # 0.000064
P_LIKE_SAT = 0.25
P_PLAY_SAT = 1 - (1 - P_COMPLETE_SAT) * (1 - P_LIKE_SAT)  # 0.50848


def p_look(cfg=Config(), k=ROW):
    """Chance of looking at slots 1..k: on screen keep^(r-1), glimpsed keep^(r-2)(1-keep)
    and looked at glimpse_look as often, all divided by r^look_decay (Chapter 5)."""
    r = np.arange(1, k + 1)
    on = cfg.scroll_keep ** (r - 1)
    glimpse = np.where(r >= 2, cfg.scroll_keep ** np.maximum(r - 2, 0) * (1 - cfg.scroll_keep), 0.0)
    return (on + cfg.glimpse_look * glimpse) / r ** cfg.look_decay


def play_given_look(match, bait):
    """P(satisfied play | the user looked): click, then a completion or a like."""
    attract = sigmoid(5 * match - 3.5 + bait)
    satisfy = sigmoid(5 * match - 2.0 - 2 * bait)
    return attract * (satisfy * P_PLAY_SAT + (1 - satisfy) * P_COMPLETE_UNSAT)


def expected_play(_user_vec, items, item_ids, slot, cfg=Config()):
    """Exact chance of a satisfied play for each (user, video) pair at a slot (1-10).
    _user_vec: hidden taste vectors, one row per pair (or broadcastable); products give 0."""
    i = np.asarray(item_ids)
    match = np.einsum("...d,...d->...", _user_vec.astype(np.float64), items["vec"][i].astype(np.float64))
    p = p_look(cfg)[np.asarray(slot) - 1] * play_given_look(match, items["clickbait"][i])
    return np.where(items["is_product"][i], 0.0, p)


class Draws:
    """Uniform numbers for n requests and k slots, drawn once. Every policy uses the
    same numbers, so two policies that show the same video in the same slot of the same
    request get the same outcome, and differences come from the choices alone."""
    def __init__(self, n, k=ROW, seed=0):
        rng = np.random.default_rng(seed)
        self.depth_u = rng.random(n)
        self.u = rng.random((n, k, 6)).astype(np.float64)     # look, click, satisfy, watch, like, not interested

    def views(self, rows, cfg=Config(), k=ROW):
        """Slots on screen (Chapter 2's viewable rule) and slots looked at, as Chapter 2's
        generator draws them: the user scrolls on with probability scroll_keep."""
        m = np.floor(np.log(1.0 - self.depth_u[rows]) / np.log(cfg.scroll_keep))   # slots scrolled past
        depth = 1 + np.minimum(m, k - 1)
        pos = np.arange(1, k + 1)[None, :]
        on = pos <= depth[:, None]
        glimpse = pos == depth[:, None] + 1
        p = np.where(on, 1.0, np.where(glimpse, cfg.glimpse_look, 0.0)) / pos ** cfg.look_decay
        return on, self.u[rows, :, 0] < p

    def outcome(self, rows, slots, _user_vec, items, item_ids, cfg=Config()):
        """Events for the video item_ids[j] shown at slots[j] of request rows[j]:
        a dict of boolean arrays (viewable, click, complete, like, skip,
        not_interested, play) and the share watched, from Chapter 2's rules."""
        rows, slots, i = np.asarray(rows), np.asarray(slots), np.asarray(item_ids)
        on, looked = self.views(rows, cfg)
        s = slots - 1
        on, looked = on[np.arange(len(rows)), s], looked[np.arange(len(rows)), s]
        u = self.u[rows, s]
        match = np.einsum("nd,nd->n", _user_vec.astype(np.float64), items["vec"][i].astype(np.float64))
        bait = items["clickbait"][i]
        click = looked & (u[:, 1] < sigmoid(5 * match - 3.5 + bait))
        satisfied = click & (u[:, 2] < sigmoid(5 * match - 2.0 - 2 * bait))
        watch = np.where(satisfied, beta_dist.ppf(u[:, 3], 5, 2), beta_dist.ppf(u[:, 3], 1, 6))
        video = ~items["is_product"][i]
        complete = click & video & (watch >= 0.8)
        like = satisfied & video & (u[:, 4] < P_LIKE_SAT)
        skip = click & video & (watch < 0.1)
        not_int = click & ~satisfied & (u[:, 5] < 0.05)
        return {"viewable": on, "click": click, "complete": complete, "like": like, "skip": skip,
                "not_interested": not_int, "play": complete | like, "watch": np.where(click, watch, 0.0)}


def implicit_weight(ev):
    """Chapter 2's event weights summed for each impression (no decay): positive means
    the user engaged with the video."""
    return sum(EVENT_WEIGHTS[k] * ev[k].astype(float) for k in EVENT_WEIGHTS)


def visit_delays(served, events):
    """For each session of Chapter 2's logs: the time its phone uploaded the visit if it
    was offline (it arrived as one late batch), else nan. An event that waited on the phone
    for more than two minutes marks an offline visit."""
    n_sess = served["session_id"].max() + 1
    late = (events["server_time"] - events["_event_time"]) > 120.0
    reconnect = np.full(n_sess, np.inf)
    np.minimum.at(reconnect, events["session_id"][late], events["server_time"][late])
    return np.where(np.isfinite(reconnect), reconnect, np.nan)


def arrival_times(t_request, session, reconnect, wait=120.0):
    """When a request's outcome reaches the server: `wait` seconds after the request
    (time to look, click and watch), or when an offline phone reconnected, if later."""
    t = t_request + wait
    rc = reconnect[session]
    return np.where(np.isnan(rc), t, np.maximum(t, np.nan_to_num(rc, nan=0.0)))


def ranked_slates(cfg=Config()):
    """The slates Chapter 2's old system ranked, before exploratory items replaced some of
    them: a replay of simulate()'s seeded draws up to that point, with no exploration."""
    rng = np.random.default_rng(cfg.seed)
    users, items, _ = make_world(cfg, rng)
    rq = make_requests(cfg, rng, users)
    consent = rq["req_time"] < users["consent_off_from"][rq["req_user"]]
    slates, _, _ = choose_slates(dataclasses.replace(cfg, explore_rate=0.0), rng, users, items,
                                 rq["req_user"], consent)
    return slates
