"""Listing 7.1 - Training rows and features for Tesselo's ranking stages (Chapter 7).

The ranking stages learn from impressions: every video a consenting, logged-in
user had on screen (Chapter 2's "viewable" rule), plus any slot the user clicked
even if it never met that rule. Each row has the outcomes the app logged for
that impression: a click, a completion, a like, the share of the video watched,
and a satisfied play (a completion or a like, the goal of Chapter 5).

Features are frozen in snapshots. A snapshot at day t holds only what was known
at t: the implicit scores come from Chapter 2's prepare() (what had arrived,
de-duplicated, with consent then and now), and the video statistics from the
impression rows of the 30 days before t with the labels that had arrived by t.
Rows from the week after t get the features of the snapshot at t, and their
labels are those that had arrived by the end of that week. So no row sees the
future, which is the job a point-in-time join does in a feature store
(Chapter 28). An outcome still on its way at the cutoff counts as "did not
happen"; heavy_ranking.py measures how often that occurs.

Features of one (user, video) pair:
  dense  - iALS score (a cross-feature learned by Chapter 4's model), the video's
           popularity, click rate and satisfied plays per click, whether the
           video's language is the user's app language (a cross-feature), the
           share of the user's history in the video's category (a counted
           cross-feature), the size of the user's history, and whether they have one
  sparse - the video's id, creator, category and language, the user's app
           language, and the user's history as a list of video ids with weights

There is no user-id feature: a user is their history (as in Chapter 6). When a
user turns personalisation off, their history leaves the next snapshot and their
rows leave the next training set. What their earlier rows already taught the
shared weights stays until the model is retrained without them (section 7.1).

Needs Python 3.12 or later. Imported by the other Chapter 7 scripts.
"""
import os
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "2"              # NumPy's linear algebra: the same thread count on every run
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for folder in ("chapter_02", "chapter_03", "chapter_04", "chapter_05", "chapter_06"):
    sys.path.insert(0, str(ROOT / folder))
from tesselo_shop import DAY, LANGUAGES, Config, build  # noqa: E402,F401
from similarity_recs import MIN_BUYERS, NO, group_popularity, top_k_chain, user_item_scores  # noqa: E402,F401
from implicit_mf import eligible_and_scored, ials  # noqa: E402

HISTORY = 50                 # the user's 50 highest-weighted videos represent them
PRIOR = 20                   # smoothing: rates start from the average, worth 20 views or clicks
DENSE = ["iALS score", "popularity (log)", "click rate (log)", "plays per click (log)",
         "same language", "share of history in category", "history size (log)", "has history"]


def snapshot(events, served, users, items, cutoff, window_days=30):
    """Everything the ranking stages may use at `cutoff`: implicit scores, iALS
    vectors, video statistics and each user's history, all from data known then.
    The video statistics count the same kind of row the ranker trains on (an
    impression on screen or clicked, in the last 30 days, with the labels that had
    arrived): views are those rows, clicks and satisfied plays are counted once per
    row, so a video both completed and liked is one play."""
    n_users, n_items = len(users["vec"]), len(items["vec"])
    video = ~items["is_product"]
    weight, used = user_item_scores(events, users, cutoff, n_users, n_items)
    X, Y = ials(weight, 16, 1.0, 10.0)                          # Chapter 4's settings
    r = impressions(events, served, users, items, cutoff - window_days * DAY, cutoff, cutoff)
    views = np.bincount(r["item"], minlength=n_items).astype(float)
    clicks = np.bincount(r["item"], weights=r["click"].astype(float), minlength=n_items)
    plays = np.bincount(r["item"], weights=r["play"].astype(float), minlength=n_items)
    ctr = (clicks + PRIOR * clicks.sum() / views.sum()) / (views + PRIOR)
    ppc = (plays + PRIOR * plays.sum() / clicks.sum()) / (clicks + PRIOR)
    engaged = (weight > 0).sum(axis=0)
    n_cat = int(items["category"].max()) + 1
    cat_share = np.stack([weight[:, items["category"] == c].sum(axis=1) for c in range(n_cat)], axis=1)
    cat_share /= np.maximum(cat_share.sum(axis=1, keepdims=True), 1e-12)
    hist_id = np.argsort(-weight, axis=1, kind="stable")[:, :HISTORY]
    hist_w = np.take_along_axis(weight, hist_id, axis=1)        # 0 where the history is shorter
    lang_pop = group_popularity(weight, users["language"], len(LANGUAGES))
    return {"cutoff": cutoff, "weight": weight, "used": used, "X": X, "Y": Y,
            "ok": video & (engaged >= MIN_BUYERS), "engaged": engaged,
            "ctr": ctr, "ppc": ppc, "cat_share": cat_share,
            "n_hist": (weight > 0).sum(axis=1), "hist_id": hist_id, "hist_w": hist_w.astype(np.float32),
            "lang_pop": lang_pop}


def impressions(events, served, users, items, start, end, label_cutoff):
    """Training rows: impressions served in [start, end) to consenting, logged-in
    users, videos only, that were on screen or clicked. Labels are the events with
    that impression id that had arrived by label_cutoff, each counted once."""
    video = ~items["is_product"]
    s_user = served["user_id"]
    keep = ((served["served_time"] >= start) & (served["served_time"] < end) & (s_user >= 0)
            & served["consent_at_time"] & users["current_consent"][np.maximum(s_user, 0)]
            & video[served["item_id"]])
    arrived = (events["server_time"] < label_cutoff) & (events["impression_id"] >= 0)
    _, first = np.unique(events["event_id"][arrived], return_index=True)       # one copy per event
    imp = events["impression_id"][arrived][first]
    name = events["event"][arrived][first]
    value = events["value"][arrived][first]
    flag = lambda e: np.isin(np.arange(len(s_user)), imp[name == e])          # noqa: E731
    seen, click = flag("viewable"), flag("click")
    complete, like = flag("complete"), flag("like")
    watch = np.zeros(len(s_user))
    w = name == "watch_end"
    watch[imp[w]] = value[w] / items["duration_s"][served["item_id"][imp[w]]]
    rows = np.flatnonzero(keep & (seen | click))
    return {"user": s_user[rows], "item": served["item_id"][rows], "position": served["position"][rows],
            "click": click[rows], "complete": complete[rows], "like": like[rows],
            "play": (complete | like)[rows], "watch": np.clip(watch[rows], 0.0, 1.0),
            "explored": served["explored"][rows], "time": served["served_time"][rows]}


def pair_features(S, users, items, u, i):
    """Dense features (columns in the order of DENSE) for pairs (u[k], i[k])."""
    has = S["n_hist"][u] > 0
    return np.stack([
        np.where(has, np.einsum("nk,nk->n", S["X"][u], S["Y"][i]), 0.0),     # iALS score
        np.log1p(S["engaged"][i]),                                           # popularity
        np.log(S["ctr"][i]),                                                 # click rate
        np.log(S["ppc"][i]),                                                 # satisfied plays per click
        (users["language"][u] == items["language"][i]).astype(float),        # same language
        S["cat_share"][u, items["category"][i]],                             # share of history in category
        np.log1p(S["n_hist"][u]),                                            # history size
        has.astype(float)], axis=1).astype(np.float32)


def candidate_pools(S, users, items, rows, size):
    """This chapter's pool, adapted from Chapter 6's choice: iALS's top videos,
    filled by popular in your language (Chapter 3's chain) where Chapter 6's merge
    used "New for you"; `size` candidates per user. -1 marks an empty slot."""
    ials_score = eligible_and_scored(S["X"][rows] @ S["Y"].T, S["weight"][rows], S["ok"])
    lang = S["lang_pop"][users["language"][rows]]
    return top_k_chain([ials_score, lang], S["used"][rows], ~items["is_product"], size)
