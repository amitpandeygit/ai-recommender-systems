"""Chapter 12's scale world, replayed for Chapter 14 (sections 14.2 to 14.8).

Chapter 12 saved only the video clicks of consenting users (chapter_12/scale_world.py).
Chapter 14 also needs what a generative recommender is judged by: each request's
hidden taste, so that any row can be scored online as in Chapter 5's simulated test;
the rows that were shown and not clicked, to train a ranker as Chapter 7 does; and
which clicks ended in a satisfied play. This file replays the scale world's seeded
generator, with the same random draws in the same order, and checks that the replay
gives exactly the saved clicks. Two things are added, with their own generators so
that no saved click changes:

  * satisfied plays - each click is satisfied with Chapter 2's chance
                      sigmoid(5 x match - 2 - 2 x clickbait), from the taste at that
                      moment, and a satisfied click is logged as a satisfied play
                      (a completion or a like) with probability 0.50848, an
                      unsatisfied one with 0.000064 (Chapter 5); seed cfg.seed + 1400
  * shown rows      - for a quarter of the training users (seed cfg.seed + 1401),
                      every video they had on screen or clicked in days 9 to 23, with
                      its slot, click and satisfied play: Chapter 7's training rows

As in Chapter 13, the 348 videos held back as new (chapter_13/sid_data.new_videos) do
not exist before day 23: their clicks then are removed, and so are their shown rows and
their counts in the video statistics.

  replay()       - builds the world once and saves it in chapter_14/cache/replay.npz
  load()         - the saved replay (built first if needed), with Chapter 12's world
  requests(R, lo, hi) - the held-out users' requests in [lo, hi) days: user, time,
                   visit, and the hidden taste at the moment of the request
  online(top, ...) - expected satisfied plays of each row, from the hidden taste at
                   each request (Chapter 5's expected_outcomes, Listing 5.3)

Run from the repository root to build and check the replay (about six minutes):
    python chapter_14/gen_data.py
"""
import os
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "2"
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _ch in ("chapter_13", "chapter_12", "chapter_09", "chapter_05", "chapter_03", "chapter_02"):
    sys.path.insert(0, str(ROOT / _ch))
from tesselo_data import Config, choose_slates, make_requests, make_world, sigmoid  # noqa: E402
from tesselo_shop import DAY, add_attributes  # noqa: E402
from planted_world import Planted, plant, taste  # noqa: E402
from scale_world import BLOCK, USERS, VISITS, load as load_scale  # noqa: E402
from evaluate import P_COMPLETE_UNSAT, P_PLAY_SAT, expected_outcomes  # noqa: E402
from sid_data import new_videos  # noqa: E402

FILE = HERE / "cache" / "replay.npz"
CFG = Config(num_users=USERS, sessions_per_user=VISITS)
SAMPLE = 0.25                                       # share of training users whose shown rows are kept
RANK_DAYS = (9, 23)                                 # shown rows kept for the ranker (Chapter 7's weeks)


def held_out(n_users):
    """Chapter 12's held-out users (scale_world.sequences: seed 0, 10%)."""
    return np.random.default_rng(0).random(n_users) < 0.10


def replay(cfg=CFG, P=Planted()):
    rng = np.random.default_rng(cfg.seed)
    users, items, _ = make_world(cfg, rng)
    rq = make_requests(cfg, rng, users)
    intent = plant(cfg, P, users, rq)
    consent = rq["req_time"] < users["consent_off_from"][rq["req_user"]]
    slates = choose_slates(cfg, rng, users, items, rq["req_user"], consent)[0]
    add_attributes(cfg, users, items)
    k = cfg.slate_size
    pos = np.arange(1, k + 1)
    draw = np.random.default_rng(cfg.seed + 1200)                 # Chapter 12's behaviour draws
    sat_draw = np.random.default_rng(cfg.seed + 1400)             # new: satisfied plays
    held = held_out(cfg.num_users)
    sample = (np.random.default_rng(cfg.seed + 1401).random(cfg.num_users) < SAMPLE) & ~held
    video = np.flatnonzero(~items["is_product"])
    vnum = np.full(cfg.num_items, -1)
    vnum[video] = np.arange(len(video))
    new = np.zeros(cfg.num_items, dtype=bool)
    new[video[new_videos({"video": video})]] = True
    stats = np.zeros((3, cfg.days, len(video)))                   # shown, clicked, played per day and video
    clicks, shown = [], []
    for a in range(0, len(slates), BLOCK):                        # Chapter 12's loop, block by block
        b = min(a + BLOCK, len(slates))
        req = np.repeat(np.arange(a, b), k)
        u = rq["req_user"][req]
        i = slates[a:b].ravel()
        t0 = rq["req_time"][req]
        sess = rq["req_session"][req]
        p = np.tile(pos, b - a)
        depth = np.repeat(1 + np.minimum(draw.geometric(1 - cfg.scroll_keep, b - a) - 1, k - 1), k)
        on_screen, glimpse = p <= depth, p == depth + 1
        p_look = np.where(on_screen, 1.0, np.where(glimpse, cfg.glimpse_look, 0.0)) / p ** cfg.look_decay
        eff = taste(users, u, t0, intent[sess], P, cfg)
        match = np.einsum("nd,nd->n", eff, items["vec"][i])
        p_attract = sigmoid(5 * match - 3.5 + items["clickbait"][i])
        clicked = (draw.random(len(i)) < p_look) & (draw.random(len(i)) < p_attract) & ~items["is_product"][i]
        t = t0 + (p - 1) * 2.5 + np.where(glimpse, draw.uniform(0.2, 0.9, len(i)), draw.uniform(1, 6, len(i)))
        keep = users["current_consent"][u]
        c = np.flatnonzero(clicked & keep)                        # Chapter 12's clicks
        p_sat = sigmoid(5 * match[c] - 2.0 - 2 * items["clickbait"][i[c]])
        sat = sat_draw.random(len(c)) < p_sat
        play = sat_draw.random(len(c)) < np.where(sat, P_PLAY_SAT, P_COMPLETE_UNSAT)
        clicks.append((u[c], i[c], t[c], t0[c], sess[c], req[c], play))
        played = np.zeros(len(i), dtype=bool)
        played[c] = play
        exists = ~items["is_product"][i] & ~(new[i] & (t0 < 23 * DAY))     # a video, uploaded by then
        row = (on_screen | clicked) & keep & exists
        day = np.minimum((t0 // DAY).astype(int), cfg.days - 1)
        tr = row & ~held[u]
        for j, flag in enumerate((np.ones(len(i), dtype=bool), clicked, played)):
            np.add.at(stats[j], (day[tr & flag], vnum[i[tr & flag]]), 1.0)
        s = np.flatnonzero(row & sample[u] & (t0 >= RANK_DAYS[0] * DAY) & (t0 < RANK_DAYS[1] * DAY))
        shown.append((u[s], vnum[i[s]], p[s], t0[s], clicked[s], played[s]))
    cu, ci, ct, ctr_, cs, cr, cp = (np.concatenate([q[j] for q in clicks]) for j in range(7))
    o = np.lexsort((ct, cu))
    su, si, sp, st, sc, spl = (np.concatenate([q[j] for q in shown]) for j in range(6))
    rq_keep = held[rq["req_user"]] & users["current_consent"][rq["req_user"]]
    return {"user": cu[o], "item": ci[o], "time": ct[o], "request_time": ctr_[o], "session": cs[o],
            "request": cr[o], "play": cp[o],
            "shown_user": su, "shown_video": si, "shown_slot": sp, "shown_time": st, "shown_click": sc,
            "shown_play": spl, "stats": stats.astype(np.float32),
            "req_index": np.flatnonzero(rq_keep), "req_user": rq["req_user"][rq_keep],
            "req_time": rq["req_time"][rq_keep], "req_session": rq["req_session"][rq_keep],
            "intent": intent.astype(np.int8), "users_vec": users["vec"], "users_drift": users["_drift"],
            "users_language": users["language"], "consent": users["current_consent"], "held": held,
            "sample": sample, "items_vec": items["vec"], "clickbait": items["clickbait"]}


def check(R, W):
    """The replay's clicks are Chapter 12's saved clicks, exactly."""
    for name in ("user", "item", "time", "request_time", "session"):
        assert np.array_equal(R[name], W[name]), name


def load():
    """The replay (built and checked first if needed) and Chapter 12's world."""
    W = load_scale()
    if not FILE.exists():
        R = replay()
        check(R, W)
        FILE.parent.mkdir(exist_ok=True)
        np.savez_compressed(FILE, **R)
    z = np.load(FILE)
    return {k: z[k] for k in z.files}, W


def requests(R, lo, hi, P=Planted(), cfg=CFG):
    """The held-out, consenting users' requests with time in [lo, hi) days, and the
    hidden taste at the moment of each (Chapter 9's planted taste: visit intent and
    drift)."""
    m = (R["req_time"] >= lo * DAY) & (R["req_time"] < hi * DAY)
    users = {"vec": R["users_vec"], "_drift": R["users_drift"]}
    u, t, s = R["req_user"][m], R["req_time"][m], R["req_session"][m]
    return {"user": u, "time": t, "session": s, "request": R["req_index"][m],
            "taste": taste(users, u, t, R["intent"][s].astype(np.int64), P, cfg)}


def online(top_video, Q, R, video):
    """Expected clicks and satisfied plays of each row (videos numbered 0 to V - 1, -1
    for an empty slot), from the hidden taste at each request: Chapter 5's
    expected_outcomes (Listing 5.3), unchanged."""
    top = np.where(top_video >= 0, video[np.maximum(top_video, 0)], -1)
    items = {"vec": R["items_vec"], "clickbait": R["clickbait"]}
    return expected_outcomes(top, np.arange(len(top)), {"vec": Q["taste"]}, items, CFG)


if __name__ == "__main__":
    t0 = time.time()
    if FILE.exists():
        FILE.unlink()
    R, W = load()
    print(f"Replay of the scale world: {len(R['user']):,} clicks, the same as Chapter 12's saved clicks; "
          f"satisfied plays {R['play'].sum():,} ({R['play'].mean():.1%} of clicks)  [{time.time() - t0:.0f} s]")
    print(f"  shown rows kept for the ranker (a quarter of the training users, days 9 to 23): "
          f"{len(R['shown_user']):,}; clicked {R['shown_click'].mean():.1%}; satisfied plays per click "
          f"{R['shown_play'].sum() / R['shown_click'].sum():.1%}")
    for lo, hi, name in ((16, 23, "validation"), (23, 30, "test")):
        Q = requests(R, lo, hi)
        print(f"  held-out users' requests, {name} (days {lo} to {hi}): {len(Q['user']):,} "
              f"from {len(np.unique(Q['user'])):,} users")
