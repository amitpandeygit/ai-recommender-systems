"""A larger planted Tesselo world for Chapter 12's scaling runs (sections 12.4 to 12.6).

Chapter 9's planted world, in which each visit has an intent and every taste drifts
slowly (chapter_09/planted_world.py), built with Chapter 2's generator for 8 times as
many users, 64,000 instead of 8,000, each three times as active (30 visits in the 30
days on average instead of 10), so that histories are long enough to study history
length. The 5,000 items' settings and the 30 days are Chapter 2's. Chapter 3's rules give every item a category and a
language. The old system that chose what users saw is Chapter 2's, unchanged.

Only what the scaling runs need is kept: every video click of a consenting user, with
its user, its moment, the moment of the request that showed it and its visit, in true
order. As in Chapter 9's click log, a user's clicks are kept only if the user consents
to personalisation now (and so has consented all month): the clicks of a user who
turned it off are not used for training or testing. The identity gaps, upload delays
and duplicates of Chapter 2's logs are left out. Chapters 9 to 11 test them; here they would only add
noise to a study of model size, data and history length. The behaviour draws use their
own generator (seed cfg.seed + 1200) and are made in blocks of requests, to keep memory
small. The result is saved in scale_world.npz next to this file, so it is built once.

  build()   - the world: clicks of consenting users (user, item, time, request time,
              session, sorted by user and time),
              items (creator, category, language, is_product) and the number of users
  load()    - the saved world, built first if needed
  sequences(W) - the data of the scaling runs: videos renumbered 0 to V - 1; 10% of
              users, drawn at random (seed 0), held out; the other users' clicks before
              day 23 for training; the held-out users' whole sequences, to score their
              clicks before day 23 (validation) and in days 23 to 30 (test). Each part
              is (videos, click times, request times), one array of each per user

Needs Python 3.12 or later. Run from the repository root (about 4 minutes):
    python chapter_12/scale_world.py
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
for _ch in ("chapter_02", "chapter_03", "chapter_09"):
    sys.path.insert(0, str(ROOT / _ch))
from tesselo_data import Config, choose_slates, make_requests, make_world, sigmoid  # noqa: E402
from tesselo_shop import DAY, add_attributes  # noqa: E402
from planted_world import Planted, plant, taste  # noqa: E402

FILE = HERE / "scale_world.npz"
USERS = 64_000                                   # 8 times Chapter 2's 8,000
VISITS = 30.0                                    # 3 times Chapter 2's 10 visits per user
BLOCK = 100_000                                  # requests per block of behaviour draws


def build(cfg=Config(num_users=USERS, sessions_per_user=VISITS), P=Planted()):
    rng = np.random.default_rng(cfg.seed)
    users, items, _ = make_world(cfg, rng)
    rq = make_requests(cfg, rng, users)
    intent = plant(cfg, P, users, rq)
    consent = rq["req_time"] < users["consent_off_from"][rq["req_user"]]
    slates = choose_slates(cfg, rng, users, items, rq["req_user"], consent)[0]
    k = cfg.slate_size
    pos = np.arange(1, k + 1)
    draw = np.random.default_rng(cfg.seed + 1200)
    parts = []
    for a in range(0, len(slates), BLOCK):                       # Chapter 2's behaviour, block by block
        b = min(a + BLOCK, len(slates))
        u = np.repeat(rq["req_user"][a:b], k)
        i = slates[a:b].ravel()
        t0 = np.repeat(rq["req_time"][a:b], k)
        sess = np.repeat(rq["req_session"][a:b], k)
        p = np.tile(pos, b - a)
        depth = np.repeat(1 + np.minimum(draw.geometric(1 - cfg.scroll_keep, b - a) - 1, k - 1), k)
        on_screen, glimpse = p <= depth, p == depth + 1
        p_look = np.where(on_screen, 1.0, np.where(glimpse, cfg.glimpse_look, 0.0)) / p ** cfg.look_decay
        match = np.einsum("nd,nd->n", taste(users, u, t0, intent[sess], P, cfg), items["vec"][i])
        p_attract = sigmoid(5 * match - 3.5 + items["clickbait"][i])
        clicked = (draw.random(len(i)) < p_look) & (draw.random(len(i)) < p_attract) & ~items["is_product"][i]
        t = t0 + (p - 1) * 2.5 + np.where(glimpse, draw.uniform(0.2, 0.9, len(i)), draw.uniform(1, 6, len(i)))
        c = np.flatnonzero(clicked & users["current_consent"][u])          # consenting users only
        parts.append((u[c], i[c], t[c], t0[c], sess[c]))
    user, item, t, t_req, sess = (np.concatenate([q[j] for q in parts]) for j in range(5))
    o = np.lexsort((t, user))
    add_attributes(cfg, users, items)
    return {"user": user[o], "item": item[o], "time": t[o], "request_time": t_req[o], "session": sess[o],
            "n_users": cfg.num_users, "consenting": int(users["current_consent"].sum()),
            "creator": items["creator"], "category": items["category"], "language": items["language"],
            "is_product": items["is_product"]}


def load():
    if not FILE.exists():
        np.savez_compressed(FILE, **build())
    z = np.load(FILE)
    return {k: (int(z[k]) if k in ("n_users", "consenting") else z[k]) for k in z.files}


def sequences(W, held_share=0.10):
    video = np.flatnonzero(~W["is_product"])
    renum = np.full(len(W["is_product"]), -1)
    renum[video] = np.arange(len(video))
    item = renum[W["item"]]
    starts = np.searchsorted(W["user"], np.arange(W["n_users"] + 1))
    held = np.random.default_rng(0).random(W["n_users"]) < held_share
    train, other = ([], [], []), ([], [], [])
    for u in range(W["n_users"]):
        s, t, r = (a[starts[u]:starts[u + 1]] for a in (item, W["time"], W["request_time"]))
        if not len(s):
            continue                                  # no clicks: a user who does not consent, or never clicked
        if held[u]:
            for part, a in zip(other, (s, t, r)):
                part.append(a)
        else:
            for part, a in zip(train, (s, t, r)):
                part.append(a[t < 23 * DAY])
    return {"train": train, "held": other, "n_videos": len(video), "held_users": len(other[0])}


if __name__ == "__main__":
    t0 = time.time()
    if FILE.exists():
        FILE.unlink()
    W = load()
    per_user = np.bincount(W["user"], minlength=W["n_users"])[np.unique(W["user"])]
    q = np.percentile(per_user, [10, 50, 90, 99])
    print(f"Scale world: {W['n_users']:,} users, {W['consenting']:,} of them consenting to personalisation; "
          f"{len(W['is_product']):,} items ({(~W['is_product']).sum():,} videos); {len(W['item']):,} video clicks of "
          f"consenting users in 30 days  [{time.time() - t0:.0f} s]")
    print(f"  clicks before day 23: {(W['time'] < 23 * DAY).sum():,}; in days 23 to 30: {(W['time'] >= 23 * DAY).sum():,}")
    print(f"  clicks per consenting user with a click: {len(per_user):,} users; mean {per_user.mean():.1f}; "
          f"10th, 50th, 90th and 99th percentiles "
          f"{q[0]:.0f}, {q[1]:.0f}, {q[2]:.0f}, {q[3]:.0f}")
    D = sequences(W)
    print("  Clicks after each user's first, and the clicks before their request (section 12.5):")
    for label, (S, T, R), lo, hi in (("training users, before day 23", D["train"], -np.inf, 23 * DAY),
                                     ("held-out users, before day 23", D["held"], -np.inf, 23 * DAY),
                                     ("held-out users, days 23 to 30", D["held"], 23 * DAY, np.inf)):
        n = late = same_row = none = 0
        for s, t, r in zip(S, T, R):
            ok = (t[1:] >= lo) & (t[1:] < hi)
            after = (t[:-1] >= r[1:]) & ok                 # the click before was made after this one's request
            n += ok.sum()
            late += after.sum()
            same_row += (after & (r[:-1] == r[1:])).sum()
            none += ((t[0] >= r[1:]) & ok).sum()           # no click at all before the request
        print(f"    {label}: {n:,}; the click before came after the request for {late:,} ({late / n:.1%}), "
              f"{same_row:,} of them from the same row; no click before the request: {none:,}")
