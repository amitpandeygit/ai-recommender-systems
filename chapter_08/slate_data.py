"""Shared setup for Chapter 8: from the heavy ranker's scores to the row a user sees.

  world()          - Chapter 2's logs with Chapter 7's snapshots and training rows
  split()          - one week's setup: Chapter 7's shipped pipeline (arm A) trained on
                     the weeks before, and the 50 candidates it keeps for every user
  slates()         - the logged slates inside a week's training rows
  rules()          - the data the filters of section 8.2 need (synthetic, see below)
  slate_outcomes() - Chapter 5's simulated online test, plus section 8.6's planted
                     boredom test
  row_metrics()    - the offline metrics, the guardrails and the variety of a row

Arm A is Chapter 7's shipped pipeline: Chapter 6's pool of 500 (iALS, then popular
in your language), a pre-ranker distilled from the heavy ranker that keeps 50, and
the heavy ranker, the MLP with id vectors trained for one epoch. split("test")
rebuilds exactly the models of Chapter 7's test week, so its online result matches
section 7.6 (0.1773 expected satisfied plays per list).

The rule data are not in Chapter 2's logs: which creators each user has blocked,
which videos are rated 18+ and which users have a teen account, which regions may
show each video, which videos moderation has removed, and which products are out of
stock. rules() makes them with its own random generator (seed cfg.seed + 800), so
nothing that Chapters 2 to 7 use changes. Every share is a toy setting.

Needs Python 3.12 or later and PyTorch. Imported by the other Chapter 8 scripts.
"""
import os
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "2"              # NumPy's linear algebra: the same thread count on every run
import sys
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(2)
ROOT = Path(__file__).resolve().parent.parent
for folder in ("chapter_02", "chapter_03", "chapter_04", "chapter_05", "chapter_06", "chapter_07"):
    sys.path.insert(0, str(ROOT / folder))
from ranking_data import DAY, Config, build, candidate_pools, snapshot  # noqa: E402
from rankers import Ranker, meta_for, predict, rerank, score_pools, tensors, train  # noqa: E402
from heavy_ranking import POOL, SHIP, rows_for  # noqa: E402
from pre_ranking import distil, student_scores  # noqa: E402
from similarity_recs import test_positives  # noqa: E402
from evaluate import P_COMPLETE_UNSAT, P_PLAY_SAT, expected_outcomes  # noqa: E402
from tesselo_data import sigmoid  # noqa: E402
from metrics import diversity, ndcg_at_k  # noqa: E402
from two_tower import guardrails  # noqa: E402

KEPT = 50                     # Chapter 7: the distilled pre-ranker keeps 50
ROW = 10                      # the user sees a row of 10
BOREDOM = 0.75                # section 8.6's planted test (a toy setting, not a measurement)


def world(cfg=Config()):
    """Chapter 2's world and logs, Chapter 7's weekly snapshots and training rows."""
    users, items, served, events, orders, truth = build(cfg)
    W = {"cfg": cfg, "users": users, "items": items, "served": served, "events": events,
         "video": ~items["is_product"], "consenting": np.flatnonzero(users["current_consent"])}
    W["snaps"] = {d: snapshot(events, served, users, items, d * DAY) for d in (9, 16, 23)}
    W["weeks"] = {"validation": rows_for(events, served, users, items, [(9, 16)]),
                  "validation, held out": rows_for(events, served, users, items, [(16, 23)]),
                  "test": rows_for(events, served, users, items, [(9, 16), (16, 23)]),
                  "test, held out": rows_for(events, served, users, items, [(23, 30)])}
    return W


def split(W, name, heavy=None):
    """One week's setup. 'validation': models trained on days 9 to 16 and tested on
    the pools of day 16; 'test': trained on days 9 to 23 and tested on day 23
    (Chapter 7's two weeks, with the same seeds)."""
    cfg, users, items = W["cfg"], W["users"], W["items"]
    if name == "validation":
        snaps, S_distil, seed, day = [W["snaps"][9]], W["snaps"][9], cfg.seed + 700, 16
    else:
        snaps, S_distil, seed, day = [W["snaps"][9], W["snaps"][16]], W["snaps"][16], cfg.seed + 701, 23
    R = W["weeks"][name]
    S = W["snaps"][day]
    b = tensors(snaps, users, items, R["user"], R["item"], R["snap"])
    meta = meta_for(users, items, b["dense"].numpy())
    if heavy is None:
        heavy = train(Ranker(meta, SHIP["arch"]), b, R, epochs=SHIP["epochs"])
    d_rows = np.sort(np.random.default_rng(seed).choice(W["consenting"], 3_000, replace=False))
    d_pools = candidate_pools(S_distil, users, items, d_rows, POOL)
    teacher = score_pools(heavy, S_distil, users, items, d_rows, d_pools)
    pre = distil(Ranker(meta, "pre-ranker", position=False), teacher, S_distil, users, items, d_rows, d_pools)
    pos = test_positives(W["events"], users, day * DAY, (day + 7) * DAY, S["used"], W["video"])
    rows = np.array(sorted(pos))
    out = {"name": name, "day": day, "S": S, "snaps": snaps, "R": R, "b": b, "meta": meta,
           "heavy": heavy, "pre": pre, "pos": pos, "rows": rows, "aud": W["consenting"]}
    for who in ("rows", "aud"):
        r = out[who]
        pools = candidate_pools(S, users, items, r, POOL)
        sc = np.where(pools >= 0, student_scores(pre, S, users, items, r, pools), -np.inf)
        keep = np.argsort(-sc, axis=1, kind="stable")[:, :KEPT]
        kept = np.where(np.take_along_axis(sc, keep, axis=1) > -np.inf, np.take_along_axis(pools, keep, axis=1), -1)
        out[f"kept_{who}"] = kept
        out[f"heavy_{who}"] = score_pools(heavy, S, users, items, r, kept)
    return out


def slates(R):
    """The logged slates in a week's training rows: rows with the same snapshot,
    user and serve time came from one request. Returns a (slates x 10) array of row
    indices, -1 where a slate has fewer rows, sorted by slot."""
    key = np.stack([R["snap"], R["user"], R["time"].astype(np.int64)], axis=1)
    _, group = np.unique(key, axis=0, return_inverse=True)
    order = np.lexsort((R["position"], group))
    g = group[order]
    start = np.r_[0, np.flatnonzero(np.diff(g)) + 1]
    out = np.full((len(start), ROW), -1)
    for s, (a, e) in enumerate(zip(start, np.r_[start[1:], len(g)])):
        out[s, :e - a] = order[a:e]
    return out


def order_by(kept, scores, k=ROW):
    """Each user's top k of the kept candidates by score (ties keep the kept order)."""
    return rerank(kept, np.where(kept >= 0, scores, -np.inf), k)


def slate_outcomes(top, rows, users, items, cfg, boredom=1.0):
    """Expected clicks and satisfied plays per slot, from Chapter 2's hidden
    behaviour model, as in Chapter 5's expected_outcomes(). With boredom < 1 (section
    8.6's planted test), a video's chance of satisfying is multiplied by boredom
    once for every video of the same category shown above it in the row. Products
    in the row (ad slots, section 8.7) can be clicked but give no satisfied play."""
    r = np.arange(1, top.shape[1] + 1)
    keep = cfg.scroll_keep
    on = keep ** (r - 1)
    glimpse = np.where(r >= 2, keep ** np.maximum(r - 2, 0) * (1 - keep), 0.0)
    p_look = (on + cfg.glimpse_look * glimpse) / r ** cfg.look_decay
    filled = top >= 0
    i = np.where(filled, top, 0)
    match = np.einsum("nd,nkd->nk", users["vec"][rows].astype(np.float64), items["vec"][i].astype(np.float64))
    bait = items["clickbait"][i]
    attract = sigmoid(5 * match - 3.5 + bait)
    satisfy = sigmoid(5 * match - 2.0 - 2 * bait)
    cat = np.where(filled & ~items["is_product"][i], items["category"][i], -1)
    repeats = np.zeros(top.shape)
    for k in range(1, top.shape[1]):
        repeats[:, k] = ((cat[:, :k] == cat[:, k:k + 1]) & (cat[:, k:k + 1] >= 0)).sum(axis=1)
    satisfy = satisfy * boredom ** repeats
    click = np.where(filled, p_look * attract, 0.0)
    play = np.where(items["is_product"][i], 0.0, click * (satisfy * P_PLAY_SAT + (1 - satisfy) * P_COMPLETE_UNSAT))
    return click, play


def online(top, rows, W, boredom=1.0):
    """Expected satisfied plays per list, one number per user."""
    return slate_outcomes(top, rows, W["users"], W["items"], W["cfg"], boredom)[1].sum(axis=1)


def row_metrics(top_rows, top_aud, sp, W, boredom_too=True):
    """Offline (test users), online in both worlds, the guardrails and variety (all
    consenting users) for one method's rows."""
    items, S = W["items"], sp["S"]
    nd = np.mean([ndcg_at_k(t.tolist(), {i: 1.0 for i in sp["pos"][u]}) for u, t in zip(sp["rows"], top_rows)])
    hit = np.mean([len(sp["pos"][u] & set(t.tolist())) > 0 for u, t in zip(sp["rows"], top_rows)])
    # the guardrails count videos only, each in its own slot: an ad slot counts for no creator
    videos = np.where((top_aud >= 0) & ~items["is_product"][np.maximum(top_aud, 0)], top_aud, -1)
    cov, cr = guardrails(videos, S["ok"], items["creator"], top_creators(W, S), len(items["vec"]))
    div = np.nanmean([diversity(t.tolist(), items["category"]) for t in top_aud])
    out = {"hit@10": hit, "NDCG@10": nd, "online": online(top_aud, sp["aud"], W),
           "coverage": cov, "top-10% creators": cr, "variety": div}
    if boredom_too:
        out["online, boredom"] = online(top_aud, sp["aud"], W, BOREDOM)
    return out


def top_creators(W, S):
    """Chapter 5's creator groups: the top 10% of video creators by engaged users."""
    items, video = W["items"], W["video"]
    engaged = (S["weight"] > 0).sum(axis=0)
    n_creators = int(items["creator"].max()) + 1
    vc = np.unique(items["creator"][video])
    cpop = np.bincount(items["creator"][video], weights=engaged[video].astype(float), minlength=n_creators)
    return set(vc[np.argsort(-cpop[vc], kind="stable")][: len(vc) // 10].tolist())


def paired(a, b):
    """Mean of a - b over the same users, and its standard error."""
    d = np.asarray(a) - np.asarray(b)
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))


def seed_paired(runs, base_runs):
    """Seed-paired comparison: for each seed, the mean difference over the same users
    (run s against base run s). Returns the mean of those differences, its standard
    error across seeds (training variation), and the mean per-user standard error
    (the noise of one run's measurement)."""
    d = np.array([paired(a, b)[0] for a, b in zip(runs, base_runs)])
    user_se = float(np.mean([paired(a, b)[1] for a, b in zip(runs, base_runs)]))
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d))), user_se


def rules(cfg, users, items, catalogue=None, seed_offset=800):
    """Section 8.2's rule data (synthetic). For a larger catalogue (Chapter 6's),
    pass it; the shares stay the same."""
    rng = np.random.default_rng(cfg.seed + seed_offset)
    n_users = len(users["vec"])
    cat = items if catalogue is None else catalogue
    creators, is_product = cat["creator"], cat["is_product"]
    n_items = len(creators)
    n_creators = int(creators.max()) + 1
    n_regions = 12
    blocks = [rng.choice(n_creators, rng.integers(1, 4), replace=False) if rng.random() < 0.20
              else np.array([], dtype=np.int64) for _ in range(n_users)]
    return {
        "blocked_creators": blocks,                                     # 20% of users block 1 to 3 creators
        "adult": ~is_product & (rng.random(n_items) < 0.03),            # 3% of videos are rated 18+
        "teen": rng.random(n_users) < 0.08,                             # 8% of users have a teen account
        "region": rng.integers(0, n_regions, n_users),                  # where each user is
        "unlicensed": [np.flatnonzero(~is_product & (rng.random(n_items) < 0.04)) for _ in range(n_regions)],
        "removed": ~is_product & (rng.random(n_items) < 0.005),         # taken down by moderation
        "out_of_stock": is_product & (rng.random(n_items) < 0.10),      # products only
        "creators": creators, "is_product": is_product, "n_regions": n_regions,
    }


def check_world_unchanged(W):
    """slate_outcomes() with no boredom must equal Chapter 5's expected_outcomes()."""
    sp_rows = W["consenting"][:200]
    rng = np.random.default_rng(1)
    top = rng.choice(np.flatnonzero(W["video"]), (len(sp_rows), ROW))
    a = expected_outcomes(top, sp_rows, W["users"], W["items"], W["cfg"])[1]
    b = slate_outcomes(top, sp_rows, W["users"], W["items"], W["cfg"])[1]
    return float(np.abs(a - b).max())


__all__ = ["DAY", "KEPT", "POOL", "ROW", "BOREDOM", "SHIP", "world", "split", "slates", "order_by",
           "slate_outcomes", "online", "row_metrics", "paired", "rules", "check_world_unchanged",
           "predict", "tensors", "train", "Ranker", "score_pools", "meta_for", "rerank"]
