"""Listing 7.5 - Pre-ranking and the truncation trade-off (section 7.6).

Retrieval hands over 500 candidates per user. The pre-ranker keeps N of them, the
heavy ranker (heavy_ranking.py's shipped model) orders those N, and the user sees
the top 10. Three ways to choose the N:
  retrieval order - no pre-ranker: keep the first N of the pool
  pre-ranker, labels    - a small MLP on five cheap columns (rankers.PreRanker),
                          trained on the impressions like the heavy ranker
  pre-ranker, distilled - the same small model, trained to imitate the heavy
                          ranker: over each user's pool, its softmax should match
                          the share of the pool's expected satisfied plays that
                          the heavy ranker gives each candidate
For each N: the share of the heavy ranker's own top 10 (out of all 500) that
survives, the recall of the test users' relevant videos, the simulated online
test, and the time per request on this machine.

N is chosen on the validation week, as the smallest N at which the distilled
pre-ranker keeps at least 90% of the heavy ranker's top 10.

Needs Python 3.12 or later and PyTorch. Run from the repository root (about two minutes):
    python chapter_07/pre_ranking.py
"""
import os
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "2"              # NumPy's linear algebra: the same thread count on every run
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ranking_data import DAY, Config, build, candidate_pools, snapshot  # noqa: E402
from rankers import Ranker, meta_for, predict, rerank, score_pools, tensors, train  # noqa: E402
from heavy_ranking import POOL, SHIP, rows_for  # noqa: E402
from similarity_recs import test_positives  # noqa: E402
from evaluate import expected_outcomes  # noqa: E402

KEEP = (10, 25, 50, 100, 200, 500)


def distil(student, teacher_scores, S, users, items, rows, pools, epochs=20, lr=0.05, batch=64, seed=0):
    """Listwise distillation: for each user, cross-entropy between the teacher's
    shares of expected satisfied plays over the pool and the softmax of the
    student's click logit over the same pool."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    ok = pools >= 0
    target = np.where(ok, teacher_scores, 0.0)
    target = torch.as_tensor(target / target.sum(axis=1, keepdims=True), dtype=torch.float32)
    uu = np.repeat(rows, pools.shape[1])
    b = tensors([S], users, items, uu, np.maximum(pools, 0).ravel(), np.zeros(len(uu), dtype=np.int64))
    mask = torch.as_tensor(ok)
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    n, k = pools.shape
    for _ in range(epochs):
        perm = torch.randperm(n, generator=gen)
        for start in range(0, n, batch):
            r = perm[start:start + batch]
            idx = (r[:, None] * k + torch.arange(k)[None, :]).ravel()
            sub = {key: b[key][idx] for key in ("u", "item", "creator", "category", "ilang", "ulang", "dense")}
            sub["H"], sub["HW"] = b["H"], b["HW"]
            logit = student(sub)["click"].view(len(r), k).masked_fill(~mask[r], -1e9)
            loss = -(target[r] * F.log_softmax(logit, dim=1)).sum(1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    student.eval()
    return student


def student_scores(student, S, users, items, rows, pools, chunk=256):
    out = np.full(pools.shape, -np.inf)
    for start in range(0, len(rows), chunk):
        r, p = rows[start:start + chunk], pools[start:start + chunk]
        uu, ii = np.repeat(r, p.shape[1]), p.ravel()
        ok = ii >= 0
        b = tensors([S], users, items, uu[ok], ii[ok], np.zeros(ok.sum(), dtype=np.int64))
        s = np.full(len(ii), -np.inf)
        s[ok] = np.log(predict(student, b, int(ok.sum()))[0])
        out[start:start + chunk] = s.reshape(p.shape)
    return out


def stage(snaps_train, rows_train, S_distil, distil_rows, users, items, events, served):
    """Train the heavy ranker and both pre-rankers on data known at the end of rows_train."""
    b = tensors(snaps_train, users, items, rows_train["user"], rows_train["item"], rows_train["snap"])
    meta = meta_for(users, items, b["dense"].numpy())
    heavy = train(Ranker(meta, SHIP["arch"]), b, rows_train, epochs=SHIP["epochs"])
    labels = train(Ranker(meta, "pre-ranker"), b, rows_train, epochs=SHIP["epochs"])
    pools = candidate_pools(S_distil, users, items, distil_rows, POOL)
    teacher = score_pools(heavy, S_distil, users, items, distil_rows, pools)
    distilled = distil(Ranker(meta, "pre-ranker", position=False), teacher, S_distil, users, items,
                       distil_rows, pools)
    return heavy, labels, distilled


def sweep(heavy, pre, S, users, items, rows, pos, cfg, measure_online=True):
    """Every N, for every way of choosing the N. pos may be None (no recall)."""
    pools = candidate_pools(S, users, items, rows, POOL)
    full = score_pools(heavy, S, users, items, rows, pools)
    best10 = rerank(pools, full, 10)
    orders = {"retrieval order": np.tile(-np.arange(POOL, dtype=float), (len(rows), 1))}
    orders["pre-ranker, labels"] = score_pools(pre["labels"], S, users, items, rows, pools)
    orders["pre-ranker, distilled"] = student_scores(pre["distilled"], S, users, items, rows, pools)
    out = {}
    for name, sc in orders.items():
        sc = np.where(pools >= 0, sc, -np.inf)
        rank = np.argsort(-sc, axis=1, kind="stable")             # pool positions, pre-ranker's order
        for n in KEEP:
            keep = rank[:, :n]
            kept = np.where(np.take_along_axis(sc, keep, axis=1) > -np.inf,
                            np.take_along_axis(pools, keep, axis=1), -1)
            hs = np.where(kept >= 0, np.take_along_axis(full, keep, axis=1), -np.inf)
            top = rerank(kept, hs, 10)                            # the heavy ranker orders the N kept
            share = np.mean([len(set(a[a >= 0]) & set(t[t >= 0])) / max(1, (a >= 0).sum())
                             for a, t in zip(best10, top)])
            rec = (np.mean([len(pos[u] & set(k_.tolist())) / len(pos[u]) for u, k_ in zip(rows, kept)])
                   if pos is not None else float("nan"))
            per_user = (expected_outcomes(top, rows, users, items, cfg)[1].sum(axis=1)
                        if measure_online else np.full(len(rows), np.nan))
            out[(name, n)] = (share, rec, float(per_user.mean()), per_user)
    return out


def latency(model, S, users, items, rows, n, repeats=200):
    """Median milliseconds for one request: the model scores n candidates of one user
    (inputs already built; fetching features is not timed). Each request is timed
    over 20 calls and divided by 20."""
    pools = candidate_pools(S, users, items, rows[:repeats], POOL)
    times = []
    with torch.no_grad():
        for r in range(repeats):
            ii = pools[r][pools[r] >= 0][:n]
            b = tensors([S], users, items, np.full(len(ii), rows[r]), ii, np.zeros(len(ii), dtype=np.int64))
            model(b)                                              # warm-up
            t = time.perf_counter()
            for _ in range(20):                                   # 20 calls, so the timer's own noise is small
                model(b)
            times.append((time.perf_counter() - t) / 20)
    return 1000 * float(np.median(times))


if __name__ == "__main__":
    t0 = time.time()
    cfg = Config()
    users, items, served, events, orders, truth = build(cfg)
    video = ~items["is_product"]
    snaps = {d: snapshot(events, users, items, d * DAY) for d in (9, 16, 23)}
    consenting = np.flatnonzero(users["current_consent"])

    # ---- validation week: models from days 9 to 16, distilled on day-9 pools, tested on day-16 pools
    week1 = rows_for(events, served, users, items, [(9, 16)])
    d9 = np.random.default_rng(cfg.seed + 700).choice(consenting, 3_000, replace=False)
    heavy, labels, distilled = stage([snaps[9]], week1, snaps[9], np.sort(d9), users, items, events, served)
    S16 = snaps[16]
    pos16 = test_positives(events, users, 16 * DAY, 23 * DAY, S16["used"], video)
    r16 = np.array(sorted(pos16))
    res = sweep(heavy, {"labels": labels, "distilled": distilled}, S16, users, items, r16, pos16, cfg,
                measure_online=False)
    print(f"Validation week, {len(r16):,} users: share of the heavy ranker's top 10 that survives pre-ranking "
          f"to N")
    print(f"  {'':<24}" + "".join(f"{'N=' + str(n):>8}" for n in KEEP))
    for name in ("retrieval order", "pre-ranker, labels", "pre-ranker, distilled"):
        print(f"  {name:<24}" + "".join(f"{res[(name, n)][0]:>8.3f}" for n in KEEP))
    chosen = next(n for n in KEEP if res[("pre-ranker, distilled", n)][0] >= 0.9)
    print(f"Chosen: N = {chosen}, the smallest N at which the distilled pre-ranker keeps 90% "
          f"[{time.time() - t0:.0f} s]")

    # ---- test week: models from days 9 to 23, distilled on day-16 pools
    weeks12 = rows_for(events, served, users, items, [(9, 16), (16, 23)])
    d16 = np.random.default_rng(cfg.seed + 701).choice(consenting, 3_000, replace=False)
    heavy, labels, distilled = stage([snaps[9], snaps[16]], weeks12, snaps[16], np.sort(d16),
                                     users, items, events, served)
    S = snaps[23]
    pos = test_positives(events, users, 23 * DAY, 30 * DAY, S["used"], video)
    rows = np.array(sorted(pos))
    aud = consenting
    pre = {"labels": labels, "distilled": distilled}
    off = sweep(heavy, pre, S, users, items, rows, pos, cfg, measure_online=False)
    onl = sweep(heavy, pre, S, users, items, aud, None, cfg)
    onl_users = {k: v[3] for k, v in onl.items()}
    print(f"\nTest week: top-10 kept (test users), recall of relevant videos in the N kept ({len(rows):,} test "
          f"users), online (expected satisfied plays, {len(aud):,} consenting users)")
    print(f"  {'':<24}{'N':>5}{'top-10 kept':>13}{'recall':>8}{'online':>9}")
    for name in ("retrieval order", "pre-ranker, labels", "pre-ranker, distilled"):
        for n in KEEP:
            print(f"  {name:<24}{n:>5}{off[(name, n)][0]:>13.3f}{off[(name, n)][1]:>8.3f}"
                  f"{onl[(name, n)][2]:>9.4f}" + ("   <- chosen" if name == "pre-ranker, distilled"
                                                   and n == chosen else ""))
    base = onl_users[("retrieval order", POOL)]
    print(f"  online, paired difference against all {POOL} (standard error), all consenting users:")
    for name in ("retrieval order", "pre-ranker, distilled"):
        for n in (50, 100, 200):
            d = onl_users[(name, n)] - base
            print(f"    {name:<24}{n:>5} {d.mean():+.4f} ({d.std(ddof=1) / np.sqrt(len(d)):.4f})")
    print(f"[{time.time() - t0:.0f} s]")

    print("\nTime per request on this machine (median of 200 requests, milliseconds; one reference run)")
    print(f"  pre-ranker on {POOL} candidates: {latency(distilled, S, users, items, aud, POOL):.3f} ms")
    for n in KEEP:
        print(f"  heavy ranker on {n:>3} candidates: {latency(heavy, S, users, items, aud, n):.3f} ms")
    print(f"[{time.time() - t0:.0f} s]")
