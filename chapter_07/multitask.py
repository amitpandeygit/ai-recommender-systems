"""Section 7.5 - Multi-task ranking: clicks are not enough.

Every model reads the same inputs, the eight dense features (the inputs of
section 7.4's best ranker online, the MLP with no id vectors), and is tested in
the same way as the rankers of heavy_ranking.py:
  click only           - one task; score = P(click). The clickbait trap.
  two separate models  - one model per task; score = P(click) x P(play | click)
  shared bottom        - one network for every task, one small tower per task
  MMoE                 - four experts, mixed by one gate per task
  PLE                  - per level, two shared experts and two per task; two levels
The multi-task models learn three tasks: click, satisfied play given a click, and
the share of the video watched given a click; the separate models learn only the
first two. Two scores are tried for MMoE: P(click) x P(play | click) (the goal
of Chapter 5) and P(click) x expected watch share (a watch-share objective, not
watch time, which would also need each video's length). Every design is trained
from three seeds.

Epochs are chosen on the validation week, by the summed validation loss of the
model's tasks after each epoch of one training run, as in heavy_ranking.py.

Needs Python 3.12 or later and PyTorch. Run from the repository root (about six minutes):
    python chapter_07/multitask.py
"""
import os
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "2"              # NumPy's linear algebra: the same thread count on every run
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ranking_data import DAY, Config, build, candidate_pools, snapshot  # noqa: E402
from rankers import MultiTask, Ranker, meta_for, rerank, take, tensors, train  # noqa: E402
from heavy_ranking import EPOCHS, POOL, full_order, impression_auc, rows_for  # noqa: E402
from similarity_recs import test_positives  # noqa: E402
from evaluate import expected_outcomes, ranking_table  # noqa: E402
from metrics import log_loss  # noqa: E402


def outputs(model, b, n, position=None, batch=20_000):
    """Sigmoid of every head, for n rows (slot 1 unless a position is given)."""
    out = {}
    with torch.no_grad():
        for start in range(0, n, batch):
            idx = torch.arange(start, min(start + batch, n))
            o = model(take(b, idx), position=None if position is None else position[idx])
            for k, v in o.items():
                out.setdefault(k, []).append(torch.sigmoid(v).numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def pool_scores(models, combine, S, users, items, rows, pools, chunk=256):
    """combine(outs) -> score per candidate; outs[k] = outputs of models[k]."""
    out = np.full(pools.shape, -np.inf)
    for start in range(0, len(rows), chunk):
        r, p = rows[start:start + chunk], pools[start:start + chunk]
        uu, ii = np.repeat(r, p.shape[1]), p.ravel()
        ok = ii >= 0
        b = tensors([S], users, items, uu[ok], ii[ok], np.zeros(ok.sum(), dtype=np.int64))
        s = np.full(len(ii), -np.inf)
        s[ok] = combine([outputs(m, b, int(ok.sum())) for m in models])
        out[start:start + chunk] = s.reshape(p.shape)
    return out


def validation_loss(o, R):
    """Summed log loss (and mean squared error for watch share) on the validation rows,
    from o[task] = each task's outputs."""
    c = R["click"]
    loss = log_loss(o["click"], c) if "click" in o else 0.0
    if "play" in o:
        loss += log_loss(o["play"][c], R["play"][c])
    if "watch" in o:
        loss += float(np.mean((o["watch"][c] - R["watch"][c]) ** 2))
    return loss


SPECS = {   # name: list of (constructor, tasks) per model, and how to score
    "click only": ([(lambda meta, seed: Ranker(meta, "mlp, no ids", seed=seed), ("click",))], lambda o: o[0]["click"]),
    "two separate models": ([(lambda meta, seed: Ranker(meta, "mlp, no ids", seed=seed), ("click",)),
                             (lambda meta, seed: Ranker(meta, "mlp, no ids", seed=seed), ("play",))],
                            lambda o: o[0]["click"] * o[1]["play"]),
    "shared bottom": ([(lambda meta, seed: MultiTask(meta, "shared", seed=seed), ("click", "play", "watch"))],
                      lambda o: o[0]["click"] * o[0]["play"]),
    "MMoE": ([(lambda meta, seed: MultiTask(meta, "mmoe", seed=seed), ("click", "play", "watch"))],
             lambda o: o[0]["click"] * o[0]["play"]),
    "PLE": ([(lambda meta, seed: MultiTask(meta, "ple", seed=seed), ("click", "play", "watch"))],
            lambda o: o[0]["click"] * o[0]["play"]),
}


def build_models(spec, meta, b, R, epochs, seed=0):
    out = []
    for k, (make, tasks) in enumerate(spec):
        out.append(train(make(meta, seed + 10 * k), b, R, epochs=epochs, seed=seed + 10 * k, tasks=tasks))
    return out


if __name__ == "__main__":
    t0 = time.time()
    cfg = Config()
    users, items, served, events, orders, truth = build(cfg)
    video = ~items["is_product"]
    snaps = {d: snapshot(events, served, users, items, d * DAY) for d in (9, 16, 23)}
    week1 = rows_for(events, served, users, items, [(9, 16)])
    week2 = rows_for(events, served, users, items, [(16, 23)])
    weeks12 = rows_for(events, served, users, items, [(9, 16), (16, 23)])
    week3 = rows_for(events, served, users, items, [(23, 30)])
    b1 = tensors([snaps[9]], users, items, week1["user"], week1["item"], week1["snap"])
    b2 = tensors([snaps[16]], users, items, week2["user"], week2["item"], week2["snap"])
    b12 = tensors([snaps[9], snaps[16]], users, items, weeks12["user"], weeks12["item"], weeks12["snap"])
    b3 = tensors([snaps[23]], users, items, week3["user"], week3["item"], week3["snap"])
    meta1 = meta_for(users, items, b1["dense"].numpy())
    meta12 = meta_for(users, items, b12["dense"].numpy())

    print("Section 7.5, validation week: summed validation loss of each model's tasks after each epoch")
    chosen = {}
    pos2 = torch.as_tensor(week2["position"])
    for name, (spec, _) in SPECS.items():
        per_epoch = {e: {} for e in EPOCHS}            # epoch -> task -> validation outputs
        for k, (make, tasks) in enumerate(spec):
            m = make(meta1, 10 * k)

            def check(e, m=m, tasks=tasks):
                o = outputs(m, b2, len(week2["click"]), position=pos2)
                per_epoch[e].update({t: v for t, v in o.items() if t in tasks})

            train(m, b1, week1, epochs=max(EPOCHS), seed=10 * k, tasks=tasks, after_epoch=check)
        losses = [validation_loss(per_epoch[e], week2) for e in EPOCHS]
        chosen[name] = EPOCHS[int(np.argmin(np.round(losses, 9)))]
        print(f"  {name:<22}" + "".join(f"{x:>9.4f}" for x in losses) + f"   chosen: {chosen[name]} epoch(s)")
    print(f"[{time.time() - t0:.0f} s]")

    S = snaps[23]
    pos = test_positives(events, users, 23 * DAY, 30 * DAY, S["used"], video)
    rows = np.array(sorted(pos))
    aud = np.flatnonzero(users["current_consent"])
    pools_rows = candidate_pools(S, users, items, rows, POOL)
    pools_aud = candidate_pools(S, users, items, aud, POOL)
    bait = items["clickbait"] > 0
    c3 = week3["click"]
    print(f"\nTest week: {len(rows):,} test users offline, {len(aud):,} consenting users online; "
          f"pools of {POOL}; held-out impressions of days 23 to 30 for the task columns")
    print(f"  {'':<34}{'NDCG@10':>8}{'online, seeds 0, 1, 2':>24}{'mean':>8}{'clicks':>8}{'clickbait':>10}"
          f"{'click AUC':>10}{'play AUC':>9}{'watch corr.':>12}")
    trained = {}
    for name, (spec, combine) in SPECS.items():
        variants = [(name, combine)]
        if name == "MMoE":
            variants.append(("MMoE, score P(click) x watch share", lambda o: o[0]["click"] * o[0]["watch"]))
        online = {label: [] for label, _ in variants}
        for seed in (0, 1, 2):
            ms = build_models(spec, meta12, b12, weeks12, chosen[name], seed=seed)
            for label, comb in variants:
                top_a = rerank(pools_aud, pool_scores(ms, comb, S, users, items, aud, pools_aud), 10)
                clicks, plays = expected_outcomes(top_a, aud, users, items, cfg)
                online[label].append(plays.sum(axis=1).mean())
                if seed == 0:
                    online[label + " (seed 0)"] = (top_a, clicks)
            if seed == 0:
                trained[name] = ms
        ms = trained[name]
        o3 = {}
        for m, (_, tasks) in zip(ms, spec):               # each task's output from the model trained for it
            o = outputs(m, b3, len(c3), position=torch.as_tensor(week3["position"]))
            o3.update({k: v for k, v in o.items() if k in tasks})
        for label, comb in variants:
            top_r = rerank(pools_rows, pool_scores(ms, comb, S, users, items, rows, pools_rows), 10)
            nd = ranking_table(full_order(top_r, pools_rows), rows, pos, S["used"], video)["NDCG@10"]
            top_a, clicks = online[label + " (seed 0)"]
            share = bait[np.maximum(top_a, 0)][top_a >= 0].mean()
            pa = impression_auc(o3["play"][c3], week3["play"][c3]) if "play" in o3 else float("nan")
            wc = np.corrcoef(o3["watch"][c3], week3["watch"][c3])[0, 1] if "watch" in o3 else float("nan")
            print(f"  {label:<34}{nd:>8.3f}" + "".join(f"{x:>8.4f}" for x in online[label])
                  + f"{np.mean(online[label]):>8.4f}{clicks.sum(axis=1).mean():>8.3f}"
                  f"{share:>10.1%}{impression_auc(o3['click'], c3):>10.3f}{pa:>9.3f}{wc:>12.3f}")
    print("  (online = expected satisfied plays per list, for models trained from seeds 0, 1 and 2; the other "
          "columns are seed 0: clicks = expected clicks per list, clickbait = share of shown slots holding a "
          "clickbait video, all consenting users; AUCs and watch correlation on the held-out impressions)")
    print(f"  clickbait videos: {bait[video].mean():.1%} of videos; in Chapter 6's pools of {POOL}: "
          f"{bait[np.maximum(pools_aud, 0)][pools_aud >= 0].mean():.1%}")

    # What the gates learned: MMoE's mean expert weights per task on the held-out rows
    mm = trained["MMoE"][0]
    with torch.no_grad():
        x = mm.inputs.dense(take(b3, torch.arange(20_000)))
        g = [torch.softmax(gate(x), dim=1).mean(0).numpy() for gate in mm.gates]
    print("  MMoE, mean gate weight on each of the 4 experts (20,000 held-out rows):")
    for t, w in zip(MultiTask.TASKS, g):
        print(f"    {t:<6}" + "".join(f"{v:>7.2f}" for v in w))
    print(f"[{time.time() - t0:.0f} s]")
