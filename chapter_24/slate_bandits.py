"""Section 24.5 - slate bandits for whole lists: Tesselo's "New for you" row.

Chapter 3 gave new videos a row of their own. Here a bandit fills that row: 10 new videos,
in order, chosen from the content model's 100 best allowed new videos for the request
(new_video_data.py). A user reads the row as Chapter 2's users read any row: they scroll on
with probability 0.92, look at slot k less often the lower it is, and click and play by the
hidden taste. So a video shown low gets fewer plays however good it is, and a bandit that
ignores this learns that videos shown low are bad.

Three ways to learn from a row (Listing 24.5), all with section 24.4's linear Thompson
model (a shared part and one term per video), starting from the content model's order:

  position-blind     - every shown video counts as one showing, whatever its slot
  cascade            - the cascade model (Craswell and colleagues, 2008; Kveton and
                       colleagues' cascading bandits, 2015): the user reads from the top
                       and stops at the first play; videos above it count as not played,
                       videos below it are not counted
  position-based     - the position-based model of Chapter 2: a play at slot k has chance
                       w_k x attraction, with w_k the attention of slot k measured on
                       Chapter 2's exploratory slots (Chapter 8's look_curve); a showing at
                       slot k counts as w_k of a showing at slot 1 (Lagrée and colleagues'
                       PBM bandits, 2016)

Each orders its draw of the candidates' attraction, best first, which is the best row
under the position-based model. Settings are chosen on the validation week; the chosen
setting then runs from day 16 to day 30 with 3 seeds, measured on the test week. The
position-based bandit's test run with seed 0 is logged for Chapter 25 (surface
"new_for_you_row"), with a 5% floor: with probability 0.05 the row is 10 candidates in a
uniformly random order. Its slot chances are estimated from 1,000 posterior draws.

Needs Python 3.12 or later and new_video_data.py's cache. Run from the repository root
(about 16 minutes):
    python chapter_24/slate_bandits.py
"""
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "chapter_08"))
from bandit_world import DAY, Config, Draws, p_look, visit_delays  # noqa: E402
from bandit_logs import Logger, check, events_table, recompute, report  # noqa: E402
from contextual import N_ITEMS, features, n_features, week_stats  # noqa: E402
from new_video_data import week  # noqa: E402
from policies import HybridLinear  # noqa: E402
from tesselo_shop import build  # noqa: E402

ROW = 10
SEEDS = 3
FLOOR = 0.05
DRAWS = 1_000
SEED = Config().seed + 2450
KINDS = ("position-blind", "cascade", "position-based")
GRID = [{"kappa": c, "alpha": a, "lam_v": v} for c in (0.005, 0.01, 0.02) for a in (0.02, 0.05) for v in (400, 1_600)]


def observations(kind, z, v, play, w):
    """Listing 24.5. What one shown row teaches: (rows of z, videos, outcomes, weights)."""
    if kind == "position-blind":
        return z, v, play.astype(float), np.ones(ROW)
    if kind == "cascade":
        hit = np.flatnonzero(play)
        last = hit[0] + 1 if len(hit) else ROW                   # read down to the first play
        return z[:last], v[:last], play[:last].astype(float), np.ones(last)
    return z * w[:, None], v, play.astype(float), w               # position-based: slot k counts w_k


def make(g):
    """Section 24.4's linear Thompson model, starting from the content model's order."""
    return HybridLinear(n_features(False), N_ITEMS, mode="ts", alpha=g["alpha"], lam_v=g["lam_v"], kappa=g["kappa"])


def run_week(d, items, D, kind, pol, zfun, w, seed, step=3_600.0, logger=None, cfg=Config()):
    """One row policy over one week; returns each request's row (indices into the 100)."""
    rng = np.random.default_rng(seed)
    order = np.argsort(d["time"], kind="stable")
    by_arrival = order[np.argsort(d["arrive"][order], kind="stable")]
    rows = np.full((len(order), ROW), -1)
    plays = {}
    pos = 0
    for r in order:
        boundary = np.floor(d["time"][r] / step) * step
        batch = []
        while pos < len(by_arrival) and d["arrive"][by_arrival[pos]] <= boundary:
            j = by_arrival[pos]
            if j in plays:
                batch.append(j)
            pos += 1
        if batch and pol is not None:
            obs = [observations(kind, zfun(j)[rows[j]], d["cols"][d["short"][j, rows[j]]], plays[j], w) for j in batch]
            pol.update(np.concatenate([o[0] for o in obs]), np.concatenate([o[1] for o in obs]),
                       np.concatenate([o[2] for o in obs]), np.concatenate([o[3] for o in obs]))
        n = int((d["short"][r] >= 0).sum())
        if n < ROW:
            continue
        if pol is None:
            row = np.arange(ROW)
        elif logger is not None:
            row = logger(r, zfun(r)[:n], d["cols"][d["short"][r, :n]], rng, pol)
        else:
            s = pol.scores(zfun(r)[:n], d["cols"][d["short"][r, :n]], rng, pol.alpha)
            row = np.argsort(-s, kind="stable")[:ROW]
        rows[r] = row
        vid = d["cols"][d["short"][r, row]]
        ev = D.outcome(np.full(ROW, r), np.arange(1, ROW + 1), np.repeat(d["_eff"][r:r + 1], ROW, axis=0), items,
                       vid, cfg)
        plays[r] = ev["play"]
    return rows


def row_value(d, rows, cfg=Config()):
    """Exact expected satisfied plays of each request's row, and of the best possible row."""
    g = d["_p"] / p_look(cfg)[3]                                  # chance of a play given a look
    look = p_look(cfg)
    ok = rows[:, 0] >= 0
    val = np.where(ok, (np.take_along_axis(g, np.maximum(rows, 0), axis=1) * look[None, :]).sum(axis=1), 0.0)
    best = (-np.sort(-g, axis=1)[:, :ROW] * look[None, :]).sum(axis=1)
    return val, np.where(ok, best, 0.0)


def make_logger(d, items, rate_floor, served, logs):
    """A logger for the position-based bandit: the row, and every candidate's chance of
    being in each slot (1,000 posterior draws, mixed with the 5% uniform floor)."""
    def log(r, Z, v, rng, pol):
        n = len(v)
        logs.snapshot(float(np.floor(d["time"][r] / 3_600.0) * 3_600.0), pol)
        S = pol.scores_many(Z, v, rng, DRAWS)                        # draws x candidates
        top = np.argsort(-S, axis=1, kind="stable")[:, :ROW]
        marg = np.zeros((n, ROW))
        np.add.at(marg, (top, np.broadcast_to(np.arange(ROW), top.shape)), 1.0 / DRAWS)
        marg = (1 - FLOOR) * marg + FLOOR / n                       # a uniform row: every slot 1/n
        if rng.random() < FLOOR:
            row = rng.permutation(n)[:ROW]
        else:
            row = np.argsort(-pol.scores(Z, v, rng, pol.alpha), kind="stable")[:ROW]
        mean = pol.scores(Z, v, rng, 0.0) if pol.mode == "ts" else np.zeros(n)
        logs.add(request_id=int(d["request"][r]), session_id=int(d["session"][r]), user_id=int(d["user"][r]),
                 anonymous_id=int(served["anon_of_request"][d["request"][r]]), served_time=float(d["time"][r]),
                 consent_at_time=True, slot=0, model_time=float(np.floor(d["time"][r] / 3_600.0) * 3_600.0),
                 chosen_item=int(v[row[0]]), chosen_prob=float(marg[row[0], 0]), chosen_pos=int(row[0]),
                 prob_method="monte_carlo", prob_draws=DRAWS, cand_item=v, cand_prob=marg[:, 0],
                 cand_prob_se=np.sqrt(marg[:, 0] * (1 - marg[:, 0]) / DRAWS), cand_mean=mean,
                 cand_sd=np.zeros(n), cand_x=Z, _cand_p=d["_p"][r, :n], slate=v[row], slate_marginal=marg)
        return row
    return log


if __name__ == "__main__":
    sys.path.insert(0, str(HERE.parent / "chapter_09"))
    from seq_data import paired  # noqa: E402
    from seq_data import load as load_world  # noqa: E402
    from slates import look_curve  # noqa: E402
    t0 = time.time()
    cfg = Config()
    users, items, served, events, _, _ = build(cfg)
    Wl = load_world("Chapter 2's world")
    w = look_curve(Wl)
    del Wl
    print("Attention by slot, relative to slot 1, from Chapter 2's exploratory slots (Chapter 8's look_curve): "
          + " ".join(f"{x:.3f}" for x in w) + "; the truth (Chapter 5): "
          + " ".join(f"{x:.3f}" for x in p_look(cfg) / p_look(cfg)[0]))
    V, T_ = week("validation"), week("test")
    sv, st = week_stats(V), week_stats(T_)
    Dv = {s: Draws(len(V["user"]), seed=SEED + s) for s in range(SEEDS)}
    Dt = {s: Draws(len(T_["user"]), seed=SEED + 100 + s) for s in range(SEEDS)}
    zv, zt = features(V, False, sv), features(T_, False, st)
    chosen = {}
    static_v = row_value(V, run_week(V, items, Dv[0], "static", None, zv, w, SEED))[0].mean()
    print(f"\nSettings, validation week (seed 0): expected satisfied plays per row (the static content row: "
          f"{static_v:.4f})")
    for kind in KINDS:
        best, line = None, []
        for g in GRID:
            pol = make(g)
            m = row_value(V, run_week(V, items, Dv[0], kind, pol, zv, w, SEED))[0].mean()
            line.append(f"{g}: {m:.4f}")
            if best is None or m > best[0]:
                best = (m, g)
        chosen[kind] = best[1]
        print(f"  {kind:<16} chosen {best[1]}; " + "; ".join(line) + f"  [{time.time() - t0:.0f} s]", flush=True)
    static_rows = run_week(T_, items, Dt[0], "static", None, zt, w, SEED)
    base, best_row = row_value(T_, static_rows)
    ok0 = static_rows[:, 0] >= 0
    print(f"\nTest week, {len(T_['user']):,} requests: static content row {base.mean():.4f} expected satisfied plays per "
          f"row, {len(np.unique(np.take_along_axis(T_['short'], np.maximum(static_rows, 0), axis=1)[ok0])):,} videos "
          f"shown; best possible row of the 100 candidates {best_row.mean():.4f}")
    print(f"  {'policy':<16}{'mean':>8}{'seeds':>26}{'against static':>16}{'(se)':>9}{'regret':>9}{'videos shown':>14}")
    days = np.floor(T_["time"] / DAY).astype(int)
    curves = {}
    for kind in KINDS:
        means, diffs, ses, shown, per_day = [], [], [], [], []
        for s in range(SEEDS):
            pol = make(chosen[kind])
            run_week(V, items, Dv[s], kind, pol, zv, w, SEED + s)
            rows = run_week(T_, items, Dt[s], kind, pol, zt, w, SEED + 50 + s)
            val = row_value(T_, rows)[0]
            means.append(val.mean())
            m_, se, _, _ = paired(val, base, T_["user"])
            diffs.append(m_)
            ses.append(se)
            ok = rows[:, 0] >= 0
            shown.append(len(np.unique(np.take_along_axis(T_["short"], np.maximum(rows, 0), axis=1)[ok])))
            per_day.append([val[days == dd].mean() for dd in range(23, 30)])
        curves[kind] = np.mean(per_day, axis=0)
        print(f"  {kind:<16}{np.mean(means):>8.4f}{'(' + ', '.join(f'{x:.4f}' for x in means) + ')':>26}"
              f"{np.mean(diffs):>+16.4f}{f'({np.mean(ses):.4f})':>9}{best_row.mean() - np.mean(means):>9.4f}"
              f"{np.mean(shown):>14,.0f}  [{time.time() - t0:.0f} s]", flush=True)
    print("  by day (23 to 29): " + " | ".join(f"{k}: " + " ".join(f"{x:.4f}" for x in c) for k, c in curves.items()))
    # ---- the logged run: position-based, seed 0, with the 5% floor
    first = served["position"] == 1
    served["anon_of_request"] = np.zeros(served["request_id"].max() + 1, dtype=np.int64)
    served["anon_of_request"][served["request_id"][first]] = served["anonymous_id"][first]
    g = chosen["position-based"]
    pol = make(g)
    run_week(V, items, Dv[0], "position-based", pol, zv, w, SEED)
    logs = Logger("new_for_you_row", "position-based linear Thompson sampling",
                  f"kappa={g['kappa']}; alpha={g['alpha']}; lam_v={g['lam_v']}; floor={FLOOR}; draws={DRAWS}; prior=carried from days 16-23")
    rows = run_week(T_, items, Dt[0], "position-based", pol, zt, w, SEED + 50, logger=make_logger(T_, items, FLOOR,
                                                                                                  served, logs))
    val = row_value(T_, rows)[0]
    m_, se, _, _ = paired(val, base, T_["user"])
    ok = np.flatnonzero(rows[:, 0] >= 0)
    rr = np.repeat(ok, ROW)
    slots = np.tile(np.arange(1, ROW + 1), len(ok))
    vids = T_["cols"][T_["short"][rr, rows[ok].ravel()]]
    ev = Dt[0].outcome(rr, slots, T_["_eff"][rr], items, vids, cfg)
    order = np.argsort(T_["time"][ok], kind="stable")                 # the logger saw requests in time order
    rr, slots, vids = (x.reshape(len(ok), ROW)[order].ravel() for x in (rr, slots, vids))
    ev = {k: v.reshape(len(ok), ROW)[order].ravel() for k, v in ev.items()}
    E = events_table(T_["request"][rr], T_["session"][rr], T_["user"][rr], T_["time"][rr],
                     np.ones(len(rr), dtype=bool), slots, vids, ev, items["duration_s"][vids].astype(float),
                     visit_delays(served, events))
    L = logs.save(E)
    print(f"\nLogged run (position-based, seed 0, floor {FLOOR}): {val.mean():.4f} per row, against static "
          f"{m_:+.4f} ({se:.4f}); {len(L['decision_id']):,} rows, {len(E['event_id']):,} events  "
          f"[{time.time() - t0:.0f} s]")
    report("new_for_you_row", check(L, FLOOR))
    pick = np.random.default_rng(1).choice(len(L["decision_id"]), 300, replace=False)
    got, noise = recompute(L, pick)
    print(f"    redrawn from the logged policy snapshots, 300 decisions: mean |redrawn - logged| slot chance {got:.2e}; "
          f"between two redraws {noise:.2e}")
    np.savez(HERE / "out_slate_curves.npz", names=np.array(list(curves)), curves=np.array(list(curves.values())))
