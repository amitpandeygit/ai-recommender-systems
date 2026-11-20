"""Section 24.4 - contextual and neural bandits for the "New for you" slot.

Tesselo's row keeps slot 4 for a new video. Today the video is chosen by Chapter 11's
content model, fitted once at the cutoff: it never learns from what happens to the
videos it shows during the week. Each policy here chooses the slot-4 video among the
content model's 100 best allowed new videos (new_video_data.py), request by request in
time order, and learns from the satisfied plays that have reached the server, in hourly
batches (an offline phone's visit arrives when it reconnects).

  static content         - today's choice: the content model's best video
  per-video Thompson     - no context: one Beta posterior per video (section 24.3's
                           bandit), with a prior at the rate of Chapter 2's exploratory
                           slots at slot 4
  LinUCB, linear         - the hybrid linear model from a flat prior: it starts knowing
  Thompson, flat prior     nothing, not even that the content score matters
  greedy                 - the hybrid linear model's mean, no exploration
  epsilon-greedy         - the same, with 5% of choices uniform among the candidates
  LinUCB                 - the hybrid linear model, mean + alpha x standard deviation
  linear Thompson        - a draw from the hybrid linear model's posterior
  logistic Thompson      - Chapelle and Li's logistic model with Laplace updates
  neural-linear Thompson - linear Thompson on the heavy ranker's last hidden layer
  neural-linear greedy   - the same model's mean, no exploration

All but the first four sit on top of the content model: their prediction starts as
kappa x the standardised content score, so before any data they rank as today's choice
does, and they learn corrections to it (kappa is a setting, chosen with the others).

The shared features z are a constant and the content gap (how far the video's content
score is below the request's best candidate, in units of its spread); the neural-linear
model adds the heavy ranker's 64 hidden numbers, as differences from the request's
average candidate. Every model also has one term per video. The script first measures the
headroom: what perfect knowledge of each video's effect, or of the best linear weights on
the hidden layer, would bring.

Each policy runs on the validation week (days 16 to 23) from its prior, once for each
setting, and the setting with the most expected satisfied plays there is kept. The kept
setting then runs from day 16 to day 30 with 5 seeds, carrying what it learned into the
test week, which is where it is measured. Each request's outcome is drawn from Chapter 2's
hidden rules with the same random numbers for every policy; the measure is the exact
expected satisfied plays of the chosen video at slot 4, paired against today's choice
with a bootstrap over whole users.

Needs Python 3.12 or later, PyTorch, pyroaring and Chapter 11's text_embeddings.npz (for
new_video_data.py). Run from the repository root (about eight minutes, after the cache):
    python chapter_24/contextual.py
"""
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bandit_world import DAY, Config, Draws  # noqa: E402
from new_video_data import SLOT, week  # noqa: E402
from policies import BetaPerVideo, HybridLinear, LogisticTS  # noqa: E402
from tesselo_shop import LANGUAGES, build  # noqa: E402

SEEDS = 5
N_ITEMS = 5_000
SEED = Config().seed + 2410
GRID = {
    "static content": [{}],
    "per-video Thompson": [{"n0": n} for n in (10.0, 100.0, 1_000.0)],
    "LinUCB, flat prior": [{"alpha": a, "lam_v": 400} for a in (0.02, 0.05, 0.14)],
    "linear Thompson, flat prior": [{"alpha": a, "lam_v": 400} for a in (0.02, 0.05, 0.14)],
    "greedy": [{"kappa": c, "lam_v": v} for c in (0.0025, 0.005, 0.01) for v in (1_600, 6_400)],
    "epsilon-greedy": [{"kappa": c, "lam_v": v} for c in (0.0025, 0.005, 0.01) for v in (1_600, 6_400)],
    "LinUCB": [{"kappa": c, "alpha": a, "lam_v": v} for c in (0.0025, 0.005, 0.01) for a in (0.02, 0.05, 0.14)
               for v in (1_600, 6_400)],
    "linear Thompson": [{"kappa": c, "alpha": a, "lam_v": v} for c in (0.0025, 0.005, 0.01)
                        for a in (0.02, 0.05, 0.14) for v in (1_600, 6_400)],
    "logistic Thompson": [{"kappa": c, "q_video": q} for c in (0.25, 0.5) for q in (64.0, 256.0, 1_024.0)],
    "neural-linear Thompson": [{"kappa": 0.005, "alpha": a, "lam_v": v, "lam_h": h} for a in (0.02, 0.05)
                               for v in (1_600, 6_400) for h in (1e4, 1e5, 1e6)],
    "neural-linear greedy": [{"kappa": 0.005, "lam_v": v, "lam_h": h} for v in (1_600, 6_400) for h in (1e4, 1e5, 1e6)]}


def n_features(neural):
    """How many shared features: the constant, the content gap, and for the neural-linear
    model the heavy ranker's 64 hidden numbers."""
    return 2 + (64 if neural else 0)


def features(d, neural, stats):
    """Listing 24.3. z for every candidate of every request: requests x 100 x k (a function of the
    request index, to keep memory small). The content score enters as its gap below the
    request's best candidate (0 for today's choice), in units of its spread, and the hidden
    numbers as their difference from the request's average candidate. Only differences
    between a request's candidates matter for the choice, and taking them out of each
    request keeps differences between users (some users play more of everything) out of
    the fitted weights."""
    has = d["short"] >= 0
    gap = np.where(has, (d["content"] - d["content"][:, :1]) / stats["c_sd"], 0.0)
    base = np.stack([np.ones_like(gap), gap], axis=-1).astype(np.float32)
    if not neural:
        return lambda r: base[r]
    hs = stats["h_sd"]

    def z(r):
        h = d["hidden"][r].astype(np.float32)
        h = (h - h[has[r]].mean(axis=0)) / hs
        return np.concatenate([base[r], h], axis=-1)
    return z


def week_stats(d):
    """The spreads used to put features in common units: of the content score and of each
    hidden number, over the week's candidates."""
    has = d["short"] >= 0
    h = d["hidden"][has][::17].astype(np.float32)
    return {"c_sd": d["content"][has].std(), "h_sd": h.std(axis=0) + 1e-3}


def outcomes_for(d, items, seed, chunk=4_000):
    """Satisfied plays for every candidate of every request, drawn once from Chapter 2's
    rules with uniforms shared by all candidates of a request (common random numbers).
    Drawn a few thousand requests at a time, to keep memory small."""
    nr, k = d["short"].shape
    D = Draws(nr, seed=seed)
    has = d["short"] >= 0
    vid = np.where(has, d["cols"][np.maximum(d["short"], 0)], 0)
    out = {}
    for a in range(0, nr, chunk):
        b = min(a + chunk, nr)
        rows = np.repeat(np.arange(a, b), k)
        ev = D.outcome(rows, np.full(len(rows), SLOT), np.repeat(d["_eff"][a:b], k, axis=0), items, vid[a:b].ravel())
        for key, val in ev.items():
            val = val.reshape(b - a, k)
            out.setdefault(key, []).append(val & has[a:b] if val.dtype == bool else val)
    return {key: np.concatenate(val) for key, val in out.items()}


def make(name, setting, k, base_rate):
    """A fresh policy. Every name without "flat prior" starts from the content model's
    ranking (kappa) and learns corrections to it."""
    if name == "logistic Thompson":
        return LogisticTS(k, N_ITEMS, q_video=setting["q_video"], offset=np.log(base_rate / (1 - base_rate)),
                          kappa=setting["kappa"])
    if name == "per-video Thompson":
        return BetaPerVideo(N_ITEMS, base_rate, setting["n0"])
    if name == "static content":
        return None
    mode = {"LinUCB": "ucb", "linear Thompson": "ts", "neural-linear Thompson": "ts", "greedy": "greedy",
            "epsilon-greedy": "epsilon", "neural-linear greedy": "greedy"}[name.split(",")[0]]
    lam0 = np.r_[1.0, 1.0, np.full(k - 2, setting.get("lam_h", 1.0))]     # the hidden weights shrink harder
    return HybridLinear(k, N_ITEMS, mode=mode, alpha=setting.get("alpha", 0.0), lam0=lam0, lam_v=setting["lam_v"],
                        eps=0.05 if mode == "epsilon" else 0.0, kappa=setting.get("kappa", 0.0))


def run_week(d, ev, policy, zfun, seed, step=3_600.0, logger=None):
    """One policy over one week, request by request in time order. Returns the index
    (0-99) of the chosen candidate for every request."""
    rng = np.random.default_rng(seed)
    order = np.argsort(d["time"], kind="stable")
    by_arrival = order[np.argsort(d["arrive"][order], kind="stable")]
    chosen = np.full(len(order), -1)
    done, pos = 0, 0
    for r in order:
        boundary = np.floor(d["time"][r] / step) * step
        batch = []
        while pos < len(by_arrival) and d["arrive"][by_arrival[pos]] <= boundary:
            j = by_arrival[pos]
            if chosen[j] >= 0:
                batch.append(j)
            pos += 1
        if batch and policy is not None:
            b = np.array(batch)
            Z = np.stack([zfun(j)[chosen[j]] for j in b])
            v = d["cols"][d["short"][b, chosen[b]]]
            policy.update(Z, v, ev["play"][b, chosen[b]].astype(float))
        n = int((d["short"][r] >= 0).sum())
        if n == 0:
            continue
        if policy is None:
            chosen[r] = 0
        else:
            Z = zfun(r)[:n]
            v = d["cols"][d["short"][r, :n]]
            chosen[r] = policy.choose(Z, v, rng) if logger is None else logger(r, Z, v, rng)
        done += 1
    return chosen


def headroom(d, stats, chunk=2_000):
    """What perfect knowledge would bring, choosing within each request's candidates: the
    content model alone (today's choice); plus each video's true effect (its average
    expected satisfied play over the week's candidate slots, once the content gap's part is
    taken out); a linear model on the content gap and the hidden layer, with its weights
    fitted to the hidden truth; and both. Every fit uses differences from the request's
    average candidate, as the bandits' features do."""
    has = d["short"] >= 0
    n = np.maximum(has.sum(axis=1, keepdims=True), 1)

    def centred(x):
        return np.where(has, x - np.where(has, x, 0.0).sum(axis=1, keepdims=True) / n, 0.0)
    g = centred(features(d, False, stats)(slice(None))[..., 1])
    p = centred(np.where(has, d["_p"], 0.0))
    slope = (g * p).sum() / (g * g).sum()
    v = d["cols"][np.maximum(d["short"], 0)]
    cnt = np.bincount(v[has], minlength=N_ITEMS)
    eff = np.bincount(v[has], (p - slope * g)[has], minlength=N_ITEMS) / np.maximum(cnt, 1)
    zf = features(d, True, stats)

    def x(r):
        return np.c_[g[r], zf(r)[:, 2:]]
    rows = np.random.default_rng(SEED).choice(len(n), 4_000, replace=False)
    w = np.linalg.lstsq(np.concatenate([x(r)[has[r]] for r in rows]),
                        np.concatenate([p[r][has[r]] for r in rows]), rcond=None)[0]
    lin = np.concatenate([np.stack([x(r) @ w for r in range(a, min(a + chunk, len(n)))])
                          for a in range(0, len(n), chunk)])

    def value(score):
        return summary(d, np.argmax(np.where(has, score, -np.inf), axis=1)).mean()
    return {"today": value(g), "video": value(slope * g + eff[v]), "hidden": value(lin), "both": value(lin + eff[v]),
            "slope": slope, "effect_sd": eff[cnt > 0].std()}


def summary(d, ch):
    """Expected satisfied plays of each request's chosen video at slot 4."""
    ok = ch >= 0
    return np.where(ok, d["_p"][np.arange(len(ch)), np.maximum(ch, 0)], 0.0)


def exploration_rate(cfg, items, served, events):
    """Satisfied plays per exploratory slot at slot 4 for videos, from Chapter 2's logs."""
    first = np.unique(events["event_id"], return_index=True)[1]
    e = {key: val[first] for key, val in events.items()}
    play = np.zeros(len(served["item_id"]), dtype=bool)
    hit = np.isin(e["event"], ["complete", "like"]) & (e["impression_id"] >= 0)
    play[e["impression_id"][hit]] = True
    m = served["explored"] & (served["position"] == SLOT) & ~items["is_product"][served["item_id"]]
    return play[m].mean(), int(m.sum()), int(play[m].sum())


if __name__ == "__main__":
    sys.path.insert(0, str(HERE.parent / "chapter_09"))
    from seq_data import paired  # noqa: E402
    t0 = time.time()
    cfg = Config()
    users, items, served, events, _, _ = build(cfg)
    rate, n_ex, n_pl = exploration_rate(cfg, items, served, events)
    print(f"Prior rate from Chapter 2's exploratory slots at slot {SLOT}, videos: {n_pl} satisfied plays in "
          f"{n_ex:,} slots = {rate:.4f}")
    V, T_ = week("validation"), week("test")
    sv, st = week_stats(V), week_stats(T_)
    ev_v = {s: outcomes_for(V, items, SEED + s) for s in range(SEEDS)}
    ev_t = {s: outcomes_for(T_, items, SEED + 100 + s) for s in range(SEEDS)}
    for name, d in (("validation", V), ("test", T_)):
        print(f"{name} week: {len(d['user']):,} requests, {len(d['cols']):,} new videos; expected satisfied plays at "
              f"slot {SLOT}: today's choice {d['_p'][:, 0].mean():.4f}, best of the 100 candidates "
              f"{d['_p'].max(axis=1).mean():.4f}, best allowed new video {d['_p_best'].mean():.4f}  "
              f"[{time.time() - t0:.0f} s]")
    # ---- the headroom: what perfect knowledge of each kind would bring on the test week
    heads = {}
    for name, d, sd in (("validation", V, sv), ("test", T_, st)):
        h = heads[name] = headroom(d, sd)
        shows = np.bincount(d["short"][:, 0][d["short"][:, 0] >= 0])
        shows = shows[shows > 0]
        print(f"Headroom, {name} week (exact expected satisfied plays at slot {SLOT} per request): today's choice "
              f"{h['today']:.4f}; with each video's true effect added {h['video']:.4f}; with a linear model on the "
              f"hidden layer, its weights fitted to the hidden truth {h['hidden']:.4f}; with both {h['both']:.4f}. "
              f"Today's choice shows {len(shows):,} different videos, {shows.mean():.1f} times each on average. "
              f"The true video effects have a spread (standard deviation) of {h['effect_sd']:.4f}; one unit of "
              f"content gap is worth {h['slope']:.4f}")
    # ---- settings on the validation week (seed 0)
    chosen_setting = {}
    print("\nSettings, validation week (seed 0): expected satisfied plays at slot 4 per request")
    for name, grid in GRID.items():
        best, line = None, []
        for setting in grid:
            k = n_features(name.startswith("neural"))
            pol = make(name, setting, k, rate)
            ch = run_week(V, ev_v[0], pol, features(V, name.startswith("neural"), sv), SEED)
            m = summary(V, ch).mean()
            line.append(f"{setting}: {m:.4f}")
            if best is None or m > best[0]:
                best = (m, setting)
        chosen_setting[name] = best[1]
        print(f"  {name:<28} chosen {best[1]}; " + "; ".join(line) + f"  [{time.time() - t0:.0f} s]", flush=True)
    # ---- the test week: the chosen setting, carried on from the validation week, 5 seeds
    print("\nTest week: expected satisfied plays at slot 4 per request (5 seeds; each policy ran the validation week "
          "first and carries what it learned)")
    print(f"  {'policy':<28}{'mean':>8}{'seeds (min-max)':>18}{'against today':>16}{'(se)':>9}{'regret, 100':>13}"
          f"{'regret, all':>13}{'videos shown':>14}{'plays drawn':>13}")
    static_p = summary(T_, np.zeros(len(T_["user"]), dtype=int))
    days = np.floor(T_["time"] / DAY).astype(int)
    lang = T_["language"]
    curves, res = {}, {}
    for name in GRID:
        means, diffs, ses, shown, drawn, per_day, per_lang = [], [], [], [], [], [], []
        for s in range(SEEDS):
            k = n_features(name.startswith("neural"))
            pol = make(name, chosen_setting[name], k, rate)
            run_week(V, ev_v[s], pol, features(V, name.startswith("neural"), sv), SEED + s)
            ch = run_week(T_, ev_t[s], pol, features(T_, name.startswith("neural"), st), SEED + 50 + s)
            p = summary(T_, ch)
            means.append(p.mean())
            m_, se, _, _ = paired(p, static_p, T_["user"])
            diffs.append(m_)
            ses.append(se)
            ok = ch >= 0
            shown.append(len(np.unique(T_["short"][ok, ch[ok]])))
            drawn.append(ev_t[s]["play"][np.arange(len(ch))[ok], ch[ok]].sum())
            per_day.append([p[days == dd].mean() for dd in range(23, 30)])
            per_lang.append([p[lang == g].mean() for g in range(len(LANGUAGES))])
        reg100 = T_["_p"].max(axis=1).mean() - np.mean(means)
        rega = T_["_p_best"].mean() - np.mean(means)
        print(f"  {name:<28}{np.mean(means):>8.4f}{f'({min(means):.4f}-{max(means):.4f})':>18}"
              f"{np.mean(diffs):>+16.4f}{f'({np.mean(ses):.4f})':>9}{reg100:>13.4f}{rega:>13.4f}"
              f"{np.mean(shown):>14,.0f}{np.mean(drawn):>13,.0f}", flush=True)
        curves[name] = np.mean(per_day, axis=0)
        res[name] = (np.mean(means), np.mean(per_lang, axis=0), np.mean(diffs), np.mean(ses))
    print("  (against today: mean over seeds of the paired difference with today's choice, standard error from 1,000 "
          "bootstrap draws of whole users; regret, 100: below the best of the 100 candidates; regret, all: below the "
          f"best allowed new video; videos shown: distinct new videos at slot 4)  [{time.time() - t0:.0f} s]")
    print("  by day of the test week (23 to 29): " + " | ".join(
        f"{n}: " + " ".join(f"{x:.4f}" for x in c) for n, c in curves.items()))
    print("  by app language (" + ", ".join(LANGUAGES) + "): " + " | ".join(
        f"{n}: " + " ".join(f"{x:.4f}" for x in v[1]) for n, v in res.items()))
    np.savez(HERE / "out_contextual_curves.npz", names=np.array(list(curves)), curves=np.array(list(curves.values())),
             diff=np.array([v[2] for v in res.values()]), se=np.array([v[3] for v in res.values()]),
             headroom=np.array([heads["test"][k] - heads["test"]["today"] for k in ("video", "hidden", "both")]))
