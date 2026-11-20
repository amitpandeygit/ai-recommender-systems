"""Sections 24.2 and 24.3 - the cost of Chapter 2's exploration, and three ways to explore.

24.2  What Chapter 2's exploratory slots cost. The old system put a uniformly random item
      into 10% of its slots (Listing 2.3). From the logs: the satisfied plays of exploratory
      and normal slots at each position. From the hidden truth: the expected satisfied
      plays of each exploratory item against the item the old system had ranked into that
      slot (ranked_slates replays the draws and is checked against the logs).
24.3  A worked example by hand (printed), then a simulation. Tesselo's editors pick 20 new
      videos per app language for a card at slot 4 of the home row, the same for everyone
      who uses that language, logged in or not; a bandit chooses which video the card shows
      on each request. The stream is every request of Chapter 2's 30 days, in time order,
      and the outcome is drawn from Chapter 2's hidden rules with its exact chance
      (expected_play). Each policy runs 200 times with the same random numbers.

Policies (Listing 24.1): greedy, epsilon-greedy (fixed and decaying), UCB1, KL-UCB and
Thompson sampling with a flat Beta(1, 1) prior, or with a prior worth 100 showings taken
from Chapter 2's exploratory slots or from its normal slots. Regret is measured against the best of the 20 videos for that
language (pseudo-regret: the gap in expected satisfied plays at each step). Outcomes
reach the bandit either at once, or in hourly or daily batches with Chapter 2's upload
delays (an offline phone's visit arrives when it reconnects).

Needs Python 3.12 or later. Run from the repository root (about 40 minutes):
    python chapter_24/k_armed.py
"""
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bandit_world import (DAY, Config, arrival_times, expected_play, p_look, ranked_slates,  # noqa: E402
                          visit_delays)
from tesselo_shop import LANGUAGES, build  # noqa: E402

K = 20                     # videos on each language's card
SLOT = 4                   # the card's slot (Chapter 8's slot for new videos)
RUNS = 200
SEED = Config().seed + 2400


# ---------------------------------------------------------------- Listing 24.1: the policies
def kl(p, q):
    """Bernoulli Kullback-Leibler divergence KL(p || q), safe at 0 and 1."""
    p, q = np.clip(p, 1e-12, 1 - 1e-12), np.clip(q, 1e-12, 1 - 1e-12)
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


def choose(policy, S, N, t, rng, prior=(1.0, 1.0)):
    """Listing 24.1. One choice per run. S, N: plays and impressions per run and video
    that have reached the bandit; t: decisions so far (1, 2, ...). Returns the chosen
    video per run and the probability that the policy gave that choice (None where it
    has no closed form)."""
    runs, k = S.shape
    mean = S / np.maximum(N, 1)
    untried = N == 0
    noise = rng.random((runs, k)) * 1e-9                              # random tie-breaks
    if policy.startswith("Thompson"):
        a0, b0 = prior
        draw = rng.beta(a0 + S, b0 + N - S)
        return np.argmax(draw, axis=1), None                         # propensity: by Monte Carlo (24.7)
    if policy == "UCB1":
        bonus = np.sqrt(2 * np.log(t) / np.maximum(N, 1))
        index = np.where(untried, np.inf, mean + bonus)
        return np.argmax(index + noise, axis=1), np.ones(runs)        # deterministic
    if policy == "KL-UCB":                                            # largest q with N kl(mean, q) <= ln t
        lo, hi = mean.copy(), np.ones_like(mean)
        for _ in range(25):
            mid = (lo + hi) / 2
            ok = N * kl(mean, mid) <= np.log(t)
            lo, hi = np.where(ok, mid, lo), np.where(ok, hi, mid)
        index = np.where(untried, np.inf, lo)
        return np.argmax(index + noise, axis=1), np.ones(runs)
    greedy = np.argmax(np.where(untried, np.inf, mean) + noise, axis=1)   # every video once, then the best
    if policy == "greedy":
        return greedy, np.ones(runs)
    eps = {"epsilon-greedy, 0.01": 0.01, "epsilon-greedy, 0.05": 0.05, "epsilon-greedy, 0.10": 0.10,
           "epsilon-greedy, decaying": min(1.0, 10 * k / t)}[policy]
    explore = rng.random(runs) < eps
    pick = np.where(explore, rng.integers(0, k, runs), greedy)
    prob = np.where(pick == greedy, 1 - eps + eps / k, eps / k)      # exact
    return pick, prob


def run(policy, p, U, times, arrive, mode, rng, prior=(1.0, 1.0)):
    """One policy over one stream, RUNS runs at once. p: exact chance of a satisfied play
    (requests x videos); U: one uniform per run and request (the same for every policy).
    mode: 'at once' (each outcome is known before the next request), 'hourly' or 'daily'
    (outcomes that have reached the server are added at each hour or day boundary).
    Returns the chosen videos (runs x requests) and the satisfied plays drawn."""
    runs, T = U.shape
    S, N = np.zeros((runs, p.shape[1])), np.zeros((runs, p.shape[1]))
    chosen = np.empty((runs, T), dtype=np.int64)
    reward = np.empty((runs, T))
    step = {"hourly": 3_600.0, "daily": DAY}.get(mode)
    order = np.argsort(arrive, kind="stable")
    done, rr = 0, np.arange(runs)
    for t in range(T):
        if step is not None:                                          # fold in what had arrived by the last boundary
            boundary = np.floor(times[t] / step) * step
            while done < T and arrive[order[done]] <= boundary:
                j = order[done]
                np.add.at(S, (rr, chosen[:, j]), reward[:, j])
                np.add.at(N, (rr, chosen[:, j]), 1.0)
                done += 1
        a, _ = choose(policy, S, N, t + 1, rng, prior)
        chosen[:, t] = a
        reward[:, t] = U[:, t] < p[t, a]
        if step is None:
            S[rr, a] += reward[:, t]
            N[rr, a] += 1.0
    return chosen, reward


# ---------------------------------------------------------------- section 24.2
def exploration_cost(cfg, users, items, served, events):
    """What Chapter 2's exploratory slots cost, from the logs and from the hidden truth."""
    k = cfg.slate_size
    ranked = ranked_slates(cfg).ravel()
    logged = served["item_id"]
    ex = served["explored"]
    ok = np.array_equal(ranked[~ex], logged[~ex])
    print(f"Replay of the old system's ranking: identical to the logs in every normal slot: {ok}; "
          f"exploratory slots {ex.sum():,} of {len(ex):,} ({ex.mean():.1%}); requests {len(ex) // k:,}")
    first = np.unique(events["event_id"], return_index=True)[1]
    e = {key: val[first] for key, val in events.items()}
    play = np.zeros(len(logged), dtype=bool)
    hit = np.isin(e["event"], ["complete", "like"]) & (e["impression_id"] >= 0)
    play[e["impression_id"][hit]] = True
    click = np.zeros(len(logged), dtype=bool)
    click[e["impression_id"][(e["event"] == "click") & (e["impression_id"] >= 0)]] = True
    pos = served["position"]
    u = served["_true_user"]
    exp_logged = expected_play(users["vec"][u], items, logged, pos, cfg)
    exp_ranked = expected_play(users["vec"][u], items, ranked, pos, cfg)
    print("\nSection 24.2: satisfied plays per served slot (logged completions or likes, one per impression)")
    print(f"  {'position':>8}{'normal':>9}{'exploratory':>13}{'clicks, normal':>16}{'clicks, expl.':>15}"
          f"{'expected, ranked item':>23}{'expected, explored item':>25}")
    for q in range(1, k + 1):
        n_, e_ = (pos == q) & ~ex, (pos == q) & ex
        print(f"  {q:>8}{play[n_].mean():>9.4f}{play[e_].mean():>13.4f}{click[n_].mean():>16.3f}"
              f"{click[e_].mean():>15.3f}{exp_ranked[e_].mean():>23.4f}{exp_logged[e_].mean():>25.4f}")
    n_req = len(logged) // k
    lost = (exp_ranked[ex] - exp_logged[ex]).sum()
    total = exp_logged.sum()
    print(f"  expected satisfied plays over the month: {total:,.1f} as served; exploratory slots, explored items "
          f"{exp_logged[ex].sum():,.1f}, the ranked items they replaced {exp_ranked[ex].sum():,.1f}; lost {lost:,.1f} = "
          f"{lost / n_req * 1000:.2f} per 1,000 requests = {lost / (total + lost):.1%} of what the ranked slates would "
          f"have brought")
    print(f"  logged: {play[ex].sum():,} satisfied plays in {ex.sum():,} exploratory slots "
          f"({play[ex].mean():.4f} each) against {play[~ex].mean():.4f} in normal slots")


# ---------------------------------------------------------------- section 24.3: the worked example
def worked_example():
    """Three videos after 52 requests: the numbers of the worked example."""
    S = np.array([2.0, 1.0, 0.0])
    N = np.array([40.0, 10.0, 2.0])
    t = int(N.sum())
    mean = S / N
    print("\nSection 24.3, worked example: videos A, B, C with plays / impressions "
          + ", ".join(f"{int(s)}/{int(n)}" for s, n in zip(S, N)) + f"; t = {t}")
    eps = 0.10
    best = int(np.argmax(mean))
    prob = np.where(np.arange(3) == best, 1 - eps + eps / 3, eps / 3)
    print(f"  epsilon-greedy (0.10): means {np.round(mean, 4).tolist()}; best is {'ABC'[best]}; "
          f"choice probabilities {np.round(prob, 4).tolist()}")
    lnt = np.log(t)
    bonus = np.sqrt(2 * lnt / N)
    print(f"  UCB1: ln {t} = {lnt:.4f}; bonuses sqrt(2 x {lnt:.4f} / n) = {np.round(bonus, 4).tolist()}; "
          f"indices {np.round(mean + bonus, 4).tolist()}")
    q = []
    for m, n in zip(mean, N):
        lo, hi = m, 1.0
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if n * kl(np.array(m), np.array(mid)) <= lnt else (lo, mid)
        q.append(lo)
    print(f"  KL-UCB: largest q with n x KL(mean, q) <= {lnt:.4f}: {np.round(q, 4).tolist()}")
    a, b = 1 + S, 1 + N - S
    print("  Thompson: posteriors " + ", ".join(f"{v} Beta({x:.0f}, {y:.0f})" for v, x, y in zip("ABC", a, b))
          + f"; means {np.round(a / (a + b), 4).tolist()}")
    rng = np.random.default_rng(SEED)
    one = rng.beta(a, b)
    print(f"  one draw (seed {SEED}): {np.round(one, 4).tolist()}, so the card shows {'ABC'[int(np.argmax(one))]}")
    draws = np.random.default_rng(SEED + 1).beta(a, b, size=(1_000_000, 3))
    share = np.bincount(np.argmax(draws, axis=1), minlength=3) / len(draws)
    print(f"  chance that Thompson sampling chooses each video (1,000,000 draws): {np.round(share, 4).tolist()}")
    mu = np.array([0.020, 0.012, 0.008])
    gaps = mu.max() - mu
    print(f"  regret, if the true chances were {mu.tolist()}: gaps {gaps.tolist()}; "
          f"after these pulls {gaps.tolist()} x {N.astype(int).tolist()} = {float(gaps @ N):.3f} satisfied plays")


# ---------------------------------------------------------------- section 24.3: the simulation
def streams(cfg, users, items, served, events):
    """Each language's request stream, its 20 card videos and the exact chances."""
    rng = np.random.default_rng(SEED)
    first = np.flatnonzero(served["position"] == 1)
    first = first[np.argsort(served["served_time"][first], kind="stable")]     # every request, in time order
    u = served["_true_user"][first]
    t = served["served_time"][first]
    sess = served["session_id"][first]
    rc = visit_delays(served, events)
    arrive = arrival_times(t, sess, rc)
    lang = users["language"][u]
    video = ~items["is_product"]
    # a prior from the logs: satisfied plays per exploratory slot at the card's slot, for videos in each language
    fe = np.unique(events["event_id"], return_index=True)[1]
    ev = {key: val[fe] for key, val in events.items()}
    play = np.zeros(len(served["item_id"]), dtype=bool)
    hit = np.isin(ev["event"], ["complete", "like"]) & (ev["impression_id"] >= 0)
    play[ev["impression_id"][hit]] = True
    out = []
    for g in range(len(LANGUAGES)):
        pool = np.flatnonzero(video & (items["language"] == g))
        arms = np.sort(rng.choice(pool, K, replace=False))
        m = lang == g
        p = expected_play(users["vec"][u[m]][:, None, :], items, arms[None, :], SLOT, cfg)
        ex = (served["explored"] & (served["position"] == SLOT) & video[served["item_id"]]
              & (items["language"][served["item_id"]] == g))
        nm = (~served["explored"] & (served["position"] == SLOT) & video[served["item_id"]]
              & (items["language"][served["item_id"]] == g))
        out.append({"lang": LANGUAGES[g], "arms": arms, "p": p, "times": t[m], "arrive": arrive[m],
                    "prior_rate": play[ex].mean(), "prior_n": int(ex.sum()), "prior_plays": int(play[ex].sum()),
                    "normal_rate": play[nm].mean(), "normal_n": int(nm.sum())})
    return out


if __name__ == "__main__":
    t0 = time.time()
    cfg = Config()
    users, items, served, events, orders, truth = build(cfg)
    exploration_cost(cfg, users, items, served, events)
    worked_example()
    S_ = streams(cfg, users, items, served, events)
    print(f"\nSection 24.3: a card at slot {SLOT} (P(look) {p_look(cfg)[SLOT - 1]:.3f}), {K} videos per language, every "
          f"request of the 30 days, {RUNS} runs  [{time.time() - t0:.0f} s]")
    for s in S_:
        mu = s["p"].mean(axis=0)
        s["mu"] = mu
        best = mu.max()
        gaps = best - mu
        T = len(s["p"])
        d_kl = kl(mu, np.full_like(mu, best))
        lr = np.sum(np.divide(gaps, d_kl, out=np.zeros_like(gaps), where=gaps > 0)) * np.log(T)
        s["lai_robbins"] = lr
        bait = items["clickbait"][s["arms"]] > 0
        print(f"  {s['lang']}: {T:,} requests; chance of a satisfied play per showing: best {best:.4f}, "
              f"median {np.median(mu):.4f}, worst {mu.min():.4f}; {bait.sum()} clickbait; best video always: "
              f"{best * T:,.1f} plays; Lai-Robbins lower bound on regret {lr:,.1f}; prior from exploratory slots: "
              f"{s['prior_plays']} plays in {s['prior_n']:,} slots = {s['prior_rate']:.4f}; from normal slots "
              f"{s['normal_rate']:.4f} ({s['normal_n']:,} slots)")
    policies = [("greedy", "at once"), ("epsilon-greedy, 0.01", "at once"), ("epsilon-greedy, 0.05", "at once"),
                ("epsilon-greedy, 0.10", "at once"), ("epsilon-greedy, decaying", "at once"), ("UCB1", "at once"),
                ("KL-UCB", "at once"), ("Thompson sampling", "at once"),
                ("Thompson, prior from exploratory slots", "at once"), ("Thompson, prior from normal slots", "at once"),
                ("Thompson sampling", "hourly"), ("Thompson sampling", "daily"),
                ("KL-UCB", "hourly"), ("KL-UCB", "daily")]
    marks = (1_000, 10_000)
    print(f"\n  {'policy':<40}{'updates':>9}{'regret at 1,000':>17}{'at 10,000':>11}{'at the end':>12}"
          f"{'(5th-95th pct)':>18}{'per 1,000':>11}{'best video':>12}{'plays':>10}")
    curves = {}
    for pol, mode in policies:
        reg_end, reg_marks, best_share, plays, per_lang = [], {m: [] for m in marks}, [], [], {}
        cum_total = None
        for gi, s in enumerate(S_):
            T = len(s["p"])
            U = np.random.default_rng([SEED, 7, gi]).random((RUNS, T))
            prior = (1.0, 1.0)
            if pol == "Thompson, prior from exploratory slots":
                prior = (1.0 + 100 * s["prior_rate"], 1.0 + 100 * (1 - s["prior_rate"]))   # worth 100 showings
            if pol == "Thompson, prior from normal slots":
                prior = (1.0 + 100 * s["normal_rate"], 1.0 + 100 * (1 - s["normal_rate"]))
            rng = np.random.default_rng([SEED, 11, gi])
            chosen, reward = run(pol, s["p"], U, s["times"], s["arrive"], mode, rng, prior)
            gap = s["mu"].max() - s["mu"][chosen]                          # runs x requests
            cum = np.cumsum(gap, axis=1)
            reg_end.append(cum[:, -1])
            for m in marks:
                reg_marks[m].append(cum[:, min(m, T) - 1])
            best_share.append((chosen == np.argmax(s["mu"])).mean(axis=1))
            plays.append(reward.sum(axis=1))
            per_lang[s["lang"]] = cum[:, -1].mean()
            if s["lang"] == "hi":
                curves[(pol, mode)] = cum.mean(axis=0)[::50]
        tot = np.sum(reg_end, axis=0)                                      # summed over languages, per run
        n_all = sum(len(s["p"]) for s in S_)
        lo, hi = np.percentile(tot, [5, 95])
        print(f"  {pol:<40}{mode:>9}{np.sum([np.mean(r) for r in reg_marks[1_000]]):>17.2f}"
              f"{np.sum([np.mean(r) for r in reg_marks[10_000]]):>11.2f}{tot.mean():>12.1f}"
              f"{f'({lo:.1f}-{hi:.1f})':>18}{tot.mean() / n_all * 1000:>11.3f}{np.mean(best_share):>12.1%}"
              f"{np.sum([np.mean(p_) for p_ in plays]):>10.1f}")
        print("      by language: " + "  ".join(f"{k_} {v:.1f}" for k_, v in per_lang.items()))
    best_total = sum(s["mu"].max() * len(s["p"]) for s in S_)
    lr_total = sum(s["lai_robbins"] for s in S_)
    print(f"  (regret: expected satisfied plays lost against always showing each language's best video, summed over "
          f"the six languages and averaged over {RUNS} runs; at 1,000 and 10,000: after that many requests in each "
          f"language; best video: share of showings; plays: satisfied plays drawn. Always the best video: "
          f"{best_total:,.1f} expected plays; Lai-Robbins bound summed: {lr_total:,.1f})  [{time.time() - t0:.0f} s]")
    # ---- the long run: the same month of requests replayed four times in a row (the same users, the
    # same tastes, so the same chances), to see what the regret does after the first month
    MONTHS, LONG_RUNS = 4, 50
    print(f"\n  The long run: the month's requests replayed {MONTHS} times, {LONG_RUNS} runs; regret summed over "
          f"languages at the end of each month (the share of showings of the best video in the last month)")
    long_curves = {}
    for pol in ("epsilon-greedy, 0.10", "epsilon-greedy, decaying", "UCB1", "KL-UCB", "Thompson sampling",
                "Thompson, prior from exploratory slots"):
        ends, last_best = np.zeros((LONG_RUNS, MONTHS)), []
        for gi, s in enumerate(S_):
            T = len(s["p"])
            p4 = np.tile(s["p"], (MONTHS, 1))
            t4 = np.concatenate([s["times"] + m * 30 * DAY for m in range(MONTHS)])
            a4 = np.concatenate([s["arrive"] + m * 30 * DAY for m in range(MONTHS)])
            U = np.random.default_rng([SEED, 13, gi]).random((LONG_RUNS, MONTHS * T))
            prior = (1.0, 1.0)
            if pol == "Thompson, prior from exploratory slots":
                prior = (1.0 + 100 * s["prior_rate"], 1.0 + 100 * (1 - s["prior_rate"]))
            chosen, _ = run(pol, p4, U, t4, a4, "at once", np.random.default_rng([SEED, 17, gi]), prior)
            cum = np.cumsum(s["mu"].max() - s["mu"][chosen], axis=1)
            ends += cum[:, [(m + 1) * T - 1 for m in range(MONTHS)]]
            last_best.append((chosen[:, -T:] == np.argmax(s["mu"])).mean())
            if s["lang"] == "hi":
                long_curves[pol] = cum.mean(axis=0)[::200]
        print(f"  {pol:<40}" + "".join(f"{x:>10.1f}" for x in ends.mean(axis=0))
              + f"   ({np.mean(last_best):.1%})  [{time.time() - t0:.0f} s]", flush=True)
    np.savez(HERE / "out_k_armed_curves.npz", **{f"{p_}|{m_}": c for (p_, m_), c in curves.items()},
             **{f"long|{p_}": c for p_, c in long_curves.items()}, hi_T=len(S_[1]["p"]))
