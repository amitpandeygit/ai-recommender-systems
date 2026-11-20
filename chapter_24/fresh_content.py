"""Section 24.6 - exploring fresh and long-tail content without hurting engagement: the honest test.

Every arm is today's row, assembled at once exactly as in sections 9.7, 10.8 and 11.8:
section 8.2's filters, at most 2 videos per creator (the new video included), the "New for
you" video at slot 4 and one ad at slot 3, every source obeying each request's own rules,
and a final check that counts any video shown that the request did not allow. The arms
differ only in slot 4:

  A: today's row          - Chapter 11's content model chooses the new video, frozen at day 23
  B: bandit at slot 4     - section 24.4's chosen bandit chooses it, learning all week, with a
                            5% floor (a uniform choice among the 100 candidates); this is the
                            policy that is logged for Chapter 25 (surface "home_slot4", seed 0)
  B without the floor     - the same bandit with no floor
  C: bandit decides       - the bandit also decides whether slot 4 holds a new video at all,
     whether, lambda        with Chapter 8's allocation rule: it shows its chosen new video if
                            its draw of the video's satisfied plays at slot 4, plus an
                            exploration value lambda, is larger than what the new video would
                            push out (the organic videos below it each move down one slot and
                            lose (w_j - w_(j+1)) of their score, with Chapter 8's attention
                            weights w from exploratory slots); otherwise slot 4 is organic

The bandit runs the validation week first and carries what it learned into the test week,
as in section 24.4. Besides the measures of section 11.8, each arm reports what exploration
is for: how many different new videos it showed, how many of them reached Chapter 3's
threshold of 5 engaged users by day 30 (counting the users who had engaged before day 23,
and the week's slot-4 showings, with Chapter 2's event weights), and how many of the 10%
truly best new videos did.

Needs Python 3.12 or later, PyTorch, pyroaring and new_video_data.py's cache. Run from the
repository root (about 12 minutes):
    python chapter_24/fresh_content.py
"""
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
for _ch in ("chapter_09", "chapter_08", "chapter_06", "chapter_05"):
    sys.path.insert(0, str(HERE.parent / _ch))
from bandit_world import Config, implicit_weight, p_look, play_given_look, visit_delays  # noqa: E402
from bandit_logs import Logger, check, events_table, report  # noqa: E402
from contextual import (exploration_rate, features, make, n_features, outcomes_for, run_week,  # noqa: E402
                        week_stats)
from new_video_data import SLOT, week  # noqa: E402
from policies import HybridLinear, LogisticTS, propensities, sigmoid  # noqa: E402
from tesselo_shop import LANGUAGES, build  # noqa: E402

CHOSEN = {"name": "neural-linear Thompson",                                   # section 24.4's best policy
          "setting": {"kappa": 0.005, "alpha": 0.02, "lam_v": 1_600, "lam_h": 100_000.0}}
FLOOR = 0.05
DRAWS = 1_000
SEEDS = 5
LAMBDAS = (0.0, 0.005, 0.01, 0.02)
MIN_USERS = 5                  # Chapter 3's threshold: videos with fewer engaged users are "new"
SEED = Config().seed + 2410    # the same draws as section 24.4


def posterior(pol, Z, v):
    """The policy's mean and spread for each candidate (on the logit scale for LogisticTS)."""
    if isinstance(pol, LogisticTS):
        m = Z @ pol.m[:pol.k] + pol.m[pol.k + v]
        sd = np.sqrt((Z ** 2 / pol.q[:pol.k]).sum(axis=1) + 1 / pol.q[pol.k + v])
        return m, sd
    a, B, bv = pol.a[v], pol.B[v], pol.bv[v]
    mean = pol.kappa * Z[:, 1] + Z @ pol.beta + (bv - B @ pol.beta) / a
    ZA, BA = Z @ pol.A0inv, B @ pol.A0inv
    s = (np.einsum("ck,ck->c", ZA, Z) - 2 * np.einsum("ck,ck->c", ZA, B) / a + 1 / a + np.einsum("ck,ck->c", BA, B) / a ** 2)
    return mean, np.sqrt(np.maximum(s, 0.0))


def as_play(pol, s):
    """A draw of the score as a chance of a satisfied play at slot 4."""
    return sigmoid(s) if isinstance(pol, LogisticTS) else s


def chooser(d, pol, floor, lam=None, loss=None, logs=None, anon=None):
    """The slot-4 choice: with probability floor a uniform candidate, otherwise the
    policy's own choice; with lam, no new video (-2) when the choice's draw plus lam is
    below what it would push out. With logs, every decision is logged with its chances."""
    def choose(r, Z, v, rng):
        n = len(v)
        if logs is not None:
            prob, se = propensities(pol, Z, v, rng, floor=floor, draws=DRAWS)
        s = pol.scores(Z, v, rng, getattr(pol, "alpha", None))
        j = int(rng.integers(n)) if rng.random() < floor else int(np.argmax(s))
        if lam is not None and as_play(pol, s[j]) + lam < loss[r]:
            return -2
        if logs is not None:
            mean, sd = posterior(pol, Z, v)
            logs.add(request_id=int(d["request"][r]), session_id=int(d["session"][r]), user_id=int(d["user"][r]),
                     anonymous_id=int(anon[d["request"][r]]), served_time=float(d["time"][r]), consent_at_time=True,
                     slot=SLOT, model_time=float(np.floor(d["time"][r] / 3_600.0) * 3_600.0), chosen_item=int(v[j]),
                     chosen_prob=float(prob[j]), chosen_pos=j,
                     prob_method="exact" if isinstance(pol, HybridLinear) and pol.mode != "ts" else "monte_carlo",
                     prob_draws=DRAWS, cand_item=v, cand_prob=prob, cand_prob_se=se, cand_mean=mean, cand_sd=sd,
                     cand_x=Z[:, :2], week_row=r, _cand_p=d["_p"][r, :n])
        return j
    return choose


def row_outcomes(d, top, items, cfg=Config()):
    """Expected clicks and satisfied plays per slot, from the hidden taste (Chapter 5)."""
    filled = top >= 0
    i = np.where(filled, top, 0)
    match = np.einsum("nd,nkd->nk", d["_eff"], items["vec"][i].astype(np.float64))
    bait = items["clickbait"][i]
    look = p_look(cfg)[None, :]
    click = np.where(filled, look * 1.0 / (1.0 + np.exp(-(5 * match - 3.5 + bait))), 0.0)
    play = np.where(filled & ~items["is_product"][i], look * play_given_look(match, bait), 0.0)
    return click, play


if __name__ == "__main__":
    from seq_data import forbidden_shown, paired  # noqa: E402
    from pipeline_test import assemble_rows  # noqa: E402
    from honest_test import AD_SLOT, CREATOR_CAP, NEW_SLOT  # noqa: E402
    from slates import check_rows  # noqa: E402
    from two_tower import guardrails  # noqa: E402
    from metrics import ndcg_at_k  # noqa: E402
    from seq_data import load as load_world  # noqa: E402
    from slates import look_curve  # noqa: E402
    t0 = time.time()
    cfg = Config()
    Wl = load_world("Chapter 2's world")
    w = np.r_[look_curve(Wl), 0.0]
    del Wl
    users, items, served, events, _, _ = build(cfg)
    rate = exploration_rate(cfg, items, served, events)[0]
    reconnect = visit_delays(served, events)
    first = served["position"] == 1
    anon = np.zeros(served["request_id"].max() + 1, dtype=np.int64)
    anon[served["request_id"][first]] = served["anonymous_id"][first]
    del users, served, events, first                                # only what was taken from them is needed
    V, T_ = week("validation"), week("test")
    sv, st = week_stats(V), week_stats(T_)
    neural = CHOSEN["name"].startswith("neural")
    zv, zt = features(V, neural, sv), features(T_, neural, st)
    k = n_features(neural)
    n = len(items["vec"])
    nr = len(T_["user"])
    kept, rel, ad = T_["kept"], T_["rel"], T_["ad"]

    def rows_with(new):
        out = []
        for a in range(0, nr, 2_048):
            b = min(a + 2_048, nr)
            out.append(assemble_rows(kept[a:b], rel[a:b], new[a:b], ad[a:b], items))
        return np.concatenate(out)

    # what a new video at slot 4 would push out of the organic row (Chapter 8's rule)
    organic = rows_with(np.full(nr, -1))
    s_slot = np.zeros((nr, 10))
    for r in range(nr):
        m = dict(zip(kept[r].tolist(), rel[r].tolist()))
        s_slot[r] = [m.get(int(x), 0.0) if x >= 0 else 0.0 for x in organic[r]]
    loss = sum((w[j] - w[j + 1]) * s_slot[:, j] for j in range(SLOT - 1, 10))
    print(f"Test week: {nr:,} requests, {len(T_['cols']):,} new videos; the bandit: {CHOSEN['name']} "
          f"{CHOSEN['setting']}; floor {FLOOR}; a new video at slot 4 pushes out on average {loss.mean():.4f} expected "
          f"satisfied plays (Chapter 8's rule)  [{time.time() - t0:.0f} s]")

    ev_t = {s: outcomes_for(T_, items, SEED + 100 + s) for s in range(SEEDS)}
    arms = {"A: today's row": [np.zeros(nr, dtype=int)]}
    specs = [("B: bandit at slot 4", FLOOR, None), ("B without the floor", 0.0, None)]
    specs += [(f"C: bandit decides, lambda {lam}", FLOOR, lam) for lam in LAMBDAS]
    logged = None
    for label, floor, lam in specs:
        arms[label] = []
        for s in range(SEEDS):
            pol = make(CHOSEN["name"], CHOSEN["setting"], k, rate)
            run_week(V, outcomes_for(V, items, SEED + s), pol, zv, SEED + s)   # the validation week first, as in 24.4
            logs = None
            if label.startswith("B: ") and s == 0:
                logs = Logger("home_slot4", f"{CHOSEN['name']} with a uniform floor",
                              f"setting={CHOSEN['setting']}; floor={FLOOR}; draws={DRAWS}; candidates=100; "
                              f"prior=carried from days 16-23; features: cand_x holds the constant and the "
                              f"content gap; the 64 hidden numbers are row week_row of new_video_test.npz's "
                              f"hidden, as differences from the request's average candidate")
            ch = run_week(T_, ev_t[s], pol, zt, SEED + 50 + s,
                          logger=chooser(T_, pol, floor, lam, loss, logs, anon))
            arms[label].append(ch)
            if logs is not None:
                ok = np.flatnonzero(ch >= 0)
                ok = ok[np.argsort(T_["time"][ok], kind="stable")]
                vid = T_["cols"][T_["short"][ok, ch[ok]]]
                evs = {kk: vv[ok, ch[ok]] for kk, vv in ev_t[s].items()}
                E = events_table(T_["request"][ok], T_["session"][ok], T_["user"][ok], T_["time"][ok],
                                 np.ones(len(ok), dtype=bool), np.full(len(ok), SLOT), vid, evs,
                                 items["duration_s"][vid].astype(float), reconnect)
                logged = logs.save(E)
        print(f"  ran {label}  [{time.time() - t0:.0f} s]", flush=True)

    ok_v, tc = T_["ok"], set(T_["top_creators"].tolist())
    new_ok = ~items["is_product"] & ~ok_v
    lang = T_["language"]
    pos = [set(T_["pos_item"][a:b].tolist()) for a, b in zip(T_["pos_ptr"][:-1], T_["pos_ptr"][1:])]
    has_pos = np.flatnonzero([len(p_) > 0 for p_ in pos])
    best10 = np.argsort(-T_["_video_value"])[: len(T_["cols"]) // 10]          # the truly best new videos
    print(f"\n  {'arm':<30}{'NDCG@10':>9}{'coverage':>10}{'top-10%':>9}{'online':>9}{'against A':>11}{'(se)':>9}"
          f"{'new plays':>11}{'rows, new':>11}{'new shown':>11}{'reached 5':>11}{'best 10%':>10}{'revenue':>9}"
          f"{'breaks':>8}{'not allowed':>13}")
    base, results = None, {}
    for label, runs in arms.items():
        vals = {key: [] for key in ("ndcg", "cov", "cr", "on", "d", "se", "newp", "rows", "shown", "grad", "best",
                                    "rev", "breaks", "bad", "lang")}
        for s, ch in enumerate(runs):
            shown = ch >= 0
            new_id = np.where(shown, T_["cols"][T_["short"][np.arange(nr), np.maximum(ch, 0)]], -1)
            if label.startswith("A"):
                new_id = T_["static"]                                     # Chapter 11's choice, Chapter 8's if none
            top = rows_with(new_id)
            click, play = row_outcomes(T_, top, items)
            on = play.sum(axis=1)
            if base is None:
                base = on
            i = np.maximum(top, 0)
            vids = np.where((top >= 0) & ~items["is_product"][i], top, -1)
            cov, cr = guardrails(vids, ok_v, items["creator"], tc, n)
            vals["cov"].append(cov)
            vals["cr"].append(cr)
            vals["on"].append(on.mean())
            m_, se, _, _ = paired(on, base, T_["user"])
            vals["d"].append(m_)
            vals["se"].append(se)
            vals["newp"].append(np.where((top >= 0) & new_ok[i], play, 0.0).sum(axis=1).mean())
            vals["rows"].append(np.mean(new_id >= 0))
            vals["rev"].append(np.where((top >= 0) & items["is_product"][i], click * T_["bid"][i], 0.0).sum(axis=1).mean())
            vals["breaks"].append(check_rows(top, items, CREATOR_CAP, [(new_id, NEW_SLOT), (ad, AD_SLOT)])[0])
            vals["bad"].append(forbidden_shown(top, T_["bad"], items)[0])
            vals["ndcg"].append(np.mean([ndcg_at_k(top[r].tolist(), {x: 1.0 for x in pos[r]}) for r in has_pos]))
            vals["lang"].append([on[lang == g].mean() for g in range(len(LANGUAGES))])
            # exploration's own measures: which new videos were shown, and which reached 5 engaged users
            ev = ev_t[s] if not label.startswith("A") else ev_t[0]
            cidx = np.where(shown, ch, 0) if not label.startswith("A") else np.zeros(nr, dtype=int)
            use = shown if not label.startswith("A") else T_["static"] == T_["cols"][T_["short"][:, 0]]
            evr = {kk: vv[np.arange(nr), cidx] for kk, vv in ev.items()}
            engaged = (implicit_weight(evr) > 0) & use
            col_of = {int(c): j for j, c in enumerate(T_["cols"])}
            cj = np.array([col_of.get(int(x), -1) for x in new_id])
            vals["shown"].append(len(np.unique(cj[(cj >= 0)])))
            pairs = np.unique(np.stack([cj[engaged], T_["user"][engaged]], 1), axis=0)
            gained = np.bincount(pairs[:, 0], minlength=len(T_["cols"])) if len(pairs) else np.zeros(len(T_["cols"]))
            reached = (T_["engaged_before"] + gained >= MIN_USERS)
            vals["grad"].append(int(reached.sum()))
            vals["best"].append(float(reached[best10].mean()))
        results[label] = {kk: np.mean(vv, axis=0) for kk, vv in vals.items()}
        R_ = results[label]
        print(f"  {label:<30}{R_['ndcg']:>9.4f}{R_['cov']:>10.1%}{R_['cr']:>9.1%}{R_['on']:>9.4f}"
              f"{R_['d']:>+11.4f}{f'({R_['se']:.4f})':>9}{R_['newp']:>11.4f}{R_['rows']:>11.1%}{R_['shown']:>11,.0f}"
              f"{R_['grad']:>11,.0f}{R_['best']:>10.1%}{R_['rev']:>9.4f}{R_['breaks']:>8.0f}{R_['bad']:>13.0f}", flush=True)
    print(f"  (online: expected satisfied plays per row; against A: paired over the same requests, standard error from "
          f"1,000 bootstrap draws of whole users, mean over {SEEDS} seeds; new plays: from videos below the history "
          f"threshold; rows, new: share of rows with a new video at slot 4; new shown: distinct new videos at slot 4; "
          f"reached 5: new videos with at least {MIN_USERS} engaged users by day 30; best 10%: share of the "
          f"{len(best10)} truly best new videos that reached it; revenue: expected ad coins per row)  "
          f"[{time.time() - t0:.0f} s]")
    print("  online by app language (" + ", ".join(LANGUAGES) + "): " + " | ".join(
        f"{lbl}: " + " ".join(f"{x:.4f}" for x in r_['lang']) for lbl, r_ in results.items()))
    print(f"  new videos already at {MIN_USERS}+ engaged users before day 23: 0 by definition; with 1 to 4: "
          f"{int(((T_['engaged_before'] > 0) & (T_['engaged_before'] < MIN_USERS)).sum()):,}; with none: "
          f"{int((T_['engaged_before'] == 0).sum()):,}")
    print("\nSection 24.7: checks of the logged slot-4 decisions")
    report("home_slot4", check(logged, FLOOR))
