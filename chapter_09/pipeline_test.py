"""The honest test of Chapter 9: a sequence model inside Tesselo's pipeline (section 9.7).

Every arm ends in Chapter 8's production row: section 8.2's filters, at most 2 videos
per creator (the new video included), the best "New for you" video at slot 4 and one
ad at slot 3, assembled at once (Chapter 8's assemble and check_rows). Every source
obeys each request's own rules: a video clicked earlier this week, for example, is
replaced by the pre-ranker's next video, and the "New for you" slot by the user's
next new video; a final check counts any video shown that the request did not allow.
The arms differ only in the organic videos:

  A: Chapter 8's row      - the pre-ranker's first 50 allowed videos from the day-23
                            pool, in the heavy ranker's order
  + sequence candidates    - the 50 plus the sequence model's top 50 for this request,
                            all in the heavy ranker's order
  + blended score          - the 50, in the order of a blend of the heavy ranker and the
                            sequence model
  + both                   - the 100, in the blend's order

and three controls, each the blended score with one thing changed, to see where its
gain comes from:

  history frozen           - the sequence model reads only the clicks known at day 23
                            (day 16 on the validation week), as a daily snapshot would
  order shuffled           - the same clicks as the blend, in a random order (5 random
                            orders, each with its own blend; the first is in the table)
  co-occurrence            - section 9.3's unordered co-occurrence score on the same fresh
                            clicks, in place of the sequence model

The sequence model is the one section 9.4's validation week chose in each world
(seq_data.CHOICE), refitted on what had arrived by day 23, reading what had arrived
before each request. Each blend is a logistic regression on held-out impressions of
the validation week (days 16 to 23, with every model fitted on what had arrived by
day 16), as in section 8.4: the heavy ranker's logit, the second model's score (the
sequence model's log probability, or the log of the co-occurrence score), whether the
visit has a click yet, and the second score again once the visit has a click, so its
weight can grow once the visit has shown its intent.

Differences against A are paired over the same requests, with standard errors and 95%
intervals from 1,000 bootstrap draws of whole users (seq_data.paired).

Needs Python 3.12 or later, PyTorch and pyroaring. Run from the repository root (about 12 minutes;
add 0 or 1 to run one world only; with no number, each world runs in a process of
its own, one after the other, to keep memory down):
    python chapter_09/pipeline_test.py
"""
import os
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "2"
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from seq_data import (CHOICE, DAY, WORLDS, eligible_kept, environment, exclude_mask, forbidden_shown,  # noqa: E402
                      histories, load, ndcg, outcomes, paired, ranked_pools, requests, training_sequences)
import baselines as B  # noqa: E402
from seq_models import SeqModel, fit, score  # noqa: E402
from slate_data import rules, split, top_creators  # noqa: E402
from ranking_data import LANGUAGES  # noqa: E402
from filtering import RuleSets  # noqa: E402
from honest_test import CREATOR_CAP, NEW_SLOT, AD_SLOT  # noqa: E402
from hybrids import fit_logistic  # noqa: E402
from rankers import score_pools  # noqa: E402
from slates import ad_setup, assemble, best_ads, check_rows, score_order  # noqa: E402
from similarity_recs import NO, content_scores, item_features, top_k  # noqa: E402
from two_tower import guardrails  # noqa: E402

EXTRA = 50                                                      # sequence candidates added per request
SECOND = ("sequence model", "history frozen", "order shuffled", "co-occurrence")
SHUFFLES = 5                                                    # random orders tried for the shuffled control


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def log_softmax_rows(s, video):
    s = np.where(video[None, :], s, -np.inf)
    m = s.max(axis=1, keepdims=True)
    return s - m - np.log(np.exp(s - m).sum(axis=1, keepdims=True))


def features(heavy_p, seq_lp, in_visit):
    """Listing 9.7 (first part). The blend's inputs for each (request, candidate).
    seq_lp is the second model's score: the sequence model's log probability."""
    v = (in_visit > 0).astype(float)
    return np.stack([np.ones_like(heavy_p), logit(heavy_p), seq_lp, v * np.ones_like(heavy_p),
                     np.where(v > 0, seq_lp, 0.0)], axis=-1)


def heldout_rows(W, R, H, in_visit):
    """Validation impressions: the videos of each request that were on screen or
    clicked, with the satisfied play as the label."""
    s, ev, items = W["served"], W["events"], W["items"]
    seen = np.zeros(len(s["item_id"]), dtype=bool)
    hit = np.isin(ev["event"], ("viewable", "click")) & (ev["impression_id"] >= 0)
    seen[ev["impression_id"][hit]] = True
    req_row = {int(r): k for k, r in enumerate(R["request"])}
    rows, cand, y = [], [], []
    keep = seen & ~items["is_product"][s["item_id"]] & np.isin(s["request_id"], R["request"])
    for imp in np.flatnonzero(keep):
        k = req_row[int(s["request_id"][imp])]
        rows.append(k)
        cand.append(int(s["item_id"][imp]))
        y.append(cand[-1] in R["pos"][k])
    return np.array(rows), np.array(cand), np.array(y, dtype=float)


def new_candidates(W, S, rows, R, k=5):
    """Chapter 8's "New for you" source (slates.new_for_you), keeping each user's k
    best new videos instead of one, so that a request that may not show the first
    (say, because the user clicked it earlier this week) takes the next."""
    items, users = W["items"], W["users"]
    video = W["video"]
    feats = item_features(items, len(items["vec"]))
    new_ok = video & ~S["ok"] & ~R["removed"]
    same = (users["language"][rows, None] == items["language"][None, :]).astype(np.float64)
    score_ = np.where(new_ok[None, :], same + 0.5 * content_scores(S["weight"][rows], feats)[0], NO)
    for k_, u in enumerate(rows):                                # the user-specific rules
        ban = np.isin(items["creator"], R["blocked_creators"][u]) | np.isin(np.arange(len(video)),
                                                                            R["unlicensed"][R["region"][u]])
        if R["teen"][u]:
            ban |= R["adult"]
        score_[k_, ban] = NO
    return top_k(score_, S["used"][rows], video, k)


def first_allowed(cands, bad):
    """For each request, the first of its candidates (requests x k, -1 padded) that it
    may show, or -1."""
    ok = (cands >= 0) & ~np.take_along_axis(bad, np.maximum(cands, 0), axis=1)
    j = np.argmax(ok, axis=1)
    return np.where(ok.any(axis=1), cands[np.arange(len(cands)), j], -1)


def shuffled(H, seed):
    """The same clicks in a random order (left-padded histories)."""
    rng = np.random.default_rng(seed)
    out = H.copy()
    for k in range(len(H)):
        m = H[k] >= 0
        out[k, m] = rng.permutation(H[k, m])
    return out


def inputs(W, R, H, iv, cutoff, seed):
    """The history and visit flag each second model reads, for requests R."""
    Hf, _ = histories(W, dict(R, time=np.full(len(R["time"]), float(cutoff))))
    out = {"sequence model": (H, iv), "history frozen": (Hf, np.zeros_like(iv)),
           "order shuffled": (shuffled(H, seed), iv), "co-occurrence": (H, iv)}
    for j in range(1, SHUFFLES):                                    # the same control with other random orders
        out[f"order shuffled {j + 1}"] = (shuffled(H, seed + 1_000 * j), iv)
    return out


def second_scores(kind, m, G, H, video):
    """Requests x videos: the second model's score for every video."""
    if kind == "co-occurrence":
        return np.log(B.cooc_scores(G, H) + 1e-3)
    return log_softmax_rows(score(m, H), video)


def fit_blends(W, n):
    """Listing 9.7 (second part). Every model fitted on what had arrived by day 16;
    one blend per second model, fitted on the impressions of days 16 to 23."""
    sp = split(W, "validation")
    Rv = requests(W, 16, 23)
    Hv, ivv = histories(W, Rv)
    kind, loss, ep = CHOICE[W["name"]]
    m = fit(SeqModel(n, kind, W["items"]), training_sequences(W, 16 * DAY)[0], ep, loss)
    G = B.cooccurrence(training_sequences(W, 16 * DAY)[0], n)
    rows, cand, y = heldout_rows(W, Rv, Hv, ivv)
    video = ~W["items"]["is_product"]
    hp = score_pools(sp["heavy"], sp["S"], W["users"], W["items"], Rv["user"][rows], cand[:, None])[:, 0]
    weights = {}
    for name, (Hx, ivx) in inputs(W, Rv, Hv, ivv, 16 * DAY, seed=16).items():
        lp = np.empty(len(rows))
        for a in range(0, len(Rv["user"]), 2_048):
            sel = (rows >= a) & (rows < a + 2_048)
            s = second_scores(name, m, G, Hx[a:a + 2_048], video)
            lp[sel] = s[rows[sel] - a, cand[sel]]
        weights[name] = fit_logistic(features(hp, lp, ivx[rows]), y)
    return weights, len(y), int(y.sum())


def assemble_rows(cands, rel, new, ad, items):
    return assemble(cands, lambda allowed, above: score_order(cands, rel, allowed), items,
                    CREATOR_CAP, [(new, NEW_SLOT), (ad, AD_SLOT)])


if __name__ == "__main__":
    t0 = time.time()
    if not sys.argv[1:]:                                           # each world in its own process, to save memory
        print(environment())
        for k in range(len(WORLDS)):
            subprocess.run([sys.executable, "-u", __file__, str(k)], check=True)
        sys.exit()
    worlds = [WORLDS[int(a)] for a in sys.argv[1:]]               # one world, by its number
    for name in worlds:
        W = load(name)
        users, items = W["users"], W["items"]
        n = len(items["vec"])
        video = ~items["is_product"]
        wts, n_rows, n_pos = fit_blends(W, n)
        print(f"\n{name}: sequence model {', '.join(str(x) for x in CHOICE[name])} epochs; blends fitted on "
              f"{n_rows:,} validation impressions ({n_pos:,} satisfied)  [{time.time() - t0:.0f} s]")
        for k, w in ((k, w) for k, w in wts.items() if k in SECOND):
            print(f"  blend with {k + ':':<16} {w[0]:+.3f} {w[1]:+.3f} x heavy logit {w[2]:+.3f} x second score "
                  f"{w[3]:+.3f} x visit has a click {w[4]:+.3f} x second score once the visit has a click")
        sp = split(W, "test")
        S = sp["S"]
        R = requests(W, 23, 30)
        H, iv = histories(W, R)
        bad = exclude_mask(W, S, R, H)
        kind, loss, ep = CHOICE[name]
        m = fit(SeqModel(n, kind, items), training_sequences(W, 23 * DAY)[0], ep, loss)
        G = B.cooccurrence(training_sequences(W, 23 * DAY)[0], n)
        rl = rules(W["cfg"], users, items)
        rs = RuleSets(rl, [np.flatnonzero(S["used"][u]) for u in range(len(users["vec"]))], n, "numpy")
        uu, inv = np.unique(R["user"], return_inverse=True)
        ranked = ranked_pools(sp, W, rs, uu)
        kept = eligible_kept(ranked, inv, bad)                    # the pre-ranker's first 50 allowed videos
        new_k = new_candidates(W, S, uu, rl)[inv]
        new = first_allowed(new_k, bad)
        ads = ad_setup(W, W["cfg"], rl)
        ad = best_ads(W, S, uu, ads)[0][:, 0][inv]
        old = ranked[:, :50][inv]                                 # the snapshot's 50 and new video, kept all week
        old_hp = score_pools(sp["heavy"], S, users, items, R["user"], old)
        old_top = assemble_rows(old, np.where(old >= 0, old_hp, -np.inf), new_k[:, 0], ad, items)
        n_bad, n_req = forbidden_shown(old_top, bad, items)
        print(f"  without the request-time check, Chapter 8's weekly row would show {n_bad:,} videos that the request "
              f"did not allow (most already clicked), in {n_req:,} requests")
        inp = inputs(W, R, H, iv, 23 * DAY, seed=23)
        arms = {}
        for a in range(0, len(R["user"]), 2_048):
            b = min(a + 2_048, len(R["user"]))
            sec = {k: second_scores(k, m, G, Hx[a:b], video) for k, (Hx, _) in inp.items()}
            lp = sec["sequence model"]
            s_ = np.where(bad[a:b], -np.inf, lp)                      # the sequence model's own top 50
            seq_top = np.argsort(-s_, axis=1, kind="stable")[:, :EXTRA]
            seq_top = np.where(np.isfinite(np.take_along_axis(s_, seq_top, axis=1)), seq_top, -1)
            k50 = kept[a:b]
            dup = np.array([np.isin(x, np.r_[y, z]) for x, y, z in zip(seq_top, k50, new[a:b])])  # no video twice
            both = np.concatenate([k50, np.where(dup, -1, seq_top)], axis=1)
            for cand_label, cands in (("50", k50), ("100", both)):
                hp = score_pools(sp["heavy"], S, users, items, R["user"][a:b], np.maximum(cands, 0))
                hp = np.where(cands >= 0, hp, np.nan)
                orders = {"heavy": hp}
                for k in (list(inp) if cand_label == "50" else SECOND[:1]):
                    second = np.where(cands >= 0, np.take_along_axis(sec[k], np.maximum(cands, 0), axis=1), 0.0)
                    orders[k] = features(np.nan_to_num(hp, nan=0.5), second, inp[k][1][a:b, None]) @ wts[k]
                for score_label, sc in orders.items():
                    rel = np.where(cands >= 0, np.nan_to_num(sc, nan=-np.inf), -np.inf)
                    arms.setdefault((cand_label, score_label), []).append(
                        assemble_rows(cands, rel, new[a:b], ad[a:b], items))
        labels = {("50", "heavy"): "A: Chapter 8's row", ("100", "heavy"): "+ sequence candidates",
                  ("50", "sequence model"): "+ blended score", ("100", "sequence model"): "+ both",
                  ("50", "history frozen"): "control: history frozen", ("50", "order shuffled"): "control: shuffled",
                  ("50", "co-occurrence"): "control: co-occurrence"}
        later = R["place"] >= 1
        ok = S["ok"]
        tc = top_creators(W, S)
        print(f"Test week: {len(R['user']):,} requests of consenting, logged-in users ({later.mean():.1%} later in "
              f"their visit)  [{time.time() - t0:.0f} s]")
        print(f"  {'':<26}{'NDCG@10':>9}{'coverage':>10}{'top-10% cr.':>12}{'online':>9}{'later':>9}"
              f"{'against A':>18}{'95% interval':>20}{'later, against A':>19}{'revenue':>9}{'rule breaks':>13}"
              f"{'not allowed':>13}")
        base = None
        res = {}
        for key, label in labels.items():
            top = np.concatenate(arms[key])
            click, play = outcomes(W, R, top)
            on = play.sum(axis=1)
            base = on if base is None else base
            vids = np.where((top >= 0) & ~items["is_product"][np.maximum(top, 0)], top, -1)
            cov, cr = guardrails(vids, ok, items["creator"], tc, n)
            i = np.maximum(top, 0)
            rev = np.where((top >= 0) & items["is_product"][i], click * ads["bid"][i], 0.0).sum(axis=1)
            breaks = check_rows(top, items, CREATOR_CAP, [(new, NEW_SLOT), (ad, AD_SLOT)])[0]
            shown_bad = forbidden_shown(top, bad, items)[0]
            if on is base:
                d = ci = dl = ""
            else:
                m_, se, lo, hi = paired(on, base, R["user"])
                d, ci = f"{m_:+.4f} ({se:.4f})", f"[{lo:+.4f}, {hi:+.4f}]"
                dl = "{:+.4f} ({:.4f})".format(*paired(on[later], base[later], R["user"][later])[:2])
            print(f"  {label:<26}{ndcg(R, top):>9.4f}{cov:>10.1%}{cr:>12.1%}{on.mean():>9.4f}{on[later].mean():>9.4f}"
                  f"{d:>18}{ci:>20}{dl:>19}{rev.mean():>9.4f}{breaks:>13}{shown_bad:>13}")
            res[label] = on
        ons = {k: outcomes(W, R, np.concatenate(arms[("50", k)]))[1].sum(axis=1) for k in inp}
        blend = ons["sequence model"]
        direct = []
        for k, lbl in (("history frozen", "history frozen"), ("order shuffled", "shuffled"),
                       ("co-occurrence", "co-occurrence")):
            direct.append(lbl + " {:+.4f} ({:.4f}) [{:+.4f}, {:+.4f}]".format(*paired(blend, ons[k], R["user"])))
        print("  The blend against each control (blend minus control, paired over the same requests, users "
              "resampled): " + "; ".join(direct))
        shuf = [k for k in inp if k.startswith("order shuffled")]
        gains = ", ".join(f"{paired(ons[k], base, R['user'])[0]:+.4f}" for k in shuf)
        gaps = ", ".join("{:+.4f} [{:+.4f}, {:+.4f}]".format(*np.array(paired(blend, ons[k], R["user"]))[[0, 2, 3]])
                         for k in shuf)
        print(f"  The shuffled control with {len(shuf)} random orders: against A {gains}; blend minus shuffled "
              f"{gaps}  (one fitted sequence model; each order has its own blend, fitted on the validation week)")
        lang = users["language"][R["user"]]
        print("  online by app language: " + "  ".join(
            f"{LANGUAGES[g]} ({(lang == g).sum():,}): " + "/".join(f"{res[lbl][lang == g].mean():.4f}" for lbl in res)
            for g in range(len(LANGUAGES))))
        print(f"  (in the language line, the numbers are the arms in the order above; against A: paired difference, "
              f"standard error and 95% interval from 1,000 bootstrap draws of whole users; not allowed: videos "
              f"shown that the request's rules forbid)  [{time.time() - t0:.0f} s]")
