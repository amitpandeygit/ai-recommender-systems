"""The honest test of Chapter 9: a sequence model inside Tesselo's pipeline (section 9.7).

Every arm ends in Chapter 8's production row: section 8.2's filters, at most 2 videos
per creator (the new video included), the best "New for you" video at slot 4 and one
ad at slot 3, assembled at once (Chapter 8's assemble and check_rows). The arms
differ only in the organic videos:

  A: Chapter 8's row      - the pre-ranker's 50 from the pool, in the heavy ranker's
                            order, from the day-23 snapshot (one row per user all week)
  + sequence candidates    - the 50 plus the sequence model's top 50 for this request,
                            all in the heavy ranker's order
  + blended score          - the 50, in the order of a blend of the heavy ranker and the
                            sequence model
  + both                   - the 100, in the blend's order

The sequence model is the one section 9.4's validation week chose in each world
(seq_data.CHOICE), refitted on what had arrived by day 23, reading what had arrived
before each request. The blend is a
logistic regression on held-out impressions of the validation week (days 16 to 23,
with both models fitted on what had arrived by day 16), as in section 8.4: the
heavy ranker's logit, the sequence model's log probability, whether the visit has a
click yet, and the product of the last two, so the weight on the sequence model can
grow once the visit has shown its intent.

Needs Python 3.12 or later, PyTorch and pyroaring. Run from the repository root (about 10 minutes;
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
from seq_data import (CHOICE, DAY, WORLDS, exclude_mask, histories, load, ndcg, outcomes, paired,  # noqa: E402
                      requests, training_sequences)
from seq_models import SeqModel, fit, score  # noqa: E402
from slate_data import rules, split, top_creators  # noqa: E402
from ranking_data import LANGUAGES  # noqa: E402
from filtering import RuleSets  # noqa: E402
from honest_test import filtered_kept, CREATOR_CAP, NEW_SLOT, AD_SLOT  # noqa: E402
from hybrids import fit_logistic  # noqa: E402
from rankers import score_pools  # noqa: E402
from slates import ad_setup, assemble, best_ads, check_rows, new_for_you, score_order  # noqa: E402
from two_tower import guardrails  # noqa: E402

EXTRA = 50                                                      # sequence candidates added per request


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def log_softmax_rows(s, video):
    s = np.where(video[None, :], s, -np.inf)
    m = s.max(axis=1, keepdims=True)
    return s - m - np.log(np.exp(s - m).sum(axis=1, keepdims=True))


def features(heavy_p, seq_lp, in_visit):
    """Listing 9.7 (first part). The blend's inputs for each (request, candidate)."""
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


def fit_blend(W, n):
    """Both models fitted on what had arrived by day 16; the blend on the impressions
    of days 16 to 23."""
    sp = split(W, "validation")
    Rv = requests(W, 16, 23)
    Hv, ivv = histories(W, Rv)
    kind, loss, ep = CHOICE[W["name"]]
    m = fit(SeqModel(n, kind, W["items"]), training_sequences(W, 16 * DAY)[0], ep, loss)
    rows, cand, y = heldout_rows(W, Rv, Hv, ivv)
    video = ~W["items"]["is_product"]
    lp = np.empty(len(rows))
    for a in range(0, len(Rv["user"]), 2_048):
        sel = (rows >= a) & (rows < a + 2_048)
        s = log_softmax_rows(score(m, Hv[a:a + 2_048]), video)
        lp[sel] = s[rows[sel] - a, cand[sel]]
    hp = score_pools(sp["heavy"], sp["S"], W["users"], W["items"], Rv["user"][rows], cand[:, None])[:, 0]
    X = features(hp, lp, ivv[rows])
    return fit_logistic(X, y), len(y), int(y.sum())


if __name__ == "__main__":
    t0 = time.time()
    if not sys.argv[1:]:                                           # each world in its own process, to save memory
        for k in range(len(WORLDS)):
            subprocess.run([sys.executable, "-u", __file__, str(k)], check=True)
        sys.exit()
    worlds = [WORLDS[int(a)] for a in sys.argv[1:]]               # one world, by its number
    for name in worlds:
        W = load(name)
        users, items = W["users"], W["items"]
        n = len(items["vec"])
        video = ~items["is_product"]
        w, n_rows, n_pos = fit_blend(W, n)
        print(f"\n{name}: sequence model {', '.join(str(x) for x in CHOICE[name])} epochs; blend fitted on "
              f"{n_rows:,} validation impressions ({n_pos:,} satisfied): "
              f"{w[0]:+.3f} {w[1]:+.3f} x heavy logit {w[2]:+.3f} x sequence log prob {w[3]:+.3f} x visit has a "
              f"click {w[4]:+.3f} x both  [{time.time() - t0:.0f} s]")
        sp = split(W, "test")
        S = sp["S"]
        R = requests(W, 23, 30)
        H, iv = histories(W, R)
        bad = exclude_mask(W, S, R, H)
        kind, loss, ep = CHOICE[name]
        m = fit(SeqModel(n, kind, items), training_sequences(W, 23 * DAY)[0], ep, loss)
        rl = rules(W["cfg"], users, items)
        rs = RuleSets(rl, [np.flatnonzero(S["used"][u]) for u in range(len(users["vec"]))], n, "numpy")
        uu, inv = np.unique(R["user"], return_inverse=True)
        kept_u = filtered_kept(dict(sp, uu=uu), W, rs, "uu")
        kept = kept_u[inv]                                        # Chapter 8's 50, per request
        new_u, _ = new_for_you(W, S, uu, rl)
        ads = ad_setup(W, W["cfg"], rl)
        ad_u = best_ads(W, S, uu, ads)[0][:, 0]
        new, ad = new_u[inv], ad_u[inv]
        res = {}
        arms = {}
        for a in range(0, len(R["user"]), 2_048):
            b = min(a + 2_048, len(R["user"]))
            lp = log_softmax_rows(score(m, H[a:b]), video)
            s_ = np.where(bad[a:b], -np.inf, lp)                      # the sequence model's own top 50
            seq_top = np.argsort(-s_, axis=1, kind="stable")[:, :EXTRA]
            seq_top = np.where(np.isfinite(np.take_along_axis(s_, seq_top, axis=1)), seq_top, -1)
            k50 = kept[a:b]
            dup = np.array([np.isin(x, np.r_[y, z]) for x, y, z in zip(seq_top, k50, new[a:b])])  # no video twice
            both = np.concatenate([k50, np.where(dup, -1, seq_top)], axis=1)
            for cand_label, cands in (("50", k50), ("100", both)):
                hp = score_pools(sp["heavy"], S, users, items, R["user"][a:b], np.maximum(cands, 0))
                hp = np.where(cands >= 0, hp, np.nan)
                lpc = np.take_along_axis(lp, np.maximum(cands, 0), axis=1)
                bl = features(np.nan_to_num(hp, nan=0.5), lpc, iv[a:b, None]) @ w
                for score_label, sc in (("heavy", hp), ("blend", bl)):
                    rel = np.where(cands >= 0, np.nan_to_num(sc, nan=-np.inf), -np.inf)
                    top = assemble(cands, lambda allowed, above: score_order(cands, rel, allowed), items,
                                   CREATOR_CAP, [(new[a:b], NEW_SLOT), (ad[a:b], AD_SLOT)])
                    arms.setdefault((cand_label, score_label), []).append(top)
        labels = {("50", "heavy"): "A: Chapter 8's row", ("100", "heavy"): "+ sequence candidates",
                  ("50", "blend"): "+ blended score", ("100", "blend"): "+ both"}
        later = R["place"] >= 1
        ok = S["ok"]
        tc = top_creators(W, S)
        print(f"Test week: {len(R['user']):,} requests of consenting, logged-in users ({later.mean():.1%} later in "
              f"their visit)  [{time.time() - t0:.0f} s]")
        print(f"  {'':<26}{'NDCG@10':>9}{'coverage':>10}{'top-10% cr.':>12}{'online':>9}{'later':>9}"
              f"{'against A':>18}{'later, against A':>19}{'revenue':>9}{'rule breaks':>13}")
        base = None
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
            d = "" if on is base else "{:+.4f} ({:.4f})".format(*paired(on, base))
            dl = "" if on is base else "{:+.4f} ({:.4f})".format(*paired(on[later], base[later]))
            print(f"  {label:<26}{ndcg(R, top):>9.4f}{cov:>10.1%}{cr:>12.1%}{on.mean():>9.4f}{on[later].mean():>9.4f}"
                  f"{d:>18}{dl:>19}{rev.mean():>9.4f}{breaks:>13}")
            res[label] = on
        lang = users["language"][R["user"]]
        print("  online by app language: " + "  ".join(
            f"{LANGUAGES[g]} ({(lang == g).sum():,}): " + "/".join(f"{res[lbl][lang == g].mean():.4f}" for lbl in res)
            for g in range(len(LANGUAGES))))
        print(f"  (in the language line, the four numbers are the four arms in the order above)  "
              f"[{time.time() - t0:.0f} s]")
