"""The honest test of Chapter 8: complete rows against Chapter 7's shipped pipeline (section 8.8).

Every arm starts from Chapter 6's pool of 500 and ends in a row of 10 for every
consenting user on the test week:

  iALS, then popular in language - Chapter 5's best fixed baseline
  Chapter 7's shipped pipeline   - pool, distilled pre-ranker keeping 50, heavy ranker
  A: the production row          - section 8.2's filters on the pool, then Chapter 7's stages,
                                   at most 2 videos per creator, the best "New for you" video at
                                   slot 4 and one ad at slot 3
  (two steps of A on their own, to show what each part costs: the filters only, then
   the filters and the new-video slot)
  B: ads where they pay          - A, but the ad goes to the slot of section 8.7's allocation rule
                                   (any slot but the new video's, or nowhere), priced on the row it
                                   changes, with the exchange rate set so that the estimated ad
                                   revenue equals A's (estimates only, no outcomes)
  C: a blended score             - A, with feature-weighted linear stacking (section 8.4) in place
                                   of the heavy ranker's score
  D: a varied row                - C, with the row built by the method and setting that section
                                   8.6's validation rule chose
Each row is assembled at once (slates.assemble): the new video and the ad take
their slots first, then the organic videos fill the rest under the creator limit,
which counts the new video; check_rows() confirms every rule. The settings are
fixed before the run: the creator limit, the new-video slot and the ad slot are
Tesselo's product rules, and every other choice comes from the validation week.
Results are reported in Chapter 2's world and in section 8.6's
planted boredom test, with Chapter 5's guardrails, for every app language.

Needs Python 3.12 or later, PyTorch and pyroaring. Run from the repository root (about three minutes):
    python chapter_08/honest_test.py
"""
import os
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "2"              # NumPy's linear algebra: the same thread count on every run
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from slate_data import (KEPT, POOL, ROW, Ranker, paired, row_metrics, rules, slate_outcomes, split, train,  # noqa: E402
                        world)
from filtering import RuleSets  # noqa: E402
from hybrids import SECOND, blend_scores, fit_blends  # noqa: E402
from slates import (ad_allocation, ad_setup, assemble, best_ads, check_rows, choose_diversity,  # noqa: E402
                    look_curve, mmr, new_for_you, revenue, row_scores, score_order, whole_row)
from ranking_data import LANGUAGES, candidate_pools  # noqa: E402
from pre_ranking import student_scores  # noqa: E402
from rankers import score_pools  # noqa: E402

CREATOR_CAP = 2
NEW_SLOT = 4
AD_SLOT = 3


def filtered_kept(sp, W, rs, who):
    """Section 8.2's filters on the pool of 500, then the distilled pre-ranker keeps 50."""
    users, items, S = W["users"], W["items"], sp["S"]
    rows = sp[who]
    pools = candidate_pools(S, users, items, rows, POOL)
    for k, u in enumerate(rows):
        p = pools[k][pools[k] >= 0]
        p = rs.filter(u, p)
        pools[k] = -1
        pools[k, :len(p)] = p
    sc = np.where(pools >= 0, student_scores(sp["pre"], S, users, items, rows, pools), -np.inf)
    keep = np.argsort(-sc, axis=1, kind="stable")[:, :KEPT]
    return np.where(np.take_along_axis(sc, keep, axis=1) > -np.inf, np.take_along_axis(pools, keep, axis=1), -1)


def builder(kept, score, items, S, method=None, x=None):
    """The organic part of the row: a function of the creator rule (allowed) and of the
    reserved videos above each pick (above) that picks videos in slot order, by score
    order, MMR or the whole-row builder; the variety builders count the reserved videos."""
    rel = np.where(kept >= 0, score, -np.inf)
    if method == "MMR":
        return lambda allowed, above: mmr(kept, rel, S["Y"], x, allowed=allowed, above=above)
    if method == "whole row":
        return lambda allowed, above: whole_row(kept, rel, items["category"], x, allowed=allowed, above=above)
    return lambda allowed, above: score_order(kept, rel, allowed)


if __name__ == "__main__":
    t0 = time.time()
    W = world()
    cfg, users, items = W["cfg"], W["users"], W["items"]
    val, test = split(W, "validation"), split(W, "test")
    S = test["S"]
    R = rules(cfg, users, items)
    rs = RuleSets(R, [np.flatnonzero(S["used"][u]) for u in range(len(users["vec"]))], len(items["vec"]), "numpy")
    C_val = train(Ranker(val["meta"], SECOND["arch"]), val["b"], val["R"], epochs=SECOND["epochs"])
    C_test = train(Ranker(test["meta"], SECOND["arch"]), test["b"], test["R"], epochs=SECOND["epochs"])
    w, _ = fit_blends(W, val, C_val)
    method, setting = choose_diversity(W, val)
    look = look_curve(W)
    ads = ad_setup(W, cfg, R)
    print(f"Settings: creator limit {CREATOR_CAP}, new video at slot {NEW_SLOT}, ad at slot {AD_SLOT}; "
          f"validation choice for arm D: {method}, {setting}  [{time.time() - t0:.0f} s]")

    rows, checks = {}, {}
    for who in ("aud", "rows"):                              # all consenting users first: arm B's rate
        r = test[who]
        kept = filtered_kept(test, W, rs, who)
        heavy = score_pools(test["heavy"], S, users, items, r, kept)
        blend = 1 / (1 + np.exp(-blend_scores(w, "fwls", test["heavy"], C_test, test, users, items, kept, r)))
        # (the blend's P(satisfied play), always positive, as MMR and the whole-row builder need)
        new, _ = new_for_you(W, S, r, R)
        ad2, val2 = best_ads(W, S, r, ads)
        ad, ad_value = ad2[:, 0], val2[:, 0]
        new_res = (new, NEW_SLOT)
        organic = builder(kept, heavy, items, S)
        only = assemble(kept, organic, items, CREATOR_CAP)
        base = assemble(kept, organic, items, CREATOR_CAP, [new_res])
        A = assemble(kept, organic, items, CREATOR_CAP, [new_res, (ad, AD_SLOT)])
        s_row = row_scores(base, kept, np.where(kept >= 0, heavy, -np.inf))   # the row the ad would change
        fixed = (np.arange(1, ROW + 1)[None, :] == NEW_SLOT) & (new >= 0)[:, None]
        if who == "aud":                                     # exchange rate: estimated revenue equal to A's
            est_a = (look[AD_SLOT - 1] * ad_value).mean()
            grid = np.round(np.arange(0.10, 1.501, 0.005), 3)
            est = []
            for e in grid:
                slot = ad_allocation(s_row, fixed, look, ad_value, e)
                est.append(np.where(slot > 0, look[np.maximum(slot, 1) - 1] * ad_value, 0.0).mean())
            e_match = float(grid[int(np.argmin(np.abs(np.array(est) - est_a)))])
        slot_b = ad_allocation(s_row, fixed, look, ad_value, e_match)
        B = assemble(kept, organic, items, CREATOR_CAP, [new_res, (ad, slot_b)])
        Cb = assemble(kept, builder(kept, blend, items, S), items, CREATOR_CAP, [new_res, (ad, AD_SLOT)])
        Dv = assemble(kept, builder(kept, blend, items, S, method, setting), items, CREATOR_CAP,
                      [new_res, (ad, AD_SLOT)])
        checks[who] = {"A: the production row": check_rows(A, items, CREATOR_CAP, [new_res, (ad, AD_SLOT)]),
                       "B: ads where they pay": check_rows(B, items, CREATOR_CAP, [new_res, (ad, slot_b)]),
                       "C: a blended score": check_rows(Cb, items, CREATOR_CAP, [new_res, (ad, AD_SLOT)]),
                       "D: a varied row": check_rows(Dv, items, CREATOR_CAP, [new_res, (ad, AD_SLOT)])}
        pools = candidate_pools(S, users, items, r, POOL)
        rows[who] = {"iALS, then popular in language": pools[:, :ROW],
                     "Chapter 7's shipped pipeline": np.take_along_axis(
                         test[f"kept_{who}"], np.argsort(-np.where(test[f"kept_{who}"] >= 0, test[f"heavy_{who}"],
                                                                   -np.inf), axis=1, kind="stable")[:, :ROW], axis=1),
                     "filters only": only, "filters and the new-video slot": base,
                     "A: the production row": A, "B: ads where they pay": B,
                     "C: a blended score": Cb, "D: a varied row": Dv}
    print("Rule check on the rows of all consenting users (rows that break a rule, empty slots): "
          + "; ".join(f"{k[0]} {v[0]}, {v[1]}" for k, v in checks["aud"].items()))
    print(f"Exchange rate for arm B: {e_match} (estimated revenue per row equal to arm A's, "
          f"{(look[AD_SLOT - 1] * best_ads(W, S, test['aud'], ads)[1][:, 0]).mean():.4f} coins)  "
          f"[{time.time() - t0:.0f} s]")

    print(f"\nSection 8.8: test week; offline on {len(test['rows']):,} test users, the rest on {len(test['aud']):,} "
          f"consenting users")
    print(f"  {'':<32}{'NDCG@10':>8}{'coverage':>9}{'top-10% cr.':>12}{'variety':>8}{'online':>8}{'boredom':>9}"
          f"{'revenue':>9}{'new-video plays':>16}")
    res, ship = {}, "Chapter 7's shipped pipeline"
    new_ok = W["video"] & ~S["ok"]
    for name in rows["aud"]:
        m = row_metrics(rows["rows"][name], rows["aud"][name], test, W)
        _, play = slate_outcomes(rows["aud"][name], test["aud"], users, items, cfg)
        top = rows["aud"][name]
        new_plays = np.where((top >= 0) & new_ok[np.maximum(top, 0)], play, 0.0).sum(axis=1).mean()
        m["revenue"] = revenue(top, test["aud"], W, ads["bid"]).mean()
        res[name] = m
        print(f"  {name:<32}{m['NDCG@10']:>8.4f}{m['coverage']:>9.1%}{m['top-10% creators']:>12.1%}"
              f"{m['variety']:>8.3f}{m['online'].mean():>8.4f}{m['online, boredom'].mean():>9.4f}"
              f"{m['revenue']:>9.4f}{new_plays:>16.4f}")
    print("  (revenue: expected ad revenue per row, coins; new-video plays: expected satisfied plays per row from "
          "videos below the history threshold)")
    print(f"\n  paired differences against {ship} (standard error), all consenting users:")
    for name in list(rows["aud"])[2:]:
        d, se = paired(res[name]["online"], res[ship]["online"])
        db, seb = paired(res[name]["online, boredom"], res[ship]["online, boredom"])
        print(f"    {name:<30} online {d:+.4f} ({se:.4f})   boredom test {db:+.4f} ({seb:.4f})")
    for a, b in (("B: ads where they pay", "A: the production row"), ("C: a blended score", "A: the production row"),
                 ("D: a varied row", "C: a blended score")):
        d, se = paired(res[a]["online"], res[b]["online"])
        db, seb = paired(res[a]["online, boredom"], res[b]["online, boredom"])
        print(f"    {a[0] + ' against ' + b[0]:<30} online {d:+.4f} ({se:.4f})   boredom test {db:+.4f} ({seb:.4f})")

    print("\nOnline (expected satisfied plays per row) by app language, all consenting users")
    lang = users["language"][test["aud"]]
    print(f"  {'':<32}" + "".join(f"{LANGUAGES[k] + ' (' + str((lang == k).sum()) + ')':>13}"
                                  for k in range(len(LANGUAGES))))
    for name in rows["aud"]:
        print(f"  {name:<32}" + "".join(f"{res[name]['online'][lang == k].mean():>13.4f}" for k in range(len(LANGUAGES))))
    print(f"[{time.time() - t0:.0f} s]")
