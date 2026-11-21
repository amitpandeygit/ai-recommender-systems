"""The honest test of Chapter 17: cleaned listings inside Tesselo's pipeline (section 17.8).

Every arm is Chapter 11's production row: Chapter 8's row (section 8.2's filters, at most
2 videos per creator, one ad at slot 3), with the "New for you" video at slot 4 chosen by
section 11.7's content map with the user's language first, every source obeying each
request's own rules, and a final check that counts any video shown that the request did
not allow. The arms differ only in the records and texts of the 1,724 new videos, which
the content map and Chapter 8's fallback source read (new_items.ARMS):

  as written        - the exact dictionary's record, and the listing as written for the encoder
  fuzzy dictionary  - the fuzzy dictionary's record
  dictionary + LLM  - the fuzzy dictionary with the language model for the words it could
                      not read (enrich.hybrid, enrich.WORKHORSE)
  true record       - the real record, a reference no platform has

The map is fitted on what had arrived by day 23 (LightGCN from seed 0, penalty 10, as in
Chapter 11). Differences are paired against "as written", with standard errors and 95%
intervals from 1,000 bootstrap draws of whole users.

Needs llm_cache and text_vectors.npz. Run from the repository root (about five minutes):
    python chapter_17/llm_pipeline.py
"""
import os
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "2"
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from new_items import ARMS, arm_records, arm_world, text_block  # noqa: E402
from listings import listings, new_uploads  # noqa: E402
from content_data import load, modalities  # noqa: E402
from content_models import fuse  # noqa: E402
from cold_start import CHOSEN, fit_map, lightgcn_video, mapped  # noqa: E402
from graph_data import environment, exclude_mask, histories, ndcg, outcomes, paired, requests  # noqa: E402
from cold_items import new_video_choice  # noqa: E402
from seq_data import eligible_kept, forbidden_shown, ranked_pools  # noqa: E402
from pipeline_test import assemble_rows, first_allowed, new_candidates  # noqa: E402
from slate_data import rules, split, top_creators  # noqa: E402
from ranking_data import LANGUAGES  # noqa: E402
from filtering import RuleSets  # noqa: E402
from honest_test import CREATOR_CAP, NEW_SLOT, AD_SLOT  # noqa: E402
from rankers import score_pools  # noqa: E402
from slates import ad_setup, best_ads, check_rows  # noqa: E402
from two_tower import guardrails  # noqa: E402
import enrich  # noqa: E402
import textvec  # noqa: E402

if __name__ == "__main__":
    t0 = time.time()
    print(environment())
    W = load()
    users, items = W["users"], W["items"]
    n = len(items["vec"])
    L = listings(W)
    arms = arm_records(W, L, sorted(L))
    textvec.vectors(sorted({t for _, texts in arms.values() for t in texts.values()}))
    new_ids = set(new_uploads(W, 23).tolist()) - set(np.flatnonzero(items["is_product"]).tolist())
    sp = split(W, "test")
    S = sp["S"]
    R = requests(W, 23, 30)
    H, iv = histories(W, R)
    bad = exclude_mask(W, S, R, H)
    rl = rules(W["cfg"], users, items)
    rs = RuleSets(rl, [np.flatnonzero(S["used"][u]) for u in range(len(users["vec"]))], n, "numpy")
    uu, inv = np.unique(R["user"], return_inverse=True)
    kept = eligible_kept(ranked_pools(sp, W, rs, uu), inv, bad)
    cols = np.flatnonzero(W["video"] & ~S["ok"] & ~rl["removed"])
    bn = bad[:, cols]
    same = (users["language"][R["user"]][:, None] == items["language"][cols][None, :]).astype(np.float32) * 1_000.0
    U, V = lightgcn_video(W, 23 * 86_400, R, H, 0, kg=False)
    est = np.flatnonzero(W["video"] & S["ok"])
    mods = list(modalities(W).values())
    new_of = {}
    for arm in ARMS:
        recs, texts = arms[arm]
        Wa = arm_world(W, {i: r for i, r in recs.items() if i in new_ids})
        fallback = first_allowed(new_candidates(Wa, S, uu, rl)[inv], bad)
        T = text_block(W, {i: t for i, t in texts.items() if i in new_ids})
        X = fuse([(T, np.ones(n, dtype=bool))] + mods, [1.0] * 4)
        Vc = mapped(X[cols], fit_map(X[est], V[est], CHOSEN["penalty"]))
        choice = np.full(len(fallback), -1)
        for a in range(0, len(fallback), 4_096):
            b = min(a + 4_096, len(fallback))
            choice[a:b] = new_video_choice((U[a:b] @ Vc.T).astype(np.float32) + same[a:b], bn[a:b], cols)
        new_of[arm] = np.where(choice >= 0, choice, fallback)
        print(f"  {arm}: slot-4 choices made  [{time.time() - t0:.0f} s]", flush=True)
    ads = ad_setup(W, W["cfg"], rl)
    ad = best_ads(W, S, uu, ads)[0][:, 0][inv]
    print(f"Chapter 2's world; test week: {len(R['user']):,} requests, {len(cols):,} new videos with messy listings; "
          f"the cleaning model: {enrich.WORKHORSE}  [{time.time() - t0:.0f} s]")
    rows = {k: [] for k in new_of}
    for a in range(0, len(R["user"]), 2_048):
        b = min(a + 2_048, len(R["user"]))
        k50 = kept[a:b]
        hp = score_pools(sp["heavy"], S, users, items, R["user"][a:b], np.maximum(k50, 0))
        rel = np.where(k50 >= 0, np.nan_to_num(np.where(k50 >= 0, hp, np.nan), nan=-np.inf), -np.inf)
        for k, nv in new_of.items():
            rows[k].append(assemble_rows(k50, rel, nv[a:b], ad[a:b], items))
    ok = S["ok"]
    tc = top_creators(W, S)
    new_ok = W["video"] & ~ok
    lang = users["language"][R["user"]]
    print(f"  {'':<20}{'NDCG@10':>9}{'coverage':>10}{'top-10% cr.':>12}{'online':>9}{'against as written':>20}"
          f"{'95% interval':>20}{'new plays':>11}{'same language':>15}{'revenue':>9}{'rule breaks':>13}"
          f"{'not allowed':>13}")
    base, res = None, {}
    for key in ARMS:
        top = np.concatenate(rows[key])
        click, play = outcomes(W, R, top)
        on = play.sum(axis=1)
        base = on if base is None else base
        vids = np.where((top >= 0) & ~items["is_product"][np.maximum(top, 0)], top, -1)
        cov, cr = guardrails(vids, ok, items["creator"], tc, n)
        i = np.maximum(top, 0)
        rev = np.where((top >= 0) & items["is_product"][i], click * ads["bid"][i], 0.0).sum(axis=1)
        new_plays = np.where((top >= 0) & new_ok[i], play, 0.0).sum(axis=1).mean()
        slot = new_of[key]
        same_lang = np.mean(items["language"][slot[slot >= 0]] == lang[slot >= 0])
        breaks = check_rows(top, items, CREATOR_CAP, [(slot, NEW_SLOT), (ad, AD_SLOT)])[0]
        shown_bad = forbidden_shown(top, bad, items)[0]
        if on is base:
            d = ci = ""
        else:
            m_, se, lo, hi = paired(on, base, R["user"])
            d, ci = f"{m_:+.4f} ({se:.4f})", f"[{lo:+.4f}, {hi:+.4f}]"
        print(f"  {key:<20}{ndcg(R, top):>9.4f}{cov:>10.1%}{cr:>12.1%}{on.mean():>9.4f}{d:>20}{ci:>20}"
              f"{new_plays:>11.4f}{same_lang:>15.1%}{rev.mean():>9.4f}{breaks:>13}{shown_bad:>13}")
        res[key] = on
    print("  dictionary + LLM against fuzzy dictionary (paired, users resampled): {:+.4f} ({:.4f}) [{:+.4f}, {:+.4f}]"
          .format(*paired(res["dictionary + LLM"], res["fuzzy dictionary"], R["user"])))
    print("  online by app language: " + "  ".join(
        f"{LANGUAGES[g]} ({(lang == g).sum():,}): " + "/".join(f"{res[k][lang == g].mean():.4f}" for k in ARMS)
        for g in range(len(LANGUAGES))))
    print("  (online: expected satisfied plays per row; new plays: the same from videos below the history threshold; "
          "same language: share of slot-4 videos in the user's app language; coverage and top-10% creators' share of "
          "video exposure are Chapter 5's guardrails; the language line lists the arms in the order above)  "
          f"[{time.time() - t0:.0f} s]")
