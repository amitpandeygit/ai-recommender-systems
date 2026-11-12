"""The honest test of Chapter 11: content for new videos inside Tesselo's pipeline (section 11.8).

Every arm is Chapter 8's production row, exactly as in sections 9.7 and 10.8: section
8.2's filters, at most 2 videos per creator (the new video included), the best "New
for you" video at slot 4 and one ad at slot 3, assembled at once, every source obeying
each request's own rules, and a final check counting any video shown that the request
did not allow. The arms differ only in how the slot-4 video is chosen:

  A: Chapter 8's row        - Chapter 8's source: the user's language first, then
                              Chapter 3's content similarity
  + graph "New for you"      - section 10.5's LightGCN + KG, the user's language first
                              (section 10.8's arm)
  + content "New for you"    - section 11.7's chosen content model, the user's language first
  + both "New for you"       - the mean of the two models' ranks among the new videos the
                              request may show (tied scores share a rank), the user's
                              language first

Each model is fitted on what had arrived by day 23, with the settings chosen on the
validation week in sections 10.5 and 11.7, from seed 0. Differences against A are paired
over the same requests, with standard errors and 95% intervals from 1,000 bootstrap
draws of whole users.

Needs Python 3.12 or later, PyTorch, pyroaring and text_embeddings.npz. Run from the
repository root (about 8 minutes):
    python chapter_11/content_pipeline.py
"""
import os
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "2"
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from content_data import MODELS, load, modalities, text_embeddings  # noqa: E402
from content_models import fuse, profiles  # noqa: E402
from cold_start import CHOSEN, embed, fit_aligner, fit_map, lightgcn_video, mapped  # noqa: E402
from graph_data import DAY, environment, exclude_mask, histories, ndcg, outcomes, paired, requests  # noqa: E402
from graph_pipeline import kg_new_scores  # noqa: E402
from cold_items import click_pairs, new_video_choice  # noqa: E402
from seq_data import eligible_kept, forbidden_shown, ranked_pools, training_sequences  # noqa: E402
from pipeline_test import assemble_rows, first_allowed, new_candidates  # noqa: E402
from slate_data import rules, split, top_creators  # noqa: E402
from ranking_data import LANGUAGES  # noqa: E402
from filtering import RuleSets  # noqa: E402
from honest_test import CREATOR_CAP, NEW_SLOT, AD_SLOT  # noqa: E402
from rankers import score_pools  # noqa: E402
from slates import ad_setup, best_ads, check_rows  # noqa: E402
from two_tower import guardrails  # noqa: E402


def ranks(s, bad):
    """Each score's rank among the request's allowed new videos, from 0 (lowest) to 1
    (highest), tied scores sharing the mean of their ranks, so a source whose scores are
    all equal gives every allowed video 0.5 and does not change the choice; a video the
    request may not show gets -1."""
    r = rankdata(np.where(bad, -np.inf, s), axis=1, method="average") - 1.0
    n_bad = bad.sum(axis=1, keepdims=True)
    r = (r - n_bad) / np.maximum((~bad).sum(axis=1, keepdims=True) - 1, 1)
    return np.where(bad, -1.0, r).astype(np.float32)


if __name__ == "__main__":
    t0 = time.time()
    print(environment())
    W = load()
    users, items = W["users"], W["items"]
    n = len(items["vec"])
    E, idx = text_embeddings(W, MODELS[-1])
    T = idx["titles"]
    text = E[T[np.arange(n), items["language"]]]
    X = fuse([(text, np.ones(n, dtype=bool))] + list(modalities(W).values()), [1.0] * 4)
    sp = split(W, "test")
    S = sp["S"]
    R = requests(W, 23, 30)
    H, iv = histories(W, R)
    bad = exclude_mask(W, S, R, H)
    rl = rules(W["cfg"], users, items)
    rs = RuleSets(rl, [np.flatnonzero(S["used"][u]) for u in range(len(users["vec"]))], n, "numpy")
    uu, inv = np.unique(R["user"], return_inverse=True)
    kept = eligible_kept(ranked_pools(sp, W, rs, uu), inv, bad)
    new = first_allowed(new_candidates(W, S, uu, rl)[inv], bad)
    cols = np.flatnonzero(W["video"] & ~S["ok"] & ~rl["removed"])
    bn = bad[:, cols]
    same = (users["language"][R["user"]][:, None] == items["language"][cols][None, :]).astype(np.float32) * 1_000.0
    kg = kg_new_scores(W, 23 * DAY, R, H, cols).astype(np.float32)   # language first already added
    if CHOSEN["method"] == "aligned with clicks":
        Z = embed(fit_aligner(X, click_pairs(training_sequences(W, 23 * DAY)[0]), CHOSEN["epochs"]), X)
        P, Vc = profiles(Z, H), Z[cols]
    else:                                                             # mapped to clicks
        U, V = lightgcn_video(W, 23 * DAY, R, H, 0, kg=False)
        est = np.flatnonzero(W["video"] & S["ok"])
        P, Vc = U, mapped(X[cols], fit_map(X[est], V[est], CHOSEN["penalty"]))
    new_of = {"8": new, "graph": new_video_choice(kg, bn, cols)}
    new_of["content"], new_of["both"] = np.full(len(new), -1), np.full(len(new), -1)
    for a in range(0, len(new), 4_096):                               # in batches, to save memory
        b = min(a + 4_096, len(new))
        content = (P[a:b] @ Vc.T).astype(np.float32) + same[a:b]
        new_of["content"][a:b] = new_video_choice(content, bn[a:b], cols)
        both = (ranks(kg[a:b] - same[a:b], bn[a:b]) + ranks(content - same[a:b], bn[a:b])) / 2 + same[a:b]
        new_of["both"][a:b] = new_video_choice(both, bn[a:b], cols)
    del kg, same
    new_of = {k: np.where(v >= 0, v, new) for k, v in new_of.items()}     # no choice: Chapter 8's
    ads = ad_setup(W, W["cfg"], rl)
    ad = best_ads(W, S, uu, ads)[0][:, 0][inv]
    print(f"Chapter 2's world; test week: {len(R['user']):,} requests, {len(cols):,} new videos; the content model: "
          f"{CHOSEN['method']} (section 11.7's choice)  [{time.time() - t0:.0f} s]")
    arms = {k: [] for k in new_of}
    for a in range(0, len(R["user"]), 2_048):
        b = min(a + 2_048, len(R["user"]))
        k50 = kept[a:b]
        hp = score_pools(sp["heavy"], S, users, items, R["user"][a:b], np.maximum(k50, 0))
        rel = np.where(k50 >= 0, np.nan_to_num(np.where(k50 >= 0, hp, np.nan), nan=-np.inf), -np.inf)
        for k, nv in new_of.items():
            arms[k].append(assemble_rows(k50, rel, nv[a:b], ad[a:b], items))
    labels = {"8": "A: Chapter 8's row", "graph": "+ graph New for you", "content": "+ content New for you",
              "both": "+ both New for you"}
    ok = S["ok"]
    tc = top_creators(W, S)
    new_ok = W["video"] & ~ok
    lang = users["language"][R["user"]]
    print(f"  {'':<24}{'NDCG@10':>9}{'coverage':>10}{'top-10% cr.':>12}{'online':>9}{'against A':>18}"
          f"{'95% interval':>20}{'new plays':>11}{'same language':>15}{'revenue':>9}{'rule breaks':>13}"
          f"{'not allowed':>13}")
    base, res = None, {}
    for key, label in labels.items():
        top = np.concatenate(arms[key])
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
        print(f"  {label:<24}{ndcg(R, top):>9.4f}{cov:>10.1%}{cr:>12.1%}{on.mean():>9.4f}{d:>18}{ci:>20}"
              f"{new_plays:>11.4f}{same_lang:>15.1%}{rev.mean():>9.4f}{breaks:>13}{shown_bad:>13}")
        res[label] = on
    for a_, b_ in (("+ content New for you", "+ graph New for you"), ("+ both New for you", "+ graph New for you"),
                   ("+ both New for you", "+ content New for you")):
        print(f"  {a_[2:]} against {b_[2:]} (paired, users resampled): " + "{:+.4f} ({:.4f}) [{:+.4f}, {:+.4f}]"
              .format(*paired(res[a_], res[b_], R["user"])))
    print("  online by app language: " + "  ".join(
        f"{LANGUAGES[g]} ({(lang == g).sum():,}): " + "/".join(f"{res[lbl][lang == g].mean():.4f}" for lbl in res)
        for g in range(len(LANGUAGES))))
    print("  (online: expected satisfied plays per row; new plays: the same from videos below the history threshold; "
          "same language: share of slot-4 videos in the user's app language; in the language line, the numbers are "
          "the arms in the order above; against A: paired difference, standard error and 95% interval from 1,000 "
          f"bootstrap draws of whole users)  [{time.time() - t0:.0f} s]")
