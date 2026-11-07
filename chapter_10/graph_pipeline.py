"""The honest test of Chapter 10: a graph model inside Tesselo's pipeline (section 10.8).

Every arm ends in Chapter 8's production row, as in section 9.7: section 8.2's filters,
at most 2 videos per creator (the new video included), the best "New for you" video
at slot 4 and one ad at slot 3, assembled at once, with every source obeying each
request's own rules and a final check counting any video shown that the request did
not allow. The arms:

  A: Chapter 8's row       - the pre-ranker's first 50 allowed videos from the day-23
                             pool, in the heavy ranker's order
  + graph candidates        - the 50 plus the graph model's top 50 for this request, in
                             the heavy ranker's order
  + blended score           - the 50, in the order of a blend of the heavy ranker and
                             the graph model
  + both                    - the 100, in the blend's order
  + graph "New for you"     - A, with the slot-4 video chosen by section 10.5's
                             knowledge-graph model, the user's language first,
                             instead of Chapter 8's source
  control: no follows       - the blended score with LightGCN on the click graph alone

The graph model is section 10.6's LightGCN + follows: users, videos and creators, with
clicks, follows and authorship, each request's user vector built from what the user
had clicked and followed before it. Each blend is a logistic regression on held-out
impressions of the validation week (every model fitted on what had arrived by day
16), as in sections 8.4 and 9.7, on five inputs: a constant, the heavy ranker's
logit, the graph score, whether the user had clicked at most 2 different videos
before the request, and the graph score again for those users, so that its weight
can grow where the user's own clicks say little.

Differences against A are paired over the same requests, with standard errors and 95%
intervals from 1,000 bootstrap draws of whole users.

Needs Python 3.12 or later, PyTorch and pyroaring. Run from the repository root (about 17 minutes):
    python chapter_10/graph_pipeline.py
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
from graph_data import (DAY, click_graph, environment, exclude_mask, histories, load, ndcg, outcomes,  # noqa: E402
                        own_follows, paired, requests)
from graph_models import fit_lightgcn, history_nodes, layer_vectors, request_vectors  # noqa: E402
from social import FEW, gcn_scores, graphs, neighbours_now  # noqa: E402
from cold_items import LAYERS, VIDEO_RELATIONS, kg_edges, new_video_choice  # noqa: E402
from seq_data import eligible_kept, forbidden_shown, ranked_pools  # noqa: E402
from pipeline_test import assemble_rows, first_allowed, heldout_rows, logit, new_candidates  # noqa: E402
from slate_data import rules, split, top_creators  # noqa: E402
from ranking_data import LANGUAGES  # noqa: E402
from filtering import RuleSets  # noqa: E402
from honest_test import CREATOR_CAP, NEW_SLOT, AD_SLOT  # noqa: E402
from hybrids import fit_logistic  # noqa: E402
from rankers import score_pools  # noqa: E402
from slates import ad_setup, best_ads, check_rows  # noqa: E402
from two_tower import guardrails  # noqa: E402

EXTRA = 50                                     # graph candidates added per request
EPOCHS = {"LightGCN": 240, "LightGCN + follows": 260, "LightGCN + KG": 240}   # validation choices of sections 10.4 to 10.6


def features(heavy_p, g, few):
    """Listing 10.7 (first part). The blend's inputs for each (request, candidate)."""
    f = few.astype(float)
    return np.stack([np.ones_like(heavy_p), logit(heavy_p), g, f * np.ones_like(heavy_p), f * g], axis=-1)


def graph_scorers(W, cutoff, R, H, fol, cre, seed=0):
    """f(a, b) -> graph scores of every video for requests a to b, for the graph model
    and the no-follows control."""
    n_u, n_i = len(W["users"]["vec"]), len(W["items"]["vec"])
    vn = np.flatnonzero(W["video"]) + n_u
    out = {}
    for name, ((edges, n_nodes), soc) in graphs(W, cutoff).items():
        deg = np.bincount(np.r_[edges[0], edges[1]], minlength=n_nodes)
        m, A_hat = fit_lightgcn(edges, n_nodes, n_u, vn, layers=LAYERS, epochs=EPOCHS[name], seed=seed)
        out[name] = gcn_scores(m, A_hat, deg, neighbours_now(W, H, fol, cre, soc), R["user"], n_u, n_i)
    return out


def kg_new_scores(W, cutoff, R, H, cols, seed=0):
    """Section 10.5's LightGCN + KG with the user's language first: each request's
    scores for the new videos (cols)."""
    n_u, n_i = len(W["users"]["vec"]), len(W["items"]["vec"])
    u, i = click_graph(W, cutoff).nonzero()
    kg, n_ent = kg_edges(W, np.flatnonzero(W["video"]), VIDEO_RELATIONS, n_u + n_i)
    rows, cols_e = np.r_[u, kg[0]], np.r_[i + n_u, kg[1]]
    n_nodes = n_u + n_i + n_ent
    deg = np.bincount(np.r_[rows, cols_e], minlength=n_nodes)
    m, A_hat = fit_lightgcn((rows, cols_e), n_nodes, n_u, np.flatnonzero(W["video"]) + n_u, layers=LAYERS,
                            epochs=EPOCHS["LightGCN + KG"], seed=seed)
    L, final = layer_vectors(m, A_hat)
    own = np.where((deg[R["user"]] > 0)[:, None], L[0][R["user"]], 0.0)
    same = W["users"]["language"][R["user"]][:, None] == W["items"]["language"][cols][None, :]
    return request_vectors(L, deg, history_nodes(H, n_u), own) @ final[n_u + cols].T + 1_000.0 * same


def fit_blends(W):
    """Listing 10.7 (second part). Every model fitted on what had arrived by day 16; one
    blend per graph model, fitted on the impressions of days 16 to 23."""
    sp = split(W, "validation")
    Rv = requests(W, 16, 23)
    Hv, ivv = histories(W, Rv)
    fv, cv = own_follows(W, Rv)
    few = np.array([len(np.unique(h[h >= 0])) <= FEW for h in Hv])
    rows, cand, y = heldout_rows(W, Rv, Hv, ivv)
    hp = score_pools(sp["heavy"], sp["S"], W["users"], W["items"], Rv["user"][rows], cand[:, None])[:, 0]
    weights = {}
    for name, f in graph_scorers(W, 16 * DAY, Rv, Hv, fv, cv).items():
        g = np.empty(len(rows))
        for a in range(0, len(Rv["user"]), 2_048):
            sel = (rows >= a) & (rows < a + 2_048)
            s = f(a, min(a + 2_048, len(Rv["user"])))
            g[sel] = s[rows[sel] - a, cand[sel]]
        weights[name] = fit_logistic(features(hp, g, few[rows]), y)
    return weights, len(y), int(y.sum())


if __name__ == "__main__":
    t0 = time.time()
    print(environment())
    W = load()
    users, items = W["users"], W["items"]
    n = len(items["vec"])
    wts, n_rows, n_pos = fit_blends(W)
    print(f"Chapter 2's world: blends fitted on {n_rows:,} validation impressions ({n_pos:,} satisfied)  "
          f"[{time.time() - t0:.0f} s]")
    for k, w in wts.items():
        print(f"  blend with {k + ':':<20} {w[0]:+.3f} {w[1]:+.3f} x heavy logit {w[2]:+.3f} x graph score "
              f"{w[3]:+.3f} x few clicks {w[4]:+.3f} x graph score for few clicks")
    sp = split(W, "test")
    S = sp["S"]
    R = requests(W, 23, 30)
    H, iv = histories(W, R)
    bad = exclude_mask(W, S, R, H)
    fol, cre = own_follows(W, R)
    few = np.array([len(np.unique(h[h >= 0])) <= FEW for h in H])
    scorers = graph_scorers(W, 23 * DAY, R, H, fol, cre)
    rl = rules(W["cfg"], users, items)
    rs = RuleSets(rl, [np.flatnonzero(S["used"][u]) for u in range(len(users["vec"]))], n, "numpy")
    uu, inv = np.unique(R["user"], return_inverse=True)
    ranked = ranked_pools(sp, W, rs, uu)
    kept = eligible_kept(ranked, inv, bad)
    new_k = new_candidates(W, S, uu, rl)[inv]
    new = first_allowed(new_k, bad)
    cols = np.flatnonzero(W["video"] & ~S["ok"] & ~rl["removed"])
    new_kg = new_video_choice(kg_new_scores(W, 23 * DAY, R, H, cols), bad[:, cols], cols)
    ads = ad_setup(W, W["cfg"], rl)
    ad = best_ads(W, S, uu, ads)[0][:, 0][inv]
    print(f"Test week: {len(R['user']):,} requests; {few.mean():.1%} from users with at most {FEW} clicked videos; "
          f"the knowledge-graph model fills slot 4 for {np.mean(new_kg >= 0):.1%}, Chapter 8's source for "
          f"{np.mean(new >= 0):.1%}  [{time.time() - t0:.0f} s]")
    arms = {}
    graph = "LightGCN + follows"
    for a in range(0, len(R["user"]), 2_048):
        b = min(a + 2_048, len(R["user"]))
        sec = {k: f(a, b) for k, f in scorers.items()}
        s_ = np.where(bad[a:b], -np.inf, sec[graph])                  # the graph model's own top 50
        g_top = np.argsort(-s_, axis=1, kind="stable")[:, :EXTRA]
        g_top = np.where(np.isfinite(np.take_along_axis(s_, g_top, axis=1)), g_top, -1)
        k50 = kept[a:b]
        dup = np.array([np.isin(x, np.r_[y, z]) for x, y, z in zip(g_top, k50, new[a:b])])   # no video twice
        both = np.concatenate([k50, np.where(dup, -1, g_top)], axis=1)
        for cand_label, cands in (("50", k50), ("100", both)):
            hp = score_pools(sp["heavy"], S, users, items, R["user"][a:b], np.maximum(cands, 0))
            hp = np.where(cands >= 0, hp, np.nan)
            orders = {"heavy": hp}
            for k in (scorers if cand_label == "50" else (graph,)):
                g = np.where(cands >= 0, np.take_along_axis(sec[k], np.maximum(cands, 0), axis=1), 0.0)
                orders[k] = features(np.nan_to_num(hp, nan=0.5), g, few[a:b, None]) @ wts[k]
            for score_label, sc in orders.items():
                rel = np.where(cands >= 0, np.nan_to_num(sc, nan=-np.inf), -np.inf)
                arms.setdefault((cand_label, score_label, "8"), []).append(
                    assemble_rows(cands, rel, new[a:b], ad[a:b], items))
                if cand_label == "50" and score_label == "heavy":
                    nk = np.where(new_kg[a:b] >= 0, new_kg[a:b], new[a:b])      # no KG choice: Chapter 8's
                    arms.setdefault(("50", "heavy", "kg"), []).append(assemble_rows(cands, rel, nk, ad[a:b], items))
    labels = {("50", "heavy", "8"): "A: Chapter 8's row", ("100", "heavy", "8"): "+ graph candidates",
              ("50", graph, "8"): "+ blended score", ("100", graph, "8"): "+ both",
              ("50", "heavy", "kg"): "+ graph New for you", ("50", "LightGCN", "8"): "control: no follows"}
    later = R["place"] >= 1
    ok = S["ok"]
    tc = top_creators(W, S)
    new_ok = W["video"] & ~ok
    print(f"  {'':<24}{'NDCG@10':>9}{'coverage':>10}{'top-10% cr.':>12}{'online':>9}{'few clicks':>12}"
          f"{'against A':>18}{'95% interval':>20}{'few, against A':>19}{'new plays':>11}{'revenue':>9}"
          f"{'rule breaks':>13}{'not allowed':>13}")
    base, res = None, {}
    for key, label in labels.items():
        top = np.concatenate(arms[key])
        slot_new = np.where(key[2] == "kg", np.where(new_kg >= 0, new_kg, new), new)
        click, play = outcomes(W, R, top)
        on = play.sum(axis=1)
        base = on if base is None else base
        vids = np.where((top >= 0) & ~items["is_product"][np.maximum(top, 0)], top, -1)
        cov, cr = guardrails(vids, ok, items["creator"], tc, n)
        i = np.maximum(top, 0)
        rev = np.where((top >= 0) & items["is_product"][i], click * ads["bid"][i], 0.0).sum(axis=1)
        new_plays = np.where((top >= 0) & new_ok[i], play, 0.0).sum(axis=1).mean()
        breaks = check_rows(top, items, CREATOR_CAP, [(slot_new, NEW_SLOT), (ad, AD_SLOT)])[0]
        shown_bad = forbidden_shown(top, bad, items)[0]
        if on is base:
            d = ci = df = ""
        else:
            m_, se, lo, hi = paired(on, base, R["user"])
            d, ci = f"{m_:+.4f} ({se:.4f})", f"[{lo:+.4f}, {hi:+.4f}]"
            df = "{:+.4f} ({:.4f})".format(*paired(on[few], base[few], R["user"][few])[:2])
        print(f"  {label:<24}{ndcg(R, top):>9.4f}{cov:>10.1%}{cr:>12.1%}{on.mean():>9.4f}{on[few].mean():>12.4f}"
              f"{d:>18}{ci:>20}{df:>19}{new_plays:>11.4f}{rev.mean():>9.4f}{breaks:>13}{shown_bad:>13}")
        res[label] = on
    blend, ctrl = res["+ blended score"], res["control: no follows"]
    print("  The blend against the no-follows control (paired, users resampled): all requests "
          "{:+.4f} ({:.4f}) [{:+.4f}, {:+.4f}]; ".format(*paired(blend, ctrl, R["user"]))
          + "few clicks {:+.4f} ({:.4f}) [{:+.4f}, {:+.4f}]".format(*paired(blend[few], ctrl[few], R["user"][few])))
    lang = users["language"][R["user"]]
    print("  online by app language: " + "  ".join(
        f"{LANGUAGES[g]} ({(lang == g).sum():,}): " + "/".join(f"{res[lbl][lang == g].mean():.4f}" for lbl in res)
        for g in range(len(LANGUAGES))))
    print(f"  (few clicks: requests from users who had clicked at most {FEW} different videos; new plays: expected "
          f"satisfied plays per row from videos below the history threshold; in the language line, the numbers are "
          f"the arms in the order above; against A: paired difference, standard error and 95% interval from 1,000 "
          f"bootstrap draws of whole users)  [{time.time() - t0:.0f} s]")
