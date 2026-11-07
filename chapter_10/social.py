"""Follows for users with few clicks: social and multi-relation graphs (section 10.6).

Many test requests come from users who have clicked almost nothing yet, but who
follow other users and creators. This script asks whether the follow graph helps
them. Every method answers each test request (days 23 to 30) with a row of 10
videos, from what the server had received before the request, follows included.
Results are given for the requests whose user had clicked at most 2 different
videos by then ("few clicks"), and for all requests.

  Chapter 8's ranker   - the reference, as in Chapter 9
  followees' clicks    - the videos that the accounts the user follows had clicked
                         before the request, counted
  followed creators    - the videos of the creators the user follows, most clicked first
  LightGCN             - the click graph only, as in section 10.4
  LightGCN + creators  - the control: the click graph plus creator nodes and authorship
                         edges (creator to video), with no follows
  LightGCN + follows   - Listing 10.6: one graph with users, videos and creators, and
                         three kinds of edges: clicks, follows (user to user and user
                         to creator) and authorship (creator to video). A user's vector
                         at the request is folded in from the videos they clicked, the
                         accounts and creators they follow and the accounts that follow
                         them; the two LightGCN + rows differ only in the follows

Privacy rules (graph_data.social_edges, own_follows and own_followers, checked on
every shared graph by graph_data.check_shared_graph): only follows made before the
moment are used (in this simulation a follow reaches the server the moment it is
made); a follow edge enters the shared graph only if both accounts are consenting and
neither is private; a private user's own follows are used for that user alone; a
followed account's clicks are shown as that account's clicks (followees' clicks) only
if it is consenting and not private; and the clicks of every consenting user, private
or not, train the shared models pooled with everyone else's, as in every chapter.

LightGCN uses the layers chosen in next_video.py; the epochs of the graph models are
chosen on the validation week by NDCG@10 over all its requests, Chapter 5's rule: each
graph is checked every 20 epochs, up to 600, and stops after 5 checks in a row without
a new best; the best check is kept.

Needs Python 3.12 or later and PyTorch. Run from the repository root (about 2 hours 15 minutes):
    python chapter_10/social.py
"""
import os
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "2"
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from graph_data import (DAY, chapter8_rows, check_shared_graph, click_graph, environment, known, load,  # noqa: E402
                        ndcg, own_followers, own_follows, paired, phase, row_line, social_edges, top10, usable)
from graph_models import check_fold_in, fit_lightgcn, graph_degrees, layer_vectors, request_vectors  # noqa: E402
import baselines as B  # noqa: E402
from slate_data import split  # noqa: E402

LAYERS = 3                       # next_video.py's validation choice for LightGCN
EPOCHS, EVERY = 600, 20          # at most 600 epochs, checked every 20 on the validation week
PATIENCE = 5                     # stop after 5 checks in a row without a new best
SEEDS = (0, 1, 2)
FEW = 2                          # "few clicks": at most this many different videos clicked before the request


def followee_clicks(W, R, followees):
    """Requests x items: how many of the accounts the user follows had clicked each
    video before the request (clicks the server had received and could credit)."""
    C = W["clicks"]
    starts = np.searchsorted(C["user"], np.arange(len(W["users"]["vec"]) + 1))
    rows, cols = [], []
    for r, (t, fs) in enumerate(zip(R["time"], followees)):
        for f in fs:
            a, b = starts[f], starts[f + 1]
            ok = known({k: C[k][a:b] for k in ("known", "until", "t")}, t)
            items = np.unique(C["item"][a:b][ok])
            rows.append(np.full(len(items), r))
            cols.append(items)
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    return sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(R["user"]), len(W["items"]["vec"])))


def creator_videos(W, creators, pop):
    """Requests x items: the videos of the creators the user follows, scored by clicks."""
    items = W["items"]
    rows, cols = [], []
    for r, cs in enumerate(creators):
        v = np.flatnonzero(np.isin(items["creator"], cs) & W["video"])
        rows.append(np.full(len(v), r))
        cols.append(v)
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    return sparse.csr_matrix((1.0 + pop[cols] / (pop.max() + 1.0), (rows, cols)),
                             shape=(len(creators), len(items["vec"])))


def social_graph(W, cutoff, follows=True):
    """Listing 10.6. Nodes: users, then items, then creators. Edges: clicks, follows
    usable in a shared graph at the cutoff (user to user, user to creator) and
    authorship (creator to video). follows=False: the same graph without follows."""
    users, items = W["users"], W["items"]
    n_u, n_i = len(users["vec"]), len(items["vec"])
    u, i = click_graph(W, cutoff).nonzero()
    (fu, fv), (cu, cc) = social_edges(W, cutoff)
    if not follows:
        fu, fv, cu, cc = fu[:0], fv[:0], cu[:0], cc[:0]
    vids = np.flatnonzero(W["video"])
    rows = np.r_[u, fu, cu, vids + n_u]
    cols = np.r_[i + n_u, fv, cc + n_u + n_i, items["creator"][vids] + n_u + n_i]
    return (rows, cols), n_u + n_i + W["cfg"].num_creators


def neighbours_now(W, H, links, social):
    """Each request's nodes: the videos in its history and, with social=True, its
    follow links (links = (followees, creators followed, followers), each counted once)."""
    n_u, n_i = len(W["users"]["vec"]), len(W["items"]["vec"])
    out = []
    for k, h in enumerate(H):
        nb = np.unique(h[h >= 0]) + n_u
        if social:
            fol, cre, fwr = (x[k] for x in links)
            nb = np.unique(np.r_[nb, fol, fwr, np.asarray(cre, dtype=np.int64) + n_u + n_i])
        out.append(nb.astype(np.int64))
    return out


def gcn_scores(model, A_hat, deg, nb, users, n_u, n_i):
    L, final = layer_vectors(model, A_hat)
    own = np.where((deg[users] > 0)[:, None], L[0][users], 0.0)
    U = request_vectors(L, deg, nb, own)
    item_vec = final[n_u:n_u + n_i]
    return lambda a, b: U[a:b] @ item_vec.T


def graphs(W, cutoff):
    """The three graph models: (edges, number of nodes) and whether follows are in them."""
    n_u, n_i = len(W["users"]["vec"]), len(W["items"]["vec"])
    u, i = click_graph(W, cutoff).nonzero()
    plain = ((u, i + n_u), n_u + n_i)
    return {"LightGCN": (plain, False), "LightGCN + creators": (social_graph(W, cutoff, follows=False), False),
            "LightGCN + follows": (social_graph(W, cutoff), True)}


if __name__ == "__main__":
    t0 = time.time()
    print(environment())
    W = load()
    users = W["users"]
    n_u, n_i = len(users["vec"]), len(W["items"]["vec"])
    vn = np.flatnonzero(W["video"]) + n_u
    g = W["graph"]
    print(f"Chapter 2's world  [{time.time() - t0:.0f} s]")
    (fu, _), (cu, _) = social_edges(W, 23 * DAY)
    uf = g["user_follows"]
    print(f"Follows made by day 23: {(uf['t'] < 23 * DAY).sum():,} user to user, "
          f"{(g['creator_follows']['t'] < 23 * DAY).sum():,} user to creator; usable in the shared graph "
          f"(consenting, not private): {len(fu):,} and {len(cu):,}")
    n_priv, u_priv = check_shared_graph(W, graphs(W, 23 * DAY)["LightGCN + follows"][0][0])
    print(f"Shared-graph check at day 23 passed: every follow edge joins consenting, non-private accounts. The click "
          f"edges include {n_priv:,} from {u_priv:,} private accounts, pooled like every consenting user's clicks")

    # ---- validation week: choose the epochs
    Rv, Hv, _, badv = phase(W, 16, W["snaps"][16])
    links_v = (*own_follows(W, Rv), own_followers(W, Rv))
    pop16 = B.popularity(W, 16 * DAY)
    ep = {}
    for name, ((edges, n_nodes), soc) in graphs(W, 16 * DAY).items():
        check_shared_graph(W, edges)
        deg = graph_degrees(*edges, n_nodes)
        nb = neighbours_now(W, Hv, links_v, soc)
        curve = []

        def check(e, m, Ah, curve=curve, deg=deg, nb=nb):
            """Record validation NDCG@10; True (stop) after PATIENCE checks without a new best."""
            curve.append((e, ndcg(Rv, top10(gcn_scores(m, Ah, deg, nb, Rv["user"], n_u, n_i), badv, tie=pop16))))
            best_at = max(range(len(curve)), key=lambda k: round(curve[k][1], 9))
            return len(curve) - 1 - best_at >= PATIENCE
        fit_lightgcn(edges, n_nodes, n_u, vn, layers=LAYERS, epochs=EPOCHS, every=EVERY, after=check)
        best = max(curve, key=lambda x: round(x[1], 9))
        ep[name] = best[0]
        print(f"  {name:<20}" + " ".join(f"{e}:{x:.4f}" for e, x in curve) + f"   chosen: {best[0]}"
              f"  [{time.time() - t0:.0f} s]")
    del Rv, Hv, badv

    # ---- test week
    sp = split(W, "test")
    R, H, _, bad = phase(W, 23, sp["S"])
    fol, cre = own_follows(W, R)
    links = (fol, cre, own_followers(W, R))
    assert not any(np.any(g["private"][f]) for f in fol)              # no private account's clicks are shown
    n_hist = np.array([len(np.unique(h[h >= 0])) for h in H])
    few = np.flatnonzero(n_hist <= FEW)
    pop = B.popularity(W, 23 * DAY)
    print(f"\nTest week: {len(R['user']):,} requests; {len(few):,} ({len(few) / len(R['user']):.1%}) from users who "
          f"had clicked at most {FEW} different videos, {np.mean(n_hist == 0):.1%} none. Follows at the request: "
          f"mean {np.mean([len(f) for f in fol]):.1f} usable accounts, {np.mean([len(c) for c in cre]):.1f} creators "
          f"(few clicks: {np.mean([len(fol[k]) for k in few]):.1f} and {np.mean([len(cre[k]) for k in few]):.1f}); "
          f"followed by {np.mean([len(x) for x in links[2]]):.1f} usable accounts  [{time.time() - t0:.0f} s]")
    ch8, _, _ = chapter8_rows(W, sp, R, bad)
    FC = followee_clicks(W, R, fol)
    CV = creator_videos(W, cre, pop)
    rows = {"followees' clicks": top10(lambda a, b: FC[a:b].toarray(), bad, tie=pop),
            "followed creators": top10(lambda a, b: CV[a:b].toarray(), bad, tie=pop)}
    errs = []
    for name, ((edges, n_nodes), soc) in graphs(W, 23 * DAY).items():
        check_shared_graph(W, edges)
        deg = graph_degrees(*edges, n_nodes)
        nb = neighbours_now(W, H, links, soc)
        rows[name] = []
        for s in SEEDS:
            m, A_hat = fit_lightgcn(edges, n_nodes, n_u, vn, layers=LAYERS, epochs=ep[name], seed=s)
            errs.append(check_fold_in(m, A_hat, edges, n_nodes, np.arange(n_u)))
            rows[name].append(top10(gcn_scores(m, A_hat, deg, nb, R["user"], n_u, n_i), bad, tie=pop))
        print(f"  {name} fitted, {ep[name]} epochs  [{time.time() - t0:.0f} s]")
    print(f"  Fold-in check: with the graph unchanged, request-time fold-in rebuilds every user's trained vector "
          f"(largest difference over the three graphs and seeds {max(errs):.1e})")
    means = {}
    for title, sub in (("Requests from users with few clicks", few), ("All requests", None)):
        print(f"\n{title}")
        print(f"  {'':<36}{'epochs':>10}{'NDCG@10':>9}{'online':>9}{'later':>9}{'against Chapter 8':>18}")
        base = row_line("Chapter 8's ranker", W, R, ch8, None, rows=sub)
        for name, top in rows.items():
            if isinstance(top, list):
                ons = [row_line(f"{name}, seed {s}", W, R, t, base, extra=str(ep[name]), rows=sub)
                       for s, t in zip(SEEDS, top)]
                idx = np.arange(len(R["user"])) if sub is None else sub
                on = np.mean(ons, axis=0)[idx]
                d, se = paired(on, base[idx], R["user"][idx])[:2]
                later = R["place"][idx] >= 1
                nd = np.mean([ndcg({k: (v[idx] if isinstance(v, np.ndarray) else [v[r] for r in idx])
                                    for k, v in R.items()}, t[idx]) for t in top])
                print(f"  {name + ', mean of 3 seeds':<36}{ep[name]:>10}{nd:>9.4f}{on.mean():>9.4f}"
                      f"{on[later].mean():>9.4f}{f'{d:+.4f} ({se:.4f})':>18}")
                means[(title, name)] = (on, idx)
            else:
                row_line(name, W, R, top, base, rows=sub)
        (f_on, idx), (c_on, _), (p_on, _) = (means[(title, k)] for k in
                                             ("LightGCN + follows", "LightGCN + creators", "LightGCN"))
        print("  LightGCN + follows against LightGCN + creators (only the follows differ): {:+.4f} ({:.4f}) "
              "[{:+.4f}, {:+.4f}]".format(*paired(f_on, c_on, R["user"][idx])) +
              "; LightGCN + creators against LightGCN: {:+.4f} ({:.4f}) [{:+.4f}, {:+.4f}]".format(
                  *paired(c_on, p_on, R["user"][idx])))
    print("  (online: expected satisfied plays per row; later: requests after the first of their visit; against: "
          "paired difference over the same requests, standard error from 1,000 bootstrap draws of whole users; "
          "in brackets, the 95% interval)")
    print(f"  Privacy: follow edges left out of the shared graph at day 23, because an account is private or not "
          f"consenting: {(uf['t'] < 23 * DAY).sum() - len(fu):,} of {(uf['t'] < 23 * DAY).sum():,}; consenting "
          f"accounts usable: {usable(W).sum():,} of {n_u:,}  [{time.time() - t0:.0f} s]")
