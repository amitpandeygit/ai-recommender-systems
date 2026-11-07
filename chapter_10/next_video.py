"""Next-video test: random walks, LightGCN and SimGCL against matrix factorisation (sections 10.3 and 10.4).

Every method answers each test request (days 23 to 30) with a row of 10 videos,
using what the server had received, and could credit to the user, before the
request: Chapter 9's test, with the same requests, histories and allowed videos.
The graph is the user-video click graph of consenting users at the cutoff. Each
method gets the request's fresh history:

  co-occurrence  - Chapter 9's item-to-item baseline
  iALS           - Chapter 4's settings on the same click edges; the user vector is
                   solved from the request's history (fold-in)
  two-tower      - Chapter 6's model with its chosen settings, trained on the
                   snapshot's implicit scores; the user tower reads the history
  RP3beta        - three-step random walks, popular end points penalised
  LightGCN       - propagated node vectors; the user vector is rebuilt from the
                   request's history (graph_models.request_vectors)
  SimGCL         - LightGCN trained with a contrastive loss between noisy views

It is compared with Chapter 8's ranker, as in Chapter 9. Settings are chosen on
the validation week (fitted on what had arrived by day 16, tested on the requests of
days 16 to 23) by NDCG@10, Chapter 5's rule (LightGCN checked every 20 epochs up to
300, SimGCL every 5 up to 100); the neural models are then refitted on
what had arrived by day 23 from seeds 0, 1 and 2. Differences against Chapter 8's
ranker are paired over the same requests, with standard errors from 1,000 bootstrap
draws of whole users.

Needs Python 3.12 or later and PyTorch. Run from the repository root (about 50 minutes):
    python chapter_10/next_video.py
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
from graph_data import (DAY, chapter8_rows, click_graph, environment, load, ndcg, paired, phase, row_line,  # noqa: E402
                        top10)
from graph_models import (fit_lightgcn, history_nodes, ials_fold_in, layer_vectors, request_vectors,  # noqa: E402
                          rp3beta, two_tower_scores, walk_scores)
import baselines as B  # noqa: E402
from seq_data import training_sequences  # noqa: E402
from slate_data import split  # noqa: E402

ALPHAS, BETAS = (0.5, 1.0), (0.0, 0.25, 0.5, 0.75)
LAYERS = (1, 2, 3)
CL = (0.05, 0.2)
EPOCHS, EVERY = 300, 20
CL_EPOCHS, CL_EVERY = 100, 5            # SimGCL learns much faster: checked every 5 epochs
SEEDS = (0, 1, 2)


def edges_of(A, n_users):
    """The click graph as undirected edges between user nodes and item nodes."""
    u, i = A.nonzero()
    return u, i + n_users


def gcn_scores(model, A_hat, deg, H, users, n_users, n_items):
    """f(a, b): scores of every item for requests a to b, from request-time user vectors."""
    L, final = layer_vectors(model, A_hat)
    item_vec = final[n_users:n_users + n_items]
    own = np.where((deg[users] > 0)[:, None], L[0][users], 0.0)  # an unseen user has no vector of its own
    nb = history_nodes(H, n_users)
    U = request_vectors(L, deg, nb, own)
    return lambda a, b: U[a:b] @ item_vec.T


if __name__ == "__main__":
    t0 = time.time()
    print(environment())
    W = load()
    n_u, n_i = len(W["users"]["vec"]), len(W["items"]["vec"])
    video_nodes = np.flatnonzero(W["video"]) + n_u
    n_nodes = n_u + n_i
    print(f"Chapter 2's world  [{time.time() - t0:.0f} s]")

    # ---- validation week: fitted on day 16, tested on days 16 to 23
    Rv, Hv, ivv, badv = phase(W, 16, W["snaps"][16])
    A16 = click_graph(W, 16 * DAY)
    pop16 = B.popularity(W, 16 * DAY)
    print(f"Validation week: {len(Rv['user']):,} requests; graph at day 16: {A16.shape[0]:,} users with "
          f"{(A16.getnnz(axis=1) > 0).sum():,} linked, {(A16.getnnz(axis=0) > 0).sum():,} videos linked, "
          f"{A16.nnz:,} user-video edges")
    grid = {}
    for a in ALPHAS:
        for b in BETAS:
            Wm = rp3beta(A16, a, b)
            grid[(a, b)] = ndcg(Rv, top10(lambda x, y: walk_scores(Wm, Hv[x:y]), badv, tie=pop16))
    best_rp3 = max(grid, key=lambda k: round(grid[k], 9))
    print("  RP3beta (alpha, beta): " + " ".join(f"({a}, {b}):{v:.4f}" for (a, b), v in grid.items())
          + f"   chosen: {best_rp3}  [{time.time() - t0:.0f} s]")
    e16 = edges_of(A16, n_u)
    chosen = {}
    for layers in LAYERS:
        curve = []

        def check(epoch, model, A_hat, curve=curve):
            deg = np.bincount(np.r_[e16[0], e16[1]], minlength=n_nodes)
            f = gcn_scores(model, A_hat, deg, Hv, Rv["user"], n_u, n_i)
            curve.append((epoch, ndcg(Rv, top10(f, badv, tie=pop16))))
        fit_lightgcn(e16, n_nodes, n_u, video_nodes, layers=layers, epochs=EPOCHS, every=EVERY, after=check)
        best = max(curve, key=lambda x: round(x[1], 9))
        chosen[("LightGCN", layers)] = best
        print(f"  LightGCN, {layers} layer{'s' if layers > 1 else ''}:  " + " ".join(f"{e}:{x:.4f}" for e, x in curve)
              + f"   best: {best[0]}  [{time.time() - t0:.0f} s]")
    lay = max(LAYERS, key=lambda k: round(chosen[("LightGCN", k)][1], 9))
    ep = chosen[("LightGCN", lay)][0]
    print(f"  chosen LightGCN: {lay} layers, {ep} epochs")
    for cl in CL:
        curve = []

        def check(epoch, model, A_hat, curve=curve):
            deg = np.bincount(np.r_[e16[0], e16[1]], minlength=n_nodes)
            f = gcn_scores(model, A_hat, deg, Hv, Rv["user"], n_u, n_i)
            curve.append((epoch, ndcg(Rv, top10(f, badv, tie=pop16))))
        fit_lightgcn(e16, n_nodes, n_u, video_nodes, layers=lay, epochs=CL_EPOCHS, cl=cl, every=CL_EVERY,
                     after=check)
        best = max(curve, key=lambda x: round(x[1], 9))
        chosen[("SimGCL", cl)] = best
        print(f"  SimGCL, weight {cl}:  " + " ".join(f"{e}:{x:.4f}" for e, x in curve)
              + f"   best: {best[0]}  [{time.time() - t0:.0f} s]")
    cl = max(CL, key=lambda k: round(chosen[("SimGCL", k)][1], 9))
    ep_cl = chosen[("SimGCL", cl)][0]
    print(f"  chosen SimGCL: weight {cl}, {ep_cl} epochs ({lay} layers)")
    del Rv, Hv, badv

    # ---- test week: fitted on day 23, tested on days 23 to 30
    sp = split(W, "test")
    R, H, iv, bad = phase(W, 23, sp["S"])
    A = click_graph(W, 23 * DAY)
    e23 = edges_of(A, n_u)
    deg23 = np.bincount(np.r_[e23[0], e23[1]], minlength=n_nodes)
    pop = B.popularity(W, 23 * DAY)
    print(f"\nTest week: {len(R['user']):,} requests; graph at day 23: {(A.getnnz(axis=1) > 0).sum():,} users and "
          f"{(A.getnnz(axis=0) > 0).sum():,} videos linked, {A.nnz:,} user-video edges  [{time.time() - t0:.0f} s]")
    print(f"  {'':<36}{'setting':>10}{'NDCG@10':>9}{'online':>9}{'later':>9}{'against Chapter 8':>18}")
    ch8, _, _ = chapter8_rows(W, sp, R, bad)
    base = row_line("Chapter 8's ranker", W, R, ch8, None)
    row_line("popularity, last 7 days", W, R, top10(pop, bad), base)
    G = B.cooccurrence(training_sequences(W, 23 * DAY)[0], n_i)
    row_line("co-occurrence (Chapter 9)", W, R, top10(lambda a, b: B.cooc_scores(G, H[a:b]), bad, tie=pop), base)
    U, Y = ials_fold_in(A, H)
    row_line("iALS, fold-in (Chapter 4)", W, R, top10(lambda a, b: U[a:b] @ Y.T, bad, tie=pop), base)
    print(f"  [{time.time() - t0:.0f} s]")
    f_tt = two_tower_scores(W, sp["S"], H, R["user"])
    row_line("two-tower (Chapter 6)", W, R, top10(f_tt, bad, tie=pop), base)
    Wm = rp3beta(A, *best_rp3)
    row_line("RP3beta", W, R, top10(lambda a, b: walk_scores(Wm, H[a:b]), bad, tie=pop), base,
             extra=f"{best_rp3[0]}, {best_rp3[1]}")
    print(f"  [{time.time() - t0:.0f} s]")
    for label, kw, setting in (("LightGCN", dict(layers=lay, epochs=ep), f"{lay}, {ep}"),
                               ("SimGCL", dict(layers=lay, epochs=ep_cl, cl=cl), f"{cl}, {ep_cl}")):
        ons, nds = [], []
        for seed in SEEDS:
            model, A_hat = fit_lightgcn(e23, n_nodes, n_u, video_nodes, seed=seed, **kw)
            top = top10(gcn_scores(model, A_hat, deg23, H, R["user"], n_u, n_i), bad, tie=pop)
            on = row_line(f"{label}, seed {seed}", W, R, top, base, extra=setting)
            ons.append(on)
            nds.append(ndcg(R, top))
        d, se = paired(np.mean(ons, axis=0), base, R["user"])[:2]
        later = R["place"] >= 1
        print(f"  {label + ', mean of 3 seeds':<36}{setting:>10}{np.mean(nds):>9.4f}{np.mean(ons):>9.4f}"
              f"{np.mean(ons, axis=0)[later].mean():>9.4f}{f'{d:+.4f} ({se:.4f})':>18}  [{time.time() - t0:.0f} s]")
    print("  (online: expected satisfied plays per row, all test requests; later: requests after the first of "
          "their visit; against: paired difference over the same requests, with its standard error in brackets "
          "from 1,000 bootstrap draws of whole users; setting: RP3beta's alpha and beta, LightGCN's layers and "
          "epochs, SimGCL's loss weight and epochs)")
