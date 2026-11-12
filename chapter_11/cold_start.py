"""Content for new items, raw and aligned with clicks (sections 11.6 and 11.7).

New videos: section 10.5's test. A video is new while it is below Chapter 8's pool
threshold (fewer than 5 users with a positive implicit score at the snapshot); it
reaches users through the "New for you" slot (slot 4). For each test request (days 23
to 30), each method picks one new video the request may show; the result is its
expected satisfied plays at slot 4, from the hidden taste at that moment, paired
against Chapter 8's choice, over all new videos and within three groups by how many
consenting users had clicked them by the cutoff. Every method except Chapter 8's
source puts the user's language first (1,000 added to the score of every video in the
user's app language), as section 10.5 found best. The methods:

  Chapter 8's source    - same language + 0.5 x Chapter 3's content similarity
  LightGCN + KG         - section 10.5's best: the click graph plus the knowledge graph
  raw text              - cosine between the video's title vector (Qwen3-Embedding) and
                          the request's profile, the mean over its history (Listing 11.2)
  raw content           - the same with the fused content vector (Listing 11.3): title,
                          thumbnail, frames and sound, each with weight 1
  mapped to clicks      - Listing 11.5: a ridge regression from the fused content vector
                          to LightGCN's trained vector of each established video (in
                          Chapter 8's pool); a new video's vector is the regression's
                          output, scored against the request's folded-in LightGCN vector
                          (van den Oord and colleagues, 2013, for music)
  aligned with clicks   - Listing 11.6: a small network trained on the fused content
                          vectors of pairs of videos clicked one after the other by the
                          same user, with a symmetric in-batch softmax loss (as QARM and
                          CAMIE align a pretrained model with co-engagement), then used
                          like raw content

New products: section 10.5's launch test. 150 products, drawn at random, are treated as
launched at the cutoff (their earlier orders hidden); every consenting user who bought
one in the following week gets the 150 ranked; NDCG@10 and recall@10, paired against
Chapter 3's content similarity. A product's content is its title and its picture. The
graph for the mapping is LightGCN on purchases (Chapter 10), and the alignment pairs are
products bought one after the other by the same user.

Settings are chosen on the validation week (cutoff at day 16, the same tests) by
NDCG@10 over new items, Chapter 5's rule: the ridge penalty, and the aligner's epochs
(checked every 5 up to 60). LightGCN and LightGCN + KG keep Chapter 10's validation
choices. The models are then refitted on day 23 from seeds 0, 1 and 2 (the regression
and raw content have no seed).

Needs Python 3.12 or later, PyTorch and text_embeddings.npz. Run from the repository
root (about 35 minutes):
    python chapter_11/cold_start.py
"""
import os
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "2"
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from content_data import MODELS, load, modalities, text_embeddings  # noqa: E402
from content_models import fuse, profiles  # noqa: E402
from graph_data import DAY, click_graph, environment, paired, phase  # noqa: E402
from graph_models import fit_lightgcn, graph_degrees, history_nodes, layer_vectors, request_vectors  # noqa: E402
from cold_items import (LAYERS, PRODUCT_RELATIONS, VIDEO_RELATIONS, chapter8_new_scores, click_pairs,  # noqa: E402
                        kg_edges, launches, new_ndcg, new_video_choice, product_ranking, product_test, slot_plays)
from seq_data import training_sequences  # noqa: E402
from similarity_recs import item_features  # noqa: E402

SEEDS = (0, 1, 2)
EPOCHS = {"videos": {"LightGCN": 240, "LightGCN + KG": 240},      # Chapter 10's validation choices (10.4, 10.5)
          "products": {"LightGCN": 200, "LightGCN + KG": 40}}
PENALTIES = (1.0, 10.0, 100.0, 1_000.0)
ALIGN_EPOCHS, ALIGN_EVERY = 60, 5
# Section 11.7's validation choices for new videos, used by content_pipeline.py: the settings
# of both methods, and the method with the higher validation NDCG@10 over new videos
CHOSEN = {"method": "aligned with clicks", "penalty": 10.0, "epochs": 5}


# ---------------------------------------------------------------- Listing 11.5: map content into the click space
def fit_map(X, Y, penalty):
    """Ridge regression from content vectors X to click-trained vectors Y (rows: items
    with enough clicks). Returns M such that X @ M estimates Y; the constant is a last
    column of ones."""
    X1 = np.hstack([X, np.ones((len(X), 1), dtype=X.dtype)]).astype(np.float64)
    A = X1.T @ X1 + penalty * np.eye(X1.shape[1])
    A[-1, -1] -= penalty                                       # the constant is not penalised
    return np.linalg.solve(A, X1.T @ Y)


def mapped(X, M):
    return np.hstack([X, np.ones((len(X), 1), dtype=X.dtype)]) @ M


# ---------------------------------------------------------------- Listing 11.6: align content with co-clicks
class Aligner(torch.nn.Module):
    """A small network from a fused content vector to a 64-number unit vector."""
    def __init__(self, d_in, d=64, hidden=256, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.net = torch.nn.Sequential(torch.nn.Linear(d_in, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, d))

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


def fit_aligner(X, pairs, epochs, batch=1_024, tau=0.1, lr=1e-3, seed=0, every=0, after=None):
    """Pairs (i, j) of items engaged one after the other by the same user should end up
    closer than the other items of the batch: a softmax over the batch in both directions."""
    model = Aligner(X.shape[1], seed=seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xt = torch.as_tensor(X)
    rng = np.random.default_rng(seed)
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(pairs))
        for s in range(0, len(pairs), batch):
            p = pairs[order[s:s + batch]]
            a, b = model(Xt[p[:, 0]]), model(Xt[p[:, 1]])
            logits = a @ b.T / tau
            target = torch.arange(len(p))
            loss = (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)) / 2
            opt.zero_grad()
            loss.backward()
            opt.step()
        if every and epoch % every == 0 and after is not None:
            after(epoch, model)
    return model


def embed(model, X):
    with torch.no_grad():
        return model(torch.as_tensor(X)).numpy()


# ---------------------------------------------------------------- the methods for new videos
def lightgcn_video(W, cutoff, R, H, seed, kg):
    """LightGCN on the click graph (kg False) or with the knowledge graph: the requests'
    folded-in user vectors and every item's final vector."""
    n_u, n_i = len(W["users"]["vec"]), len(W["items"]["vec"])
    u, i = click_graph(W, cutoff).nonzero()
    rows, cols, n_nodes = u, i + n_u, n_u + n_i
    if kg:
        e, n_ent = kg_edges(W, np.flatnonzero(W["video"]), VIDEO_RELATIONS, n_u + n_i)
        rows, cols, n_nodes = np.r_[rows, e[0]], np.r_[cols, e[1]], n_nodes + n_ent
    deg = graph_degrees(rows, cols, n_nodes)
    name = "LightGCN + KG" if kg else "LightGCN"
    m, A_hat = fit_lightgcn((rows, cols), n_nodes, n_u, np.flatnonzero(W["video"]) + n_u, layers=LAYERS,
                            epochs=EPOCHS["videos"][name], seed=seed)
    L, final = layer_vectors(m, A_hat)
    own = np.where((deg[R["user"]] > 0)[:, None], L[0][R["user"]], 0.0)
    return request_vectors(L, deg, history_nodes(H, n_u), own), final[n_u:n_u + n_i]


def video_methods(W, cutoff, S, R, H, cols, X, text, use, choice=None):
    """Every method's scores for every request and new video (cols), passed to
    use(name, seed or setting, scores), the user's language first. choice None: the
    validation settings are all passed; otherwise choice holds the chosen penalty and
    epochs and the models are fitted from seeds 0, 1 and 2."""
    users, items = W["users"], W["items"]
    same = 1_000.0 * (users["language"][R["user"]][:, None] == items["language"][cols][None, :])
    use("Chapter 8's source", None, chapter8_new_scores(W, S, R, cols))
    use("raw text", None, profiles(text, H) @ text[cols].T + same)
    use("raw content", None, profiles(X, H) @ X[cols].T + same)
    established = np.flatnonzero(W["video"] & S["ok"])
    pairs = click_pairs(training_sequences(W, cutoff)[0])
    for seed in (SEEDS if choice else (0,)):
        if choice:
            U, V = lightgcn_video(W, cutoff, R, H, seed, kg=True)
            use("LightGCN + KG", seed, U @ V[cols].T + same)
        U, V = lightgcn_video(W, cutoff, R, H, seed, kg=False)
        for p in ((choice["penalty"],) if choice else PENALTIES):
            M = fit_map(X[established], V[established], p)
            use("mapped to clicks", seed if choice else p, U @ mapped(X[cols], M).T + same)

        def aligned(e, model, seed=seed):
            Z = embed(model, X)
            use("aligned with clicks", seed if choice else e, profiles(Z, H) @ Z[cols].T + same)
        if choice:
            aligned(None, fit_aligner(X, pairs, choice["epochs"], seed=seed))
        else:
            fit_aligner(X, pairs, ALIGN_EPOCHS, seed=seed, every=ALIGN_EVERY, after=aligned)


# ---------------------------------------------------------------- the methods for new products
def purchase_pairs(orders, ok, new):
    """Pairs of different products bought one after the other by the same user (orders
    kept by ok; launched products' orders are hidden)."""
    keep = ok & ~new[orders["item_id"]]
    u, i, t = orders["user_id"][keep], orders["item_id"][keep], orders["time"][keep]
    o = np.lexsort((t, u))
    u, i = u[o], i[o]
    same = u[1:] == u[:-1]
    p = np.stack([i[:-1][same], i[1:][same]], axis=1)
    return p[p[:, 0] != p[:, 1]]


def product_methods(W, orders, cutoff, test, new, X, feats, use, choice=None):
    """The same methods for launched products: scores of every user (test[3]) for every
    item, passed to use(name, seed or setting, scores)."""
    users, items = W["users"], W["items"]
    pairs, bought, _, tu = test
    n_u, n_i = len(users["vec"]), len(items["vec"])
    hist = np.zeros((len(tu), n_i))
    for k, u in enumerate(tu):
        hist[k, list(bought.get(int(u), []))] = 1.0
    hist /= np.maximum(hist.sum(axis=1, keepdims=True), 1.0)
    P = hist @ feats
    use("content similarity (Chapter 3)", None, P @ feats.T)
    use("raw content", None, (hist @ X) @ X.T)
    prods = np.flatnonzero(items["is_product"])
    established = prods[~new[prods]]
    nbrs = [np.array(sorted(bought.get(int(u), [])), dtype=np.int64) + n_u for u in tu]
    buy = (pairs[:, 0], pairs[:, 1] + n_u)
    ok = users["current_consent"][orders["user_id"]] & (orders["time"] < cutoff)
    seq = purchase_pairs(orders, ok, new)
    for seed in (SEEDS if choice else (0,)):
        for name in (("LightGCN + KG", "LightGCN") if choice else ("LightGCN",)):
            rows, cols, n_nodes = buy[0], buy[1], n_u + n_i
            if name == "LightGCN + KG":
                e, n_ent = kg_edges(W, prods, PRODUCT_RELATIONS, n_u + n_i)
                rows, cols, n_nodes = np.r_[rows, e[0]], np.r_[cols, e[1]], n_nodes + n_ent
            deg = graph_degrees(rows, cols, n_nodes)
            m, A_hat = fit_lightgcn((rows, cols), n_nodes, n_u, prods + n_u, layers=LAYERS,
                                    epochs=EPOCHS["products"][name], seed=seed)
            L, final = layer_vectors(m, A_hat)
            own = np.where((deg[tu] > 0)[:, None], L[0][tu], 0.0)
            U, V = request_vectors(L, deg, nbrs, own), final[n_u:n_u + n_i]
            if name == "LightGCN + KG":
                use(name, seed, U @ V.T)
                continue
            for p in ((choice["penalty"],) if choice else PENALTIES):
                M = fit_map(X[established], V[established], p)
                use("mapped to purchases", seed if choice else p, U @ mapped(X, M).T)

        def aligned(e, model, seed=seed):
            Z = embed(model, X)
            use("aligned with purchases", seed if choice else e, (hist @ Z) @ Z.T)
        if choice:
            aligned(None, fit_aligner(X, seq, choice["epochs"], seed=seed))
        else:
            fit_aligner(X, seq, ALIGN_EPOCHS, seed=seed, every=ALIGN_EVERY, after=aligned)


def choose(curves, label):
    """The setting with the highest validation NDCG@10 for each method that has settings."""
    out = {}
    for name, curve in curves.items():
        if len(curve) > 1:
            best = max(curve, key=lambda x: round(x[1], 9))
            out[name] = best[0]
            print(f"  {name:<26}" + " ".join(f"{k:g}:{v:.4f}" for k, v in curve) + f"   chosen: {best[0]:g}")
    return out


if __name__ == "__main__":
    t0 = time.time()
    print(environment())
    W = load()
    items, users = W["items"], W["users"]
    n_i = len(items["vec"])
    E, idx = text_embeddings(W, MODELS[-1])
    T = idx["titles"]
    text = E[T[np.arange(n_i), items["language"]]]
    parts = [(text, np.ones(n_i, dtype=bool))] + list(modalities(W).values())
    X = fuse(parts, [1.0] * len(parts))
    print(f"Chapter 2's world; fused content vectors of {X.shape[1]} numbers (title {text.shape[1]}, thumbnail, "
          f"frames, sound)  [{time.time() - t0:.0f} s]")

    # ---- new videos, validation week
    S16 = W["snaps"][16]
    cols16 = np.flatnonzero(W["video"] & ~S16["ok"])
    Rv, Hv, _, badv = phase(W, 16, S16)
    bv = badv[:, cols16]
    del badv
    curves = {}

    def check(name, key, s):
        if key is not None:
            curves.setdefault(name, []).append((key, new_ndcg(Rv, s, bv, cols16)[0]))
    video_methods(W, 16 * DAY, S16, Rv, Hv, cols16, X, text, check)
    print(f"\nNew videos, validation week: {len(cols16):,} new videos at day 16. NDCG@10 over new videos for each "
          f"setting (ridge penalty; aligner epochs)  [{time.time() - t0:.0f} s]")
    ch = choose(curves, "videos")
    choice = {"penalty": ch["mapped to clicks"], "epochs": int(ch["aligned with clicks"])}
    best = {k: max(round(v, 4) for _, v in c) for k, c in curves.items()}
    print(f"  higher validation NDCG@10 (section 11.8's content model): {max(best, key=best.get)} "
          f"({', '.join(f'{k} {v:.4f}' for k, v in best.items())})")
    del Rv, Hv, bv

    # ---- new videos, test week
    S = W["snaps"][23]
    cols = np.flatnonzero(W["video"] & ~S["ok"])
    R, H, _, bad = phase(W, 23, S)
    bn = bad[:, cols]
    del bad
    clicked = click_graph(W, 23 * DAY).getnnz(axis=0)[cols]
    groups = {"no clicks": clicked == 0, "1 to 4 users": (clicked >= 1) & (clicked <= 4),
              "5 or more users": clicked >= 5}
    res, res_g = {}, {}

    def collect(name, seed, s):
        c = new_video_choice(s, bn, cols)
        res.setdefault(name, []).append((slot_plays(W, R, c), new_ndcg(R, s, bn, cols)[0], c))
        for gname, gm in groups.items():
            res_g.setdefault((name, gname), []).append(slot_plays(W, R, new_video_choice(s[:, gm], bn[:, gm],
                                                                                         cols[gm])))
    video_methods(W, 23 * DAY, S, R, H, cols, X, text, collect, choice=choice)
    print(f"\nNew videos, test week: {len(cols):,} new videos at day 23; {len(R['user']):,} requests. Consenting users "
          f"who had clicked them: " + ", ".join(f"{g} {m.sum():,}" for g, m in groups.items())
          + f"  [{time.time() - t0:.0f} s]")
    print(f"  {'':<28}{'setting':>9}{'NDCG@10 new':>13}{'slot-4 plays':>14}{'against Chapter 8':>20}"
          f"{'same language':>15}")
    settings = {"mapped to clicks": f"{choice['penalty']:g}", "aligned with clicks": str(choice["epochs"]),
                "LightGCN + KG": "240"}
    base = None
    plays_of = {}
    for name, runs in res.items():
        plays = np.mean([r[0] for r in runs], axis=0)
        plays_of[name] = plays
        nd = np.mean([r[1] for r in runs])
        c = runs[0][2]
        lang = np.mean(items["language"][c[c >= 0]] == users["language"][R["user"][c >= 0]])
        if base is None:
            base, against = plays, ""
        else:
            against = "{:+.4f} ({:.4f})".format(*paired(plays, base, R["user"])[:2])
        print(f"  {name:<28}{settings.get(name, ''):>9}{nd:>13.4f}{plays.mean():>14.4f}{against:>20}{lang:>15.1%}")
    for a_, b_ in (("raw content", "raw text"), ("aligned with clicks", "raw content"),
                   ("mapped to clicks", "raw content"), ("aligned with clicks", "LightGCN + KG")):
        print(f"  {a_} against {b_}: " + "{:+.4f} ({:.4f}) [{:+.4f}, {:+.4f}]".format(
            *paired(plays_of[a_], plays_of[b_], R["user"])))
    print("  (slot-4 plays: expected satisfied plays of the chosen new video alone at slot 4; against: paired "
          "difference over the same requests, standard error from 1,000 bootstrap draws of whole users; same "
          "language: share of chosen videos in the user's app language, first seed; models with seeds are means over "
          "seeds 0, 1 and 2)")
    print("\nThe same choice within each group of new videos (slot-4 plays; against Chapter 8's source in the same "
          "group, standard error in brackets)")
    print(f"  {'':<28}" + "".join(f"{g + ' (' + format(m.sum(), ',') + ')':>30}" for g, m in groups.items()))
    for name in res:
        cells = []
        for gname in groups:
            plays = np.mean(res_g[(name, gname)], axis=0)
            if name == "Chapter 8's source":
                cells.append(f"{plays.mean():.4f}")
            else:
                d, se = paired(plays, np.mean(res_g[("Chapter 8's source", gname)], axis=0), R["user"])[:2]
                cells.append(f"{plays.mean():.4f} {d:+.4f} ({se:.4f})")
        print(f"  {name:<28}" + "".join(f"{c:>30}" for c in cells))
    del R, H, bn, res, res_g

    # ---- new products
    sys.path.insert(0, str(HERE.parent / "chapter_03"))
    from tesselo_shop import build
    _, items_b, _, _, orders, _ = build(W["cfg"])
    assert np.array_equal(items_b["vec"], items["vec"])
    feats = item_features(items, n_i)
    new_p = launches(W)
    pv = product_test(W, orders, 16 * DAY, 23 * DAY, new_p)
    curves = {}
    product_methods(W, orders, 16 * DAY, pv, new_p, X, feats,
                    lambda name, key, s: curves.setdefault(name, []).append(
                        (key, product_ranking(s, pv[3], new_p, pv[1], pv[2])[0].mean())) if key is not None else None)
    print(f"\nNew products: {new_p.sum():,} launched at the cutoff. Validation week: {len(pv[3]):,} consenting users "
          f"bought one in days 16 to 23. NDCG@10 for each setting  [{time.time() - t0:.0f} s]")
    ch_p = choose(curves, "products")
    choice_p = {"penalty": ch_p["mapped to purchases"], "epochs": int(ch_p["aligned with purchases"])}
    tp = product_test(W, orders, 23 * DAY, 30 * DAY, new_p)
    scores = {}
    product_methods(W, orders, 23 * DAY, tp, new_p, X, feats,
                    lambda name, key, s: scores.setdefault(name, []).append(s), choice=choice_p)
    pairs, bought, pos, tu = tp
    print(f"\nNew products, test week: {len(tu):,} consenting users bought a launched product in days 23 to 30  "
          f"[{time.time() - t0:.0f} s]")
    print(f"  {'':<34}{'setting':>9}{'NDCG@10':>9}{'recall@10':>11}{'against content':>22}")
    settings = {"mapped to purchases": f"{choice_p['penalty']:g}", "aligned with purchases": str(choice_p["epochs"]),
                "LightGCN + KG": "40"}
    base = None
    for name, runs in scores.items():
        r = [product_ranking(x, tu, new_p, bought, pos) for x in runs]
        nd, rc = np.mean([x[0] for x in r], axis=0), np.mean([x[1] for x in r], axis=0)
        if base is None:
            base, against = nd, ""
        else:
            against = "{:+.4f} ({:.4f})".format(*paired(nd, base, tu)[:2])
        print(f"  {name:<34}{settings.get(name, ''):>9}{nd.mean():>9.4f}{rc.mean():>11.4f}{against:>22}")
    print(f"  (NDCG@10 and recall@10 over each user's launched products bought in the test week, ranking all "
          f"{new_p.sum()} launched products; a random order has recall@10 {10 / new_p.sum():.4f}; against: paired "
          f"difference in NDCG@10 against content similarity, standard error from 1,000 bootstrap draws of users; "
          f"models with seeds are means over seeds 0, 1 and 2)  [{time.time() - t0:.0f} s]")
