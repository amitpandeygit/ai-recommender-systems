"""New items with few or no clicks: knowledge graphs and PinSage (section 10.5).

Chapter 8's pool only takes videos with at least 5 engaged users at the snapshot;
the others are "new" and reach users through one reserved slot, "New for you" (slot
4), filled by Chapter 6's source: the user's language first, then the content
similarity of the video's title words, category and language to the user's history.
This script asks which graph helps choose that video, and then does the same for
new products in the shop.

New videos. A video is "new" while it is below the engagement threshold of Chapter 8's
pool (fewer than 5 users with a positive implicit score at the snapshot); it may
already have a few clicks. For each test request (days 23 to 30), each method picks
one new video the request may show; the result is its expected satisfied plays at
slot 4, from the hidden taste at that moment, paired against Chapter 8's choice. The
same is then done within three groups of new videos, by how many consenting users had
clicked them by the cutoff: none, 1 to 4, and 5 or more. The methods:

  Chapter 8's source  - same language + 0.5 x content similarity to the snapshot's
                        profile of the user: the scores of section 10.8's source
  most clicked        - the new video with the most clicks so far
  LightGCN            - the click graph only
  LightGCN + KG       - the click graph plus the knowledge graph: each video linked to
                        its title words, category, language and creator, so a new
                        video receives vectors from the entities it shares with others
  PinSage-style       - Listing 10.5: each video's vector computed from its own content
                        and its graph neighbours' content, the neighbours being the
                        videos most likely to be reached in two steps, video to user to
                        video (an exact version of the random-walk visit counts of Ying
                        and colleagues, KDD 2018); it is trained with an in-batch
                        softmax on pairs of videos the same user clicked one after the
                        other, not with the paper's max-margin loss and hard negatives
Each graph model is also tested with the user's language first, as Chapter 8's source
does: 1,000 is added to the score of every video in the user's app language.

New products. Shop orders (Chapter 3) of consenting users. Almost every product has 5
buyers within days, so 150 products, drawn at random (seed cfg.seed + 1010), are
treated as launched at the cutoff: their earlier orders are hidden, so they have no
purchase history. For every consenting user who bought one in the following week, each
method ranks the 150; NDCG@10 and recall@10 against what they bought, compared with the
content similarity of Chapter 3 (title words, category and language). LightGCN + KG
adds each product's brand, seller, category, title words and language; LightGCN on
the purchases alone has nothing to go on for a launched product.

Settings (epochs) are chosen on the validation week, on the same kind of test with
the cutoff at day 16, by NDCG@10 over new items, Chapter 5's rule; LightGCN uses the
layers chosen in next_video.py.

Needs Python 3.12 or later and PyTorch. Run from the repository root (about 50 minutes):
    python chapter_10/cold_items.py
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
from graph_data import DAY, click_graph, environment, load, outcomes, paired, phase  # noqa: E402
from graph_models import fit_lightgcn, graph_degrees, history_nodes, layer_vectors, request_vectors  # noqa: E402
from seq_data import training_sequences  # noqa: E402
from similarity_recs import content_scores, item_features  # noqa: E402
from tesselo_graph import RELATIONS  # noqa: E402
from metrics import ndcg_at_k  # noqa: E402
import baselines as B  # noqa: E402

LAYERS = 3                       # next_video.py's validation choice for LightGCN
EPOCHS, EVERY = 300, 20
NEW_SLOT = 4
VIDEO_RELATIONS = ("word", "category", "language", "creator")
PRODUCT_RELATIONS = ("word", "category", "language", "brand", "seller")
SEEDS = (0, 1, 2)


# ---------------------------------------------------------------- the knowledge graph as extra nodes
def kg_edges(W, items_kept, relations, first_node):
    """Edges item node -> entity node for the given relations, the entities numbered
    from first_node. Returns (rows, cols) and the number of entity nodes."""
    g = W["graph"]
    kg, n_ent = g["kg"], g["n_entities"]
    start = np.r_[0, np.cumsum(n_ent)]
    rel = np.array([RELATIONS.index(r) for r in relations])
    m = np.isin(kg["relation"], rel) & np.isin(kg["head"], items_kept)
    n_users = len(W["users"]["vec"])
    rows = kg["head"][m] + n_users
    cols = first_node + start[kg["relation"][m]] + kg["tail"][m]
    return (rows, cols), int(start[-1])


# ---------------------------------------------------------------- Listing 10.5: PinSage
def walk_neighbours(A, k=10):
    """Each item's k most visited items on a two-step walk item -> user -> item (the
    visit probabilities, as an exact version of PinSage's counts over many short walks),
    and their importance weights (summing to 1)."""
    from graph_models import rp3beta
    P = rp3beta(A, alpha=1.0, beta=0.0)                          # P[i, j]: chance a walk from i ends at j
    nb = np.argsort(-P, axis=1, kind="stable")[:, :k]
    w = np.take_along_axis(P, nb, axis=1)
    w = w / np.maximum(w.sum(axis=1, keepdims=True), 1e-12)
    return nb, w                                                 # an item with no clicks: weights all 0


class PinSage(torch.nn.Module):
    """Listing 10.5. One PinSage-style convolution: h_i = normalise(G [x_i, sum_j w_ij relu(Q x_j)])."""

    def __init__(self, n_feat, dim=32, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.Q = torch.nn.Linear(n_feat, 64)                     # what a neighbour passes on
        self.G = torch.nn.Sequential(torch.nn.Linear(n_feat + 64, 64), torch.nn.ReLU(), torch.nn.Linear(64, dim))

    def forward(self, x, x_nb, w_nb):
        agg = (w_nb[:, :, None] * F.relu(self.Q(x_nb))).sum(dim=1)   # importance pooling
        return F.normalize(self.G(torch.cat([x, agg], dim=1)), dim=1)


def fit_pinsage(feats, nb, w, pairs, epochs, batch=512, tau=0.1, every=0, after=None, seed=0):
    """In-batch softmax on pairs (i, j) of videos clicked one after the other by the same user."""
    X = torch.as_tensor(feats, dtype=torch.float32)
    NB, WN = torch.as_tensor(nb), torch.as_tensor(w, dtype=torch.float32)
    model = PinSage(feats.shape[1], seed=seed)
    opt = torch.optim.Adam(model.parameters(), lr=0.003)
    rng = np.random.default_rng(seed)
    embed = lambda idx: model(X[idx], X[NB[idx]], WN[idx])      # noqa: E731
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(pairs))
        for s in range(0, len(order), batch):
            p = torch.as_tensor(pairs[order[s:s + batch]])
            a, b = embed(p[:, 0]), embed(p[:, 1])
            loss = F.cross_entropy(a @ b.T / tau, torch.arange(len(p)))
            opt.zero_grad()
            loss.backward()
            opt.step()
        if every and epoch % every == 0 and after is not None:
            with torch.no_grad():
                after(epoch, embed(torch.arange(len(feats))).numpy().astype(np.float64))
    with torch.no_grad():
        return embed(torch.arange(len(feats))).numpy().astype(np.float64)


def click_pairs(seqs):
    """Pairs of different videos clicked one after the other by the same user."""
    out = [np.stack([s[:-1], s[1:]], axis=1) for s in seqs if len(s) >= 2]
    p = np.concatenate(out)
    return p[p[:, 0] != p[:, 1]]


# ---------------------------------------------------------------- new videos: the slot-4 choice
def new_video_choice(scores, bad_new, cols):
    """Each request's best allowed new video (-1 if none). scores and bad_new have one
    column per new video (cols)."""
    s = np.where(bad_new, -np.inf, scores)
    best = np.argmax(s, axis=1)
    return np.where(np.isfinite(s[np.arange(len(s)), best]), cols[best], -1)


def slot_plays(W, R, choice):
    """Expected satisfied plays of each request's chosen video alone at slot 4."""
    top = np.full((len(choice), 10), -1)
    top[:, NEW_SLOT - 1] = choice
    return outcomes(W, R, top)[1].sum(axis=1)


def new_ndcg(R, scores, bad_new, cols):
    """NDCG@10 over the allowed new videos, for requests with a satisfied new video."""
    is_new = set(cols.tolist())
    out = []
    for k, pos in enumerate(R["pos"]):
        hit = {i for i in pos if i in is_new}
        if hit:
            s = np.where(bad_new[k], -np.inf, scores[k])
            top = np.argsort(-s, kind="stable")[:10]
            out.append(ndcg_at_k([int(cols[j]) for j in top if np.isfinite(s[j])], {i: 1.0 for i in hit}))
    return float(np.mean(out)) if out else 0.0, len(out)


def chapter8_new_scores(W, S, R, cols):
    """Chapter 8's "New for you" scores (section 10.8's source, pipeline_test.new_candidates):
    the user's language first, then the content similarity of each new video to the
    snapshot's profile of the user."""
    items, users = W["items"], W["users"]
    feats = item_features(items, len(items["vec"]))
    uu, inv = np.unique(R["user"], return_inverse=True)
    sim = content_scores(S["weight"][uu], feats)[0][:, cols].astype(np.float32)
    same = (users["language"][uu][:, None] == items["language"][cols][None, :]).astype(np.float32)
    return (same + 0.5 * sim)[inv]


def video_methods(W, cutoff, S, R, H, cols, use, epochs=None):
    """Every method's scores for every request and new video (cols), passed to
    use(name, epoch or seed, scores). With epochs None, the graph models are fitted for
    EPOCHS and checked every EVERY epochs (validation); otherwise each is fitted for its
    chosen epochs from seeds 0, 1 and 2. Each graph model's scores are passed again with
    the user's language first."""
    items, users = W["items"], W["users"]
    n_u, n_i = len(users["vec"]), len(items["vec"])
    video = W["video"]
    feats = item_features(items, n_i)
    A = click_graph(W, cutoff)
    hist = np.zeros((len(H), n_i), dtype=np.float32)
    for k, h in enumerate(H):
        hist[k, np.unique(h[h >= 0])] = 1.0
    same = (users["language"][R["user"]][:, None] == items["language"][cols][None, :]).astype(np.float32)
    use("Chapter 8's source", None, chapter8_new_scores(W, S, R, cols))

    def both(name, key, s):
        use(name, key, s)
        use(name + ", same language first", key, s + 1_000.0 * same)
    use("most clicked", None, np.broadcast_to(B.popularity(W, cutoff, days=30)[cols], (len(H), len(cols))))
    u, i = A.nonzero()
    clicks = (u, i + n_u)
    kg, n_ent = kg_edges(W, np.flatnonzero(video), VIDEO_RELATIONS, n_u + n_i)
    for name, extra, n_extra in (("LightGCN", None, 0), ("LightGCN + KG", kg, n_ent)):
        rows = clicks[0] if extra is None else np.r_[clicks[0], extra[0]]
        cols_e = clicks[1] if extra is None else np.r_[clicks[1], extra[1]]
        n_nodes = n_u + n_i + n_extra
        deg = graph_degrees(rows, cols_e, n_nodes)

        def scores_of(model, A_hat, deg=deg):
            L, final = layer_vectors(model, A_hat)
            own = np.where((deg[R["user"]] > 0)[:, None], L[0][R["user"]], 0.0)
            return (request_vectors(L, deg, history_nodes(H, n_u), own) @ final[n_u + cols].T).astype(np.float32)
        vn = np.flatnonzero(video) + n_u
        if epochs is None:
            fit_lightgcn((rows, cols_e), n_nodes, n_u, vn, layers=LAYERS, epochs=EPOCHS, every=EVERY,
                         after=lambda e, m, Ah, name=name, f=scores_of: both(name, e, f(m, Ah)))
        else:
            for seed in SEEDS:
                m, Ah = fit_lightgcn((rows, cols_e), n_nodes, n_u, vn, layers=LAYERS, epochs=epochs[name], seed=seed)
                both(name, seed, scores_of(m, Ah))
    nb, w = walk_neighbours(A)
    pairs = click_pairs(training_sequences(W, cutoff)[0])

    def ps_scores(V):
        U = hist @ V
        U /= np.maximum(np.linalg.norm(U, axis=1, keepdims=True), 1e-12)
        return (U @ V[cols].T).astype(np.float32)
    if epochs is None:
        fit_pinsage(feats, nb, w, pairs, EPOCHS, every=EVERY,
                    after=lambda e, V: both("PinSage-style", e, ps_scores(V)))
    else:
        for seed in SEEDS:
            both("PinSage-style", seed, ps_scores(fit_pinsage(feats, nb, w, pairs, epochs["PinSage-style"], seed=seed)))


# ---------------------------------------------------------------- new products
def launches(W, n=150):
    """Products treated as launched at the cutoff (seed cfg.seed + 1010): their orders
    before it are hidden, so they have no purchase history; the data top-up is kept out."""
    items = W["items"]
    prods = np.flatnonzero(items["is_product"])
    top = prods[np.argmax(items["popularity"][prods])]
    rng = np.random.default_rng(W["cfg"].seed + 1010)
    new = np.zeros(len(items["vec"]), dtype=bool)
    new[rng.choice(prods[prods != top], n, replace=False)] = True
    return new


def product_test(W, orders, cutoff, end, new):
    """Users who bought a launched product in [cutoff, end), each user's earlier
    purchases, and the purchase graph at the cutoff (consenting users only; the
    launched products' earlier orders hidden)."""
    users = W["users"]
    ok = users["current_consent"][orders["user_id"]]
    before = ok & (orders["time"] < cutoff) & ~new[orders["item_id"]]
    pu, pi = orders["user_id"][before], orders["item_id"][before]
    pairs = np.unique(np.stack([pu, pi], axis=1), axis=0)
    after_ = ok & (orders["time"] >= cutoff) & (orders["time"] < end)
    bought = {}
    for u, i in pairs.tolist():
        bought.setdefault(u, set()).add(i)
    pos = {}
    for u, i in zip(orders["user_id"][after_].tolist(), orders["item_id"][after_].tolist()):
        if new[i]:
            pos.setdefault(u, set()).add(i)
    test_users = np.array(sorted(pos))
    return pairs, bought, pos, test_users


def product_methods(W, pairs, bought, test_users, epochs=None, after=None, feats=None):
    users, items = W["users"], W["items"]
    n_u, n_i = len(users["vec"]), len(items["vec"])
    hist = np.zeros((len(test_users), n_i))
    for k, u in enumerate(test_users):
        hist[k, list(bought.get(int(u), []))] = 1.0
    profile = hist @ feats
    profile /= np.maximum(np.linalg.norm(profile, axis=1, keepdims=True), 1e-12)
    out = {"content similarity": profile @ feats.T}
    prods = np.flatnonzero(items["is_product"])
    buys = (pairs[:, 0], pairs[:, 1] + n_u)
    kg, n_ent = kg_edges(W, prods, PRODUCT_RELATIONS, n_u + n_i)
    for name, extra, n_extra in (("LightGCN", None, 0), ("LightGCN + KG", kg, n_ent)):
        rows = buys[0] if extra is None else np.r_[buys[0], extra[0]]
        cols = buys[1] if extra is None else np.r_[buys[1], extra[1]]
        n_nodes = n_u + n_i + n_extra
        deg = graph_degrees(rows, cols, n_nodes)
        nbrs = [np.array(sorted(bought.get(int(u), [])), dtype=np.int64) + n_u for u in test_users]

        def scores_of(model, A_hat):
            L, final = layer_vectors(model, A_hat)
            own = np.where((deg[test_users] > 0)[:, None], L[0][test_users], 0.0)
            return request_vectors(L, deg, nbrs, own) @ final[n_u:n_u + n_i].T
        pn = prods + n_u
        if epochs is None:
            fit_lightgcn((rows, cols), n_nodes, n_u, pn, layers=LAYERS, epochs=EPOCHS, every=EVERY,
                         after=lambda e, m, Ah, name=name, f=scores_of: after(name, e, f(m, Ah)))
        else:
            out[name] = [scores_of(*fit_lightgcn((rows, cols), n_nodes, n_u, pn, layers=LAYERS,
                                                 epochs=epochs[name], seed=s)) for s in SEEDS]
    return out


def product_ranking(scores, test_users, new, bought, pos):
    """NDCG@10 and recall@10 per user, ranking the new products each had not bought."""
    nd, rc = [], []
    for k, u in enumerate(test_users):
        s = np.where(new, scores[k], -np.inf)
        s[list(bought.get(int(u), []))] = -np.inf
        top = [int(i) for i in np.argsort(-s, kind="stable")[:10]]
        nd.append(ndcg_at_k(top, {i: 1.0 for i in pos[int(u)]}))
        rc.append(len(set(top) & pos[int(u)]) / len(pos[int(u)]))
    return np.array(nd), np.array(rc)


if __name__ == "__main__":
    t0 = time.time()
    print(environment())
    W = load()
    items, users = W["items"], W["users"]
    n_i = len(items["vec"])
    print(f"Chapter 2's world  [{time.time() - t0:.0f} s]")
    rel = W["graph"]["kg"]["relation"]
    vid = W["video"][W["graph"]["kg"]["head"]]
    print("Knowledge-graph edges of videos: " + ", ".join(
        f"{r} {np.sum(vid & (rel == RELATIONS.index(r))):,}" for r in VIDEO_RELATIONS))

    # ---- new videos, validation week: choose the epochs
    S16 = W["snaps"][16]
    cols16 = np.flatnonzero(W["video"] & ~S16["ok"])
    Rv, Hv, _, badv = phase(W, 16, S16)
    bv = badv[:, cols16]
    del badv
    curves = {}

    def check(name, e, s):
        if e is not None and "same language" not in name:
            curves.setdefault(name, []).append((e, new_ndcg(Rv, s, bv, cols16)[0]))
    video_methods(W, 16 * DAY, S16, Rv, Hv, cols16, check)
    n_val = new_ndcg(Rv, np.zeros(bv.shape), bv, cols16)[1]
    print(f"\nNew videos, validation week: {len(cols16):,} new videos at day 16; {n_val:,} requests with a "
          f"satisfied new video. NDCG@10 over new videos after each check (epochs)")
    ep = {}
    for name, curve in curves.items():
        best = max(curve, key=lambda x: round(x[1], 9))
        ep[name] = best[0]
        print(f"  {name:<16}" + " ".join(f"{e}:{x:.4f}" for e, x in curve) + f"   chosen: {best[0]}")
    print(f"  [{time.time() - t0:.0f} s]")
    del Rv, Hv, bv

    # ---- new videos, test week
    S = W["snaps"][23]
    cols = np.flatnonzero(W["video"] & ~S["ok"])
    R, H, _, bad = phase(W, 23, S)
    bn = bad[:, cols]
    del bad
    clicked = click_graph(W, 23 * DAY).getnnz(axis=0)[cols]     # consenting users who had clicked each new video
    groups = {"no clicks": clicked == 0, "1 to 4 users": (clicked >= 1) & (clicked <= 4),
              "5 or more users": clicked >= 5}
    res, res_g = {}, {}

    def collect(name, seed, s):
        choice = new_video_choice(s, bn, cols)
        res.setdefault(name, []).append((slot_plays(W, R, choice), new_ndcg(R, s, bn, cols)[0], choice))
        for gname, gm in groups.items():                         # the same choice within each group
            res_g.setdefault((name, gname), []).append(slot_plays(W, R, new_video_choice(s[:, gm], bn[:, gm],
                                                                                          cols[gm])))
    video_methods(W, 23 * DAY, S, R, H, cols, collect, epochs=ep)
    is_new = set(cols.tolist())
    print(f"\nNew videos, test week: {len(cols):,} of {W['video'].sum():,} videos are new at day 23 "
          f"({np.mean([any(i in is_new for i in p) for p in R['pos']]):.1%} of requests have a satisfied "
          f"new video); {(~bn).any(axis=1).mean():.1%} of {len(R['user']):,} requests may show at least one. "
          f"Consenting users who had clicked them: " + ", ".join(f"{g} {m.sum():,}" for g, m in groups.items())
          + f"; at most {clicked.max()}  [{time.time() - t0:.0f} s]")
    print(f"  {'':<38}{'epochs':>8}{'NDCG@10 new':>13}{'slot-4 plays':>14}{'against Chapter 8':>20}"
          f"{'same language':>15}")
    base = None
    for name, runs in res.items():
        plays = np.mean([r[0] for r in runs], axis=0)
        nd = np.mean([r[1] for r in runs])
        choice = runs[0][2]
        lang = np.mean(items["language"][choice[choice >= 0]] == users["language"][R["user"][choice >= 0]])
        if base is None:
            base, against = plays, ""
        else:
            d, se = paired(plays, base, R["user"])[:2]
            against = f"{d:+.4f} ({se:.4f})"
        print(f"  {name:<38}{ep.get(name.split(',')[0], ''):>8}{nd:>13.4f}{plays.mean():>14.4f}{against:>20}"
              f"{lang:>15.1%}")
    print("  (slot-4 plays: expected satisfied plays of the chosen new video alone at slot 4, from the hidden taste "
          "at the request; against: paired difference over the same requests, standard error from 1,000 bootstrap "
          "draws of whole users; same language: share of chosen videos in the user's app language, first seed; "
          "graph models are means over seeds 0, 1 and 2)")
    print("\nThe same choice within each group of new videos (slot-4 plays; against Chapter 8's source in the same "
          "group, standard error in brackets)")
    print(f"  {'':<38}" + "".join(f"{g + ' (' + format(m.sum(), ',') + ')':>30}" for g, m in groups.items()))
    for name in res:
        cells = []
        for gname in groups:
            plays = np.mean(res_g[(name, gname)], axis=0)
            if name == "Chapter 8's source":
                cells.append(f"{plays.mean():.4f}")
            else:
                d, se = paired(plays, np.mean(res_g[("Chapter 8's source", gname)], axis=0), R["user"])[:2]
                cells.append(f"{plays.mean():.4f} {d:+.4f} ({se:.4f})")
        print(f"  {name:<38}" + "".join(f"{c:>30}" for c in cells))
    del R, H, bn, res, res_g

    # ---- new products
    sys.path.insert(0, str(HERE.parent / "chapter_03"))
    from tesselo_shop import build
    _, items_b, _, _, orders, _ = build(W["cfg"])
    assert np.array_equal(items_b["vec"], items["vec"])          # the same world
    feats = item_features(items, n_i)
    new_p = launches(W)
    pv = product_test(W, orders, 16 * DAY, 23 * DAY, new_p)
    curves = {}
    product_methods(W, pv[0], pv[1], pv[3], feats=feats, after=lambda name, e, s: curves.setdefault(
        name, []).append((e, product_ranking(s, pv[3], new_p, pv[1], pv[2])[0].mean())))
    print(f"\nNew products: {new_p.sum():,} of {items['is_product'].sum():,} products treated as launched at the "
          f"cutoff, their earlier orders hidden.\nValidation week: {len(pv[3]):,} consenting users bought one in "
          f"days 16 to 23. NDCG@10 after each check (epochs)")
    ep_p = {}
    for name, curve in curves.items():
        best = max(curve, key=lambda x: round(x[1], 9))
        ep_p[name] = best[0]
        print(f"  {name:<16}" + " ".join(f"{e}:{x:.4f}" for e, x in curve) + f"   chosen: {best[0]}")
    pairs, bought, pos, tu = product_test(W, orders, 23 * DAY, 30 * DAY, new_p)
    sc = product_methods(W, pairs, bought, tu, epochs=ep_p, feats=feats)
    print(f"\nNew products, test week: {len(tu):,} consenting users bought a launched product in days 23 to 30  "
          f"[{time.time() - t0:.0f} s]")
    print(f"  {'':<22}{'epochs':>8}{'NDCG@10':>9}{'recall@10':>11}{'against content':>22}")
    base = None
    for name, s in sc.items():
        runs = s if isinstance(s, list) else [s]
        res = [product_ranking(x, tu, new_p, bought, pos) for x in runs]
        nd, rc = np.mean([r[0] for r in res], axis=0), np.mean([r[1] for r in res], axis=0)
        if base is None:
            base, against = nd, ""
        else:
            d, se = paired(nd, base, tu)[:2]
            against = f"{d:+.4f} ({se:.4f})"
        print(f"  {name:<22}{ep_p.get(name, ''):>8}{nd.mean():>9.4f}{rc.mean():>11.4f}{against:>22}")
    print("  (NDCG@10 and recall@10 over each user's launched products bought in the test week, ranking all "
          f"{new_p.sum()} launched products; against: paired difference in NDCG@10 against content similarity, "
          "standard error from 1,000 bootstrap draws of users; graph models are means over seeds 0, 1 and 2)  "
          f"[{time.time() - t0:.0f} s]")
