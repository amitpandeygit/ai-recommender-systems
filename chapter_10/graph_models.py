"""Graph models for Chapter 10 (sections 10.3 to 10.6).

  rp3beta(A, alpha, beta)  - Listing 10.2. Item-to-item scores from three-step random
                             walks on the user-item graph, with popular items
                             penalised (Paudel and colleagues, 2016)
  normalised_adjacency()   - D^-1/2 A D^-1/2 of a graph, as a sparse tensor
  graph_degrees()          - each node's degree in that same graph, every pair of
                             linked nodes counted once (training and request time
                             must use the same degrees)
  LightGCN                 - Listing 10.3. Free vectors for every node, propagated
                             over the graph without weights or non-linearities, and
                             the mean of the layers (He and colleagues, SIGIR 2020)
  fit_lightgcn()           - BPR training on the user-video edges, with SimGCL's
                             noise views as an option (Yu and colleagues, SIGIR 2022);
                             as in that paper, SimGCL leaves layer 0 out of every
                             final vector
  request_vectors()        - Listing 10.4. Request-time fold-in: a user's vector from
                             the nodes they link to now (videos clicked, accounts
                             followed) and the frozen, trained layers of those nodes
  check_fold_in()          - with the graph unchanged, request_vectors must rebuild
                             every user's trained vector exactly
  ials_fold_in()           - Chapter 4's iALS on the same edges, with each request's
                             user vector solved from its history
  two_tower_scores()       - Chapter 6's two-tower model (its chosen settings), with the
                             user tower reading each request's history

Nodes are numbered users first, then items, then any other nodes (creators,
knowledge-graph entities).
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import sparse

ROOT = Path(__file__).resolve().parent.parent
for _ch in ("chapter_03", "chapter_04", "chapter_06"):
    sys.path.insert(0, str(ROOT / _ch))
from implicit_mf import fold_in, ials  # noqa: E402

torch.set_num_threads(2)


# ---------------------------------------------------------------- section 10.3: random walks
def rp3beta(A, alpha=1.0, beta=0.5):
    """Listing 10.2. W[i, j]: the chance that a walk item i -> user -> item j ends at j,
    each step's probability raised to alpha, divided by (degree of j) ** beta.
    A is users x items, 1 where the user clicked the item."""
    A = sparse.csr_matrix(A, dtype=np.float64)
    d_user = np.asarray(A.sum(axis=1)).ravel()
    d_item = np.asarray(A.sum(axis=0)).ravel()
    P_ui = sparse.diags(1.0 / np.maximum(d_user, 1)) @ A        # user -> item: 1 / (items the user clicked)
    P_iu = sparse.diags(1.0 / np.maximum(d_item, 1)) @ A.T      # item -> user: 1 / (users who clicked it)
    P_ui.data **= alpha
    P_iu.data **= alpha
    Wm = (P_iu @ P_ui).toarray()                                 # item -> user -> item
    Wm /= np.maximum(d_item, 1)[None, :] ** beta                 # penalise popular end points
    np.fill_diagonal(Wm, 0.0)
    return Wm.astype(np.float32)


def walk_scores(Wm, H):
    """A request's score for every item: the sum of W[h, :] over the videos h in its history."""
    s = np.zeros((len(H), Wm.shape[1]), dtype=np.float32)
    for k, h in enumerate(H):
        h = np.unique(h[h >= 0])
        if len(h):
            s[k] = Wm[h].sum(axis=0)
    return s


# ---------------------------------------------------------------- section 10.4: LightGCN
def symmetric_adjacency(rows, cols, n_nodes):
    """The 0/1 adjacency of undirected edges (rows[k], cols[k]): an edge given twice, or
    in both directions (two users who follow each other), counts once."""
    r = np.r_[rows, cols]
    c = np.r_[cols, rows]
    A = sparse.coo_matrix((np.ones(len(r)), (r, c)), shape=(n_nodes, n_nodes)).tocsr()
    A.data[:] = 1.0
    return A


def graph_degrees(rows, cols, n_nodes):
    """Each node's number of neighbours in the graph that training propagates over."""
    return np.diff(symmetric_adjacency(rows, cols, n_nodes).indptr).astype(np.float64)


def normalised_adjacency(rows, cols, n_nodes):
    """Symmetric D^-1/2 A D^-1/2 for undirected edges (rows[k], cols[k]), and the degrees."""
    A = symmetric_adjacency(rows, cols, n_nodes)                 # an edge counts once
    deg = np.asarray(A.sum(axis=1)).ravel()
    inv = 1.0 / np.sqrt(np.maximum(deg, 1))
    Ah = (sparse.diags(inv) @ A @ sparse.diags(inv)).tocoo()
    idx = torch.as_tensor(np.vstack([Ah.row, Ah.col]), dtype=torch.long)
    A_hat = torch.sparse_coo_tensor(idx, torch.as_tensor(Ah.data, dtype=torch.float32), (n_nodes, n_nodes),
                                    check_invariants=True)
    return A_hat.coalesce(), deg


class LightGCN(torch.nn.Module):
    """Listing 10.3. e^(k+1) = A_hat e^(k); the final vector is the mean of e^(0) ... e^(K)
    (of e^(1) ... e^(K) with skip0, as SimGCL does)."""

    def __init__(self, n_nodes, dim=32, layers=2, seed=0, skip0=False):
        super().__init__()
        torch.manual_seed(seed)
        self.emb = torch.nn.Embedding(n_nodes, dim)
        torch.nn.init.normal_(self.emb.weight, std=0.1)
        self.layers = layers
        self.first = 1 if skip0 else 0                           # the first layer in the mean

    def propagate(self, A_hat, eps=0.0):
        """All layers. eps > 0 adds SimGCL's noise: at each layer, a random vector of
        length eps pointing into the same orthant as the node's own vector."""
        e = self.emb.weight
        out = [e]
        for _ in range(self.layers):
            e = torch.sparse.mm(A_hat, e)
            if eps > 0:
                e = e + torch.sign(e) * F.normalize(torch.rand_like(e), dim=1) * eps
            out.append(e)
        return out

    def final(self, A_hat, eps=0.0):
        return torch.stack(self.propagate(A_hat, eps)[self.first:], dim=0).mean(dim=0)


def info_nce(a, b, tau=0.2):
    """Contrastive loss: each row of a should be closest to the same row of b."""
    a, b = F.normalize(a, dim=1), F.normalize(b, dim=1)
    return F.cross_entropy(a @ b.T / tau, torch.arange(len(a)))


def fit_lightgcn(edges, n_nodes, n_users, video_nodes, layers=2, dim=32, epochs=100, batch=2_048, lr=0.005,
                 reg=1e-4, cl=0.0, eps=0.1, every=0, after=None, seed=0):
    """BPR on the user-video edges (one negative video per positive, uniform), with
    the whole graph propagated at every step. edges: all undirected edges of the
    graph (rows, cols); the user-video ones are those with a user end. cl > 0 adds
    SimGCL's loss: two noisy views of the batch's users and videos, pulled together,
    and leaves layer 0 out of every final vector, as the SimGCL paper does. Every
    `every` epochs it calls after(epoch, model, A_hat); if that returns True, training
    stops there (early stopping)."""
    rows, cols = edges
    A_hat, _ = normalised_adjacency(rows, cols, n_nodes)
    uv = (rows < n_users) & np.isin(cols, video_nodes)
    pu, pi = rows[uv], cols[uv]
    model = LightGCN(n_nodes, dim, layers, seed, skip0=cl > 0)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(pu))
        for s in range(0, len(order), batch):
            o = order[s:s + batch]
            u = torch.as_tensor(pu[o])
            i = torch.as_tensor(pi[o])
            j = torch.as_tensor(rng.choice(video_nodes, len(o)))
            E = model.final(A_hat)
            pos, neg = (E[u] * E[i]).sum(dim=1), (E[u] * E[j]).sum(dim=1)
            e0 = model.emb.weight
            loss = -F.logsigmoid(pos - neg).mean() + reg * (e0[u].pow(2) + e0[i].pow(2) + e0[j].pow(2)).sum() / len(o)
            if cl > 0:
                v1, v2 = model.final(A_hat, eps), model.final(A_hat, eps)
                uu, ii = torch.unique(u), torch.unique(i)
                loss = loss + cl * (info_nce(v1[uu], v2[uu]) + info_nce(v1[ii], v2[ii]))
            opt.zero_grad()
            loss.backward()
            opt.step()
        if every and epoch % every == 0 and after is not None and after(epoch, model, A_hat):
            break
    return model, A_hat


def layer_vectors(model, A_hat):
    """Every node's vector at every layer, and the final (mean) vectors, as NumPy arrays."""
    with torch.no_grad():
        L = [x.numpy().astype(np.float64) for x in model.propagate(A_hat)]
    return L, np.mean(L[model.first:], axis=0)


def request_vectors(L, deg, neighbours, own=None, first=0):
    """Listing 10.4. Request-time fold-in: each request's user vector, from the nodes it
    links to now and the frozen layers of those nodes.

    Layer k of a user is the sum, over the nodes n it links to, of layer k - 1 of n,
    divided by sqrt(its own number of links x n's number of links in the trained
    graph); the user vector is the mean of layers first ... K, layer 0 being the user's
    own trained vector (zero for a user the graph has not seen). neighbours: one array
    of node ids per request. own: each request's layer-0 vector, or None. deg: from
    graph_degrees, the same degrees training used."""
    K = len(L) - 1
    n_req = len(neighbours)
    rows = np.concatenate([np.full(len(nb), r) for r, nb in enumerate(neighbours)])
    cols = np.concatenate(neighbours).astype(np.int64)
    size = np.array([len(nb) for nb in neighbours], dtype=np.float64)
    w = 1.0 / np.sqrt(np.maximum(size[rows], 1) * np.maximum(deg[cols], 1))
    M = sparse.csr_matrix((w, (rows, cols)), shape=(n_req, L[0].shape[0]))
    layers = [np.zeros((n_req, L[0].shape[1])) if own is None else own]
    for k in range(1, K + 1):
        layers.append(M @ L[k - 1])
    return np.mean(layers[first:], axis=0)


def check_fold_in(model, A_hat, edges, n_nodes, users):
    """With the graph unchanged, request-time fold-in from each user's own neighbours in
    the trained graph must rebuild the user's trained vector."""
    A = symmetric_adjacency(*edges, n_nodes)
    deg = graph_degrees(*edges, n_nodes)
    L, final = layer_vectors(model, A_hat)
    nb = [A.indices[A.indptr[u]:A.indptr[u + 1]].astype(np.int64) for u in users]
    U = request_vectors(L, deg, nb, L[0][users], first=model.first)
    err = np.abs(U - final[users]).max()
    assert err < 1e-5, f"fold-in does not rebuild the trained vectors (max difference {err:.2e})"
    return err


def history_nodes(H, n_users, extra=None):
    """Item nodes of each request's history (unique, -1 padding dropped), plus any extra nodes."""
    out = []
    for k, h in enumerate(H):
        nb = np.unique(h[h >= 0]) + n_users
        if extra is not None:
            nb = np.r_[nb, extra[k]]
        out.append(nb.astype(np.int64))
    return out


# ---------------------------------------------------------------- the matrix-factorisation baselines
def ials_fold_in(A, H, k=16, reg=1.0, alpha=10.0):
    """Chapter 4's iALS (its settings) on the click edges; each request's user vector is
    solved from its own history with the item vectors fixed. Returns (request vectors, Y)."""
    _, Y = ials(np.asarray(A.todense(), dtype=np.float64), k, reg, alpha)
    YtY = Y.T @ Y
    U = np.zeros((len(H), k))
    for r, h in enumerate(H):
        h = np.unique(h[h >= 0])
        if len(h):
            U[r] = fold_in(Y, YtY, h, np.ones(len(h)), reg, alpha)
    return U, Y


def two_tower_scores(W, S, H, R_user):
    """Chapter 6's two-tower model with its chosen settings (two_tower.MAIN), trained on
    the snapshot's implicit scores; the user tower reads each request's own history
    (each clicked video weighted equally) and app language. Returns f(a, b) -> scores."""
    from two_tower import MAIN, history_rows, make_data, scores, train_two_tower
    from similarity_recs import item_features
    items, users = W["items"], W["users"]
    n_items = len(items["vec"])
    feats = item_features(items, n_items)
    data = make_data(S["weight"], items, users, feats, int(items["creator"].max()) + 1)
    pu, pi = np.nonzero(S["weight"] > 0)
    m = train_two_tower(data, pu, pi, negatives=MAIN["negatives"], logq=MAIN["logq"], bias=MAIN["bias"],
                        temperature=MAIN["temperature"], batch=MAIN["batch"], epochs=MAIN["epochs"])
    with torch.no_grad():
        V = m.item_tower(torch.arange(n_items), data["feats"], data["creator"])

    def f(a, b):
        hist = torch.zeros(b - a, n_items)
        for r in range(a, b):
            h = np.unique(H[r][H[r] >= 0])
            hist[r - a, h] = 1.0
        with torch.no_grad():
            U = m.user_tower(history_rows(hist, torch.arange(b - a)), data["lang"][torch.as_tensor(R_user[a:b])])
            return scores(U, V, m.temperature).numpy().astype(np.float64)
    return f
