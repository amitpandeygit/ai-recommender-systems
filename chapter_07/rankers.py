"""Listings 7.2, 7.3, 7.4, 7.6 and 7.7 - Tesselo's rankers in PyTorch (sections 7.3 to 7.8).

Every ranker reads the same inputs (ranking_data.py) and ends in the same heads:
  click       - P(click | the video was on screen in slot 1), with a shallow
                position tower added during training (section 7.3)
  play        - P(satisfied play | click): a completion or a like
  watch       - share of the video watched, given a click (multi-task models only)
A video's score is P(click) x P(play | click), the expected satisfied plays
of showing it, so the score and Chapter 5's goal measure are the same thing.

The interaction layers differ (section 7.4):
  logistic    - a linear model over the dense features and one weight per id;
                no learned interactions, only the hand-made cross-features
  mlp, no ids - an MLP on the eight dense features only, with no id vectors
  mlp         - all vectors concatenated, then a plain multi-layer perceptron
  wide&deep   - the logistic model (plus two crossed ids) beside an MLP (history box)
  deepfm      - a factorisation machine beside an MLP, sharing the vectors (history box)
  dlrm        - a dot product between every pair of vectors, then an MLP
  dcnv2       - cross layers x_{l+1} = x_0 * (W x_l + b) + x_l, then an MLP
  wukong      - stacked factorisation-machine and linear-compression blocks (section 7.8)
and the multi-task models of section 7.5 share their tasks in three ways: one
shared bottom, MMoE (experts mixed by a gate per task) and PLE (shared and
per-task experts).

Section 7.7's item tables (full, hashed, quotient-remainder, rare ids shared)
are in ItemTable (Listing 7.6), and section 7.6's pre-ranker inputs in PreRanker.
This is a laptop-scale implementation for teaching (CPU only).

Needs Python 3.12 or later and PyTorch. Imported by the other Chapter 7 scripts.
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ranking_data import DENSE, pair_features

DIM = 16                      # length of every id vector
torch.set_num_threads(2)      # the same thread count on every machine
N_FIELDS = 6                  # user, video, creator, category, video language, app language


# ---------------------------------------------------------------- section 7.7: item tables
def hash_rows(ids, rows):
    """Multiplicative (Fibonacci) hashing: multiply by 2,654,435,761, keep the low 32
    bits, and scale them to 0 .. rows - 1. Works on tensors and on NumPy arrays."""
    return ((ids * 2_654_435_761) % 2 ** 32) * rows // 2 ** 32


class ItemTable(nn.Module):
    """Listing 7.6. Video id -> vector. full: one row per id. hash: ids share
    `rows` rows by a hash, so some videos collide. qr: the quotient-remainder
    trick, two small tables whose rows are multiplied, so every id gets its own
    pair of rows. rare: ids seen fewer than `min_count` times share one row."""
    def __init__(self, n_items, dim, mode="full", rows=None, counts=None, min_count=0):
        super().__init__()
        self.mode = mode
        if mode == "full":
            self.table = nn.Embedding(n_items, dim)
        elif mode == "hash":
            self.rows = rows
            self.table = nn.Embedding(rows, dim)
        elif mode == "qr":
            self.rows = rows                                       # remainder table size
            self.q = nn.Embedding(-(-n_items // rows), dim)
            self.r = nn.Embedding(rows, dim)
        elif mode == "rare":
            keep = np.flatnonzero(counts >= min_count)
            index = np.full(n_items, len(keep))                    # every rare id -> the last row
            index[keep] = np.arange(len(keep))
            self.register_buffer("index", torch.as_tensor(index))
            self.table = nn.Embedding(len(keep) + 1, dim)
        for p in self.parameters():
            nn.init.normal_(p, 0.0, 0.05)
        if mode == "qr":
            nn.init.normal_(self.q.weight, 1.0, 0.05)              # a product starts near the remainder row

    def forward(self, ids):
        if self.mode == "full":
            return self.table(ids)
        if self.mode == "hash":
            return self.table(hash_rows(ids, self.rows))
        if self.mode == "qr":
            return self.q(ids // self.rows) * self.r(ids % self.rows)
        return self.table(self.index[ids])

    def n_rows(self):
        return sum(p.shape[0] for p in self.parameters())


class Inputs(nn.Module):
    """The six vectors of a (user, video) pair, and the standardised dense features.
    The user vector is the weighted mean of their history videos' vectors, from the
    same table as the candidate video's vector."""
    def __init__(self, meta, item_table=None, vectors=True):
        super().__init__()
        if vectors:                                              # a model without id vectors gets no tables
            self.item = item_table or ItemTable(meta["n_items"], DIM)
            self.creator = nn.Embedding(meta["n_creators"], DIM)
            self.category = nn.Embedding(meta["n_categories"], DIM)
            self.language = nn.Embedding(meta["n_languages"], DIM)  # one table for video and app language
            for t in (self.creator, self.category, self.language):
                nn.init.normal_(t.weight, 0.0, 0.05)
        self.register_buffer("mean", torch.as_tensor(meta["dense_mean"]))
        self.register_buffer("sd", torch.as_tensor(meta["dense_sd"]))

    def dense(self, b):
        return (b["dense"] - self.mean) / self.sd

    def forward(self, b):
        users, inv = torch.unique(b["u"], return_inverse=True)      # pool each user once per batch
        h, w = b["H"][users], b["HW"][users]
        pooled = (self.item(h) * w[..., None]).sum(1) / w.sum(1, keepdim=True).clamp_min(1e-9)
        vectors = torch.stack([pooled[inv], self.item(b["item"]), self.creator(b["creator"]),
                               self.category(b["category"]), self.language(b["ilang"]),
                               self.language(b["ulang"])], dim=1)  # batch x 6 x DIM
        return vectors, self.dense(b)


def mlp(sizes):
    layers = []
    for a, c in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, c), nn.ReLU()]
    return nn.Sequential(*layers)


# ---------------------------------------------------------------- section 7.4: interaction layers
class Logistic(nn.Module):
    """No learned interactions: dense features plus one learned weight per id value."""
    uses_vectors = False

    def __init__(self, meta):
        super().__init__()
        self.w = nn.ModuleList([nn.Embedding(meta[k], 1) for k in
                                ("n_items", "n_creators", "n_categories", "n_languages", "n_languages")])
        self.out_dim = len(DENSE) + len(self.w)

    def forward(self, b, vectors, dense):
        ids = (b["item"], b["creator"], b["category"], b["ilang"], b["ulang"])
        return torch.cat([dense] + [t(i) for t, i in zip(self.w, ids)], dim=1)


class MLP(nn.Module):
    def __init__(self, meta, hidden=(128, 64)):
        super().__init__()
        self.net = mlp([N_FIELDS * DIM + len(DENSE), *hidden])
        self.out_dim = hidden[-1]

    def forward(self, b, vectors, dense):
        return self.net(torch.cat([vectors.flatten(1), dense], dim=1))


class DenseMLP(nn.Module):
    """An MLP on the dense features only: no id vectors at all."""
    uses_vectors = False

    def __init__(self, meta, hidden=(128, 64)):
        super().__init__()
        self.net = mlp([len(DENSE), *hidden])
        self.out_dim = hidden[-1]

    def forward(self, b, vectors, dense):
        return self.net(dense)


class WideDeep(nn.Module):
    """History box: a wide linear part with crossed ids beside a deep MLP."""
    def __init__(self, meta):
        super().__init__()
        self.wide = Logistic(meta)
        n_l, n_c = meta["n_languages"], meta["n_categories"]
        self.cross = nn.Embedding(n_l * n_c + n_l * n_l, 1)        # app language x category, x video language
        self.n_l, self.n_c = n_l, n_c
        self.deep = MLP(meta)
        self.out_dim = self.wide.out_dim + 2 + self.deep.out_dim

    def forward(self, b, vectors, dense):
        a = b["ulang"] * self.n_c + b["category"]
        c = self.n_l * self.n_c + b["ulang"] * self.n_l + b["ilang"]
        return torch.cat([self.wide(b, vectors, dense), self.cross(a), self.cross(c),
                          self.deep(b, vectors, dense)], dim=1)


class DeepFM(nn.Module):
    """History box: a factorisation machine (every pair of vectors, dotted, summed)
    beside an MLP; both read the same vectors."""
    def __init__(self, meta):
        super().__init__()
        self.first = Logistic(meta)
        self.deep = MLP(meta)
        self.out_dim = self.first.out_dim + 1 + self.deep.out_dim

    def forward(self, b, vectors, dense):
        s = vectors.sum(1)
        fm = 0.5 * ((s * s).sum(1) - (vectors * vectors).sum((1, 2)))   # sum of all pairwise dot products
        return torch.cat([self.first(b, vectors, dense), fm[:, None], self.deep(b, vectors, dense)], dim=1)


class DLRM(nn.Module):
    """Listing 7.2. Dense features -> one more vector (bottom MLP); a dot product
    between every pair of the seven vectors; the dots and the dense vector -> top MLP."""
    def __init__(self, meta, hidden=(128, 64)):
        super().__init__()
        self.bottom = nn.Sequential(mlp([len(DENSE), 32]), nn.Linear(32, DIM))
        n = N_FIELDS + 1
        self.pairs = torch.triu_indices(n, n, offset=1)             # 21 pairs of 7 vectors
        self.top = mlp([DIM + self.pairs.shape[1], *hidden])
        self.out_dim = hidden[-1]

    def forward(self, b, vectors, dense):
        d = self.bottom(dense)
        T = torch.cat([vectors, d[:, None, :]], dim=1)               # batch x 7 x DIM
        dots = torch.bmm(T, T.transpose(1, 2))[:, self.pairs[0], self.pairs[1]]
        return self.top(torch.cat([d, dots], dim=1))


class CrossLayer(nn.Module):
    """Listing 7.3. One DCN v2 cross layer: x_{l+1} = x_0 * (W x_l + b) + x_l."""
    def __init__(self, width):
        super().__init__()
        self.linear = nn.Linear(width, width)
        nn.init.zeros_(self.linear.weight)                           # each layer starts as the identity
        nn.init.zeros_(self.linear.bias)

    def forward(self, x0, x):
        return x0 * self.linear(x) + x


class DCNv2(nn.Module):
    """Stacked DCN v2: three cross layers, then an MLP on their output."""
    def __init__(self, meta, n_cross=3, hidden=(128, 64)):
        super().__init__()
        width = N_FIELDS * DIM + len(DENSE)
        self.cross = nn.ModuleList([CrossLayer(width) for _ in range(n_cross)])
        self.deep = mlp([width, *hidden])
        self.out_dim = hidden[-1]

    def forward(self, b, vectors, dense):
        x0 = x = torch.cat([vectors.flatten(1), dense], dim=1)
        for layer in self.cross:
            x = layer(x0, x)
        return self.deep(x)


class WukongLayer(nn.Module):
    """Listing 7.7 (section 7.8). A factorisation-machine block (all pairwise dots
    of the n input vectors -> MLP -> k new vectors) beside a linear compression
    block (l new vectors, each a weighted sum of the inputs). k + l = n, so the
    output has the same shape, with a residual connection and layer norm."""
    def __init__(self, n, k, hidden=64):
        super().__init__()
        self.k = k
        self.fmb = nn.Sequential(mlp([n * n, hidden]), nn.Linear(hidden, k * DIM))
        self.lcb = nn.Linear(n, n - k, bias=False)
        self.norm = nn.LayerNorm(DIM)

    def forward(self, X):
        inter = torch.bmm(X, X.transpose(1, 2)).flatten(1)            # batch x n*n
        fm = self.fmb(inter).view(len(X), self.k, DIM)
        lc = self.lcb(X.transpose(1, 2)).transpose(1, 2)              # batch x (n - k) x DIM
        return self.norm(torch.cat([fm, lc], dim=1) + X)


class Wukong(nn.Module):
    def __init__(self, meta, n_layers=2, k=4, hidden=(128, 64)):
        super().__init__()
        n = N_FIELDS + 1
        self.bottom = nn.Sequential(mlp([len(DENSE), 32]), nn.Linear(32, DIM))
        self.layers = nn.ModuleList([WukongLayer(n, k) for _ in range(n_layers)])
        self.top = mlp([n * DIM, *hidden])
        self.out_dim = hidden[-1]

    def forward(self, b, vectors, dense):
        X = torch.cat([vectors, self.bottom(dense)[:, None, :]], dim=1)
        for layer in self.layers:
            X = layer(X)
        return self.top(X.flatten(1))


class PreRanker(nn.Module):
    """Section 7.6: the pre-ranker, a small MLP (16 hidden units) on five cheap
    columns: the iALS score (one dot product of stored vectors), three numbers
    stored per video (popularity, click rate, plays per click), and whether the
    video is in the app language (one comparison). It leaves out the per-user
    aggregates. Not in ARCHITECTURES."""
    uses_vectors = False
    COLUMNS = [0, 1, 2, 3, 4]

    def __init__(self, meta, hidden=16):
        super().__init__()
        self.net = mlp([len(self.COLUMNS), hidden])
        self.out_dim = hidden

    def forward(self, b, vectors, dense):
        return self.net(dense[:, self.COLUMNS])


ARCHITECTURES = {"logistic": Logistic, "mlp, no ids": DenseMLP, "mlp": MLP, "wide&deep": WideDeep, "deepfm": DeepFM,
                 "dlrm": DLRM, "dcnv2": DCNv2, "wukong": Wukong}


class Ranker(nn.Module):
    """Inputs -> interaction layers -> a click head and a play head. The shallow
    position tower adds one learned number per slot to the click logit; serving
    uses slot 1 for every candidate (position=None)."""
    def __init__(self, meta, arch="dlrm", position=True, item_table=None, seed=0):
        super().__init__()
        torch.manual_seed(seed)                                  # the same starting weights on every run
        self.body = PreRanker(meta) if arch == "pre-ranker" else ARCHITECTURES[arch](meta)
        self.inputs = Inputs(meta, item_table, vectors=getattr(self.body, "uses_vectors", True))
        self.click = nn.Linear(self.body.out_dim, 1)
        self.play = nn.Linear(self.body.out_dim, 1)
        self.position = nn.Embedding(11, 1) if position else None
        if position:
            nn.init.zeros_(self.position.weight)

    def forward(self, b, position=None):
        if getattr(self.body, "uses_vectors", True):
            vectors, dense = self.inputs(b)
        else:                                                    # no id vectors: skip the lookups
            vectors, dense = None, self.inputs.dense(b)
        z = self.body(b, vectors, dense)
        click = self.click(z)[:, 0]
        if self.position is not None:
            slot = position if position is not None else torch.ones_like(b["item"])
            click = click + self.position(slot)[:, 0] - self.position.weight[1, 0]
        return {"click": click, "play": self.play(z)[:, 0]}


# ---------------------------------------------------------------- section 7.5: multi-task
class Expert(nn.Module):
    """One expert: a one-layer MLP on the shared input."""
    def __init__(self, width, out):
        super().__init__()
        self.net = mlp([width, out])

    def forward(self, x):
        return self.net(x)


class MultiTask(nn.Module):
    """Listing 7.4. Tasks: click (all rows), play and watch share (clicked rows).
    Every model reads the eight dense features only.
    shared: one bottom for every task. mmoe: n_experts experts, mixed by one
    softmax gate per task. ple: per level, shared experts plus experts owned by each
    task; a task's gate mixes its own and the shared ones (two levels)."""
    TASKS = ("click", "play", "watch")

    def __init__(self, meta, kind="mmoe", n_experts=4, expert_dim=64, levels=2, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.kind = kind
        self.inputs = Inputs(meta, vectors=False)
        width = len(DENSE)
        T = len(self.TASKS)
        if kind == "shared":
            self.bottom = mlp([width, 128, expert_dim])
        elif kind == "mmoe":
            self.experts = nn.ModuleList([Expert(width, expert_dim) for _ in range(n_experts)])
            self.gates = nn.ModuleList([nn.Linear(width, n_experts) for _ in range(T)])
        elif kind == "ple":
            self.levels = nn.ModuleList()
            w = width
            for _ in range(levels):
                lvl = nn.Module()
                lvl.shared = nn.ModuleList([Expert(w, expert_dim) for _ in range(2)])
                lvl.own = nn.ModuleList([nn.ModuleList([Expert(w, expert_dim) for _ in range(2)])
                                         for _ in range(T)])
                lvl.gates = nn.ModuleList([nn.Linear(w, 4) for _ in range(T)])
                lvl.shared_gate = nn.Linear(w, 2 + 2 * T)               # feeds the next level's shared input
                self.levels.append(lvl)
                w = expert_dim
        self.towers = nn.ModuleList([nn.Sequential(mlp([expert_dim, 32]), nn.Linear(32, 1)) for _ in range(T)])
        self.position = nn.Embedding(11, 1)
        nn.init.zeros_(self.position.weight)

    def forward(self, b, position=None):
        x = self.inputs.dense(b)
        if self.kind == "shared":
            h = [self.bottom(x)] * len(self.TASKS)
        elif self.kind == "mmoe":
            e = torch.stack([ex(x) for ex in self.experts], dim=1)          # batch x E x dim
            h = [(F.softmax(g(x), dim=1)[..., None] * e).sum(1) for g in self.gates]
        else:
            task_in, shared_in = [x] * len(self.TASKS), x
            for lvl in self.levels:
                s = [ex(shared_in) for ex in lvl.shared]
                own = [[ex(task_in[t]) for ex in lvl.own[t]] for t in range(len(self.TASKS))]
                task_out = []
                for t, g in enumerate(lvl.gates):
                    e = torch.stack(own[t] + s, dim=1)
                    task_out.append((F.softmax(g(task_in[t]), dim=1)[..., None] * e).sum(1))
                every = torch.stack([o for ot in own for o in ot] + s, dim=1)
                shared_in = (F.softmax(lvl.shared_gate(shared_in), dim=1)[..., None] * every).sum(1)
                task_in = task_out
            h = task_in
        out = {name: tower(hh)[:, 0] for name, tower, hh in zip(self.TASKS, self.towers, h)}
        slot = position if position is not None else torch.ones_like(b["item"])
        out["click"] = out["click"] + self.position(slot)[:, 0] - self.position.weight[1, 0]
        return out


# ---------------------------------------------------------------- data and training
def meta_for(users, items, dense):
    return {"n_items": len(items["vec"]), "n_creators": int(items["creator"].max()) + 1,
            "n_categories": int(items["category"].max()) + 1, "n_languages": int(items["language"].max()) + 1,
            "dense_mean": dense.mean(0), "dense_sd": dense.std(0) + 1e-6}


def tensors(snapshots, users, items, u, i, snap_index):
    """Model inputs for pairs (u[k], i[k]) whose features come from snapshot
    snap_index[k]. Each snapshot's history table is stacked, so a row's user
    index is snapshot * n_users + user."""
    n_users = len(users["vec"])
    dense = np.zeros((len(u), len(DENSE)), dtype=np.float32)
    for s, S in enumerate(snapshots):
        m = snap_index == s
        dense[m] = pair_features(S, users, items, u[m], i[m])
    return {"u": torch.as_tensor(snap_index * n_users + u), "item": torch.as_tensor(i),
            "creator": torch.as_tensor(items["creator"][i]), "category": torch.as_tensor(items["category"][i]),
            "ilang": torch.as_tensor(items["language"][i]), "ulang": torch.as_tensor(users["language"][u]),
            "dense": torch.as_tensor(dense),
            "H": torch.as_tensor(np.concatenate([S["hist_id"] for S in snapshots])),
            "HW": torch.as_tensor(np.concatenate([S["hist_w"] for S in snapshots]))}


ROW_KEYS = ("u", "item", "creator", "category", "ilang", "ulang", "dense")


def take(b, idx):
    out = {k: b[k][idx] for k in ROW_KEYS}
    out["H"], out["HW"] = b["H"], b["HW"]
    return out


def train(model, b, labels, epochs=4, lr=2e-3, batch=1024, seed=0, tasks=("click", "play"), after_epoch=None):
    """Click loss on every row; play (and watch) loss only on clicked rows. `tasks`
    names the losses used, so a model can be trained for one task alone.
    after_epoch(e), if given, is called after epoch e (1, 2, ...), so a validation
    check sees exactly the model that `epochs=e` would return: one optimiser, with
    its state, runs through all the epochs."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    y = {k: torch.as_tensor(labels[k], dtype=torch.float32) for k in ("click", "play", "watch")}
    pos = torch.as_tensor(labels["position"])
    n = len(y["click"])
    model.train()
    for e in range(1, epochs + 1):
        perm = torch.randperm(n, generator=gen)
        for start in range(0, n, batch):
            idx = perm[start:start + batch]
            out = model(take(b, idx), position=pos[idx])
            loss = torch.zeros(())
            if "click" in tasks:
                loss = loss + F.binary_cross_entropy_with_logits(out["click"], y["click"][idx])
            c = y["click"][idx] > 0
            if c.any() and "play" in tasks:
                loss = loss + F.binary_cross_entropy_with_logits(out["play"][c], y["play"][idx][c])
            if c.any() and "watch" in tasks:
                loss = loss + F.mse_loss(torch.sigmoid(out["watch"][c]), y["watch"][idx][c])
            if not loss.requires_grad:                       # e.g. a play-only model and a batch with no click
                continue
            opt.zero_grad()
            loss.backward()
            opt.step()
        if after_epoch is not None:
            model.eval()
            after_epoch(e)
            model.train()
    model.eval()
    return model


def predict(model, b, n, position=None, batch=20_000):
    """P(click) and P(play | click) for n rows; slot 1 unless a position is given."""
    pc, pp = np.empty(n, dtype=np.float32), np.empty(n, dtype=np.float32)
    with torch.no_grad():
        for start in range(0, n, batch):
            idx = torch.arange(start, min(start + batch, n))
            out = model(take(b, idx), position=None if position is None else position[idx])
            pc[idx] = torch.sigmoid(out["click"]).numpy()
            pp[idx] = torch.sigmoid(out["play"]).numpy()
    return pc, pp


def score_pools(model, S, users, items, rows, pools, chunk=256):
    """Expected satisfied plays, P(click) x P(play | click), for every candidate in
    each user's pool; -inf where the pool has no item."""
    out = np.full(pools.shape, -np.inf)
    for start in range(0, len(rows), chunk):
        r = rows[start:start + chunk]
        p = pools[start:start + chunk]
        uu = np.repeat(r, p.shape[1])
        ii = p.ravel()
        ok = ii >= 0
        b = tensors([S], users, items, uu[ok], ii[ok], np.zeros(ok.sum(), dtype=np.int64))
        pc, pp = predict(model, b, int(ok.sum()))
        s = np.full(len(ii), -np.inf)
        s[ok] = pc.astype(np.float64) * pp
        out[start:start + chunk] = s.reshape(p.shape)
    return out


def rerank(pools, scores, k):
    """Each user's top k of their pool by score (stable: ties keep the pool's order)."""
    order = np.argsort(-scores, axis=1, kind="stable")[:, :k]
    top = np.take_along_axis(pools, order, axis=1)
    return np.where(np.take_along_axis(scores, order, axis=1) > -np.inf, top, -1)
