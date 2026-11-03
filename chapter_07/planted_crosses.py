"""Section 7.4 - What learned crosses find when the data has them: a planted experiment.

Tesselo's simulator (Chapter 2) decides every click and play from one dot product
of hidden taste vectors, plus clickbait and position. It has no context, so no
interaction between the request and the video. This separate experiment plants two
(its own seed; nothing else in the book uses it):
  network x video length - on a slow network (2G, 3G) a long video stalls, so long
                           videos lose and short ones gain; on wifi, length does
                           not matter (a dense feature crossed with a categorical one)
  taste group x category - each of 8 taste groups has its own liking for each of 8
                           categories (two categorical features crossed)
plus one hand-made-looking effect, the same language for app and video.

  logit = -2 + 1.0 * [same language] + T[group, category] - 1.0 * slow[network] * log(length)

Every model sees the same raw inputs (six ids and log length) and no hand-made
cross-feature. The number of epochs is chosen on a validation set. The measure is
the test log loss above the truth's own log loss (0 would be perfect).

Needs Python 3.12 or later and PyTorch. Run from the repository root (about three minutes):
    python chapter_07/planted_crosses.py
"""
import os
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "2"              # NumPy's linear algebra: the same thread count on every run
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rankers import CrossLayer, mlp  # noqa: E402
from heavy_ranking import impression_auc  # noqa: E402
from metrics import log_loss  # noqa: E402

FIELDS = {"app language": 6, "video language": 6, "network": 4, "phone tier": 3, "taste group": 8, "category": 8}
SLOW = np.array([1.0, 0.6, 0.2, 0.0])          # 2G, 3G, 4G, wifi
D = 8                                          # id vector length


def make_rows(n, rng, table):
    ids = {k: rng.integers(0, v, n) for k, v in FIELDS.items()}
    ids["video language"] = np.where(rng.random(n) < 0.4, ids["app language"], ids["video language"])
    length = np.exp(rng.uniform(np.log(0.25), np.log(10.0), n))   # minutes, 15 seconds to 10 minutes
    logit = (-2.0 + 1.0 * (ids["app language"] == ids["video language"])
             + table[ids["taste group"], ids["category"]]
             - 1.0 * SLOW[ids["network"]] * np.log(length))
    p = 1 / (1 + np.exp(-logit))
    y = rng.random(n) < p
    X = np.stack([ids[k] for k in FIELDS], axis=1)
    return X, np.log(length).astype(np.float32), y, p


class Model(nn.Module):
    def __init__(self, kind, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.kind = kind
        self.emb = nn.ModuleList([nn.Embedding(v, D) for v in FIELDS.values()])
        self.first = nn.ModuleList([nn.Embedding(v, 1) for v in FIELDS.values()])
        n, width = len(FIELDS), len(FIELDS) * D + 1
        if kind == "mlp":
            self.net = nn.Sequential(mlp([width, 64, 64]), nn.Linear(64, 1))
        elif kind == "deepfm":
            self.dense_vec = nn.Linear(1, D)
            self.net = nn.Sequential(mlp([width, 64, 64]), nn.Linear(64, 1))
        elif kind == "dlrm":
            self.dense_vec = nn.Linear(1, D)
            self.pairs = torch.triu_indices(n + 1, n + 1, offset=1)
            self.net = nn.Sequential(mlp([D + self.pairs.shape[1], 64, 64]), nn.Linear(64, 1))
        elif kind == "dcnv2":
            self.cross = nn.ModuleList([CrossLayer(width) for _ in range(2)])
            self.net = nn.Sequential(mlp([width, 64, 64]), nn.Linear(64, 1))
        self.w_dense = nn.Linear(1, 1)
        for t in list(self.emb) + list(self.first):
            nn.init.normal_(t.weight, 0.0, 0.05)              # small starting vectors, as in rankers.py

    def forward(self, X, length):
        x1 = length[:, None]
        linear = sum(t(X[:, k])[:, 0] for k, t in enumerate(self.first)) + self.w_dense(x1)[:, 0]
        if self.kind == "logistic":
            return linear
        V = torch.stack([t(X[:, k]) for k, t in enumerate(self.emb)], dim=1)
        flat = torch.cat([V.flatten(1), x1], dim=1)
        if self.kind == "mlp":
            return self.net(flat)[:, 0]
        if self.kind == "deepfm":
            W = torch.cat([V, self.dense_vec(x1)[:, None, :]], dim=1)
            s = W.sum(1)
            fm = 0.5 * ((s * s).sum(1) - (W * W).sum((1, 2)))
            return linear + fm + self.net(flat)[:, 0]
        if self.kind == "dlrm":
            d = self.dense_vec(x1)
            W = torch.cat([V, d[:, None, :]], dim=1)
            dots = torch.bmm(W, W.transpose(1, 2))[:, self.pairs[0], self.pairs[1]]
            return self.net(torch.cat([d, dots], dim=1))[:, 0]
        x = flat
        for layer in self.cross:
            x = layer(flat, x)
        return self.net(x)[:, 0]


def fit(kind, train, valid, max_epochs=10, seed=0):
    """Train one epoch at a time; keep the epoch with the lowest validation log loss."""
    model = Model(kind, seed)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    gen = torch.Generator().manual_seed(seed)
    X, L, y = (torch.as_tensor(a) for a in train)
    best, best_state, best_ep = np.inf, None, 0
    for ep in range(1, max_epochs + 1):
        perm = torch.randperm(len(y), generator=gen)
        for s in range(0, len(y), 512):
            i = perm[s:s + 512]
            loss = F.binary_cross_entropy_with_logits(model(X[i], L[i]), y[i].float())
            opt.zero_grad()
            loss.backward()
            opt.step()
        v = log_loss(predict(model, valid), valid[2])
        if v < best - 1e-9:
            best, best_state, best_ep = v, {k: t.clone() for k, t in model.state_dict().items()}, ep
    model.load_state_dict(best_state)
    return model, best_ep


def predict(model, rows):
    with torch.no_grad():
        return torch.sigmoid(model(torch.as_tensor(rows[0]), torch.as_tensor(rows[1]))).numpy()


if __name__ == "__main__":
    t0 = time.time()
    rng = np.random.default_rng(7_007)
    table = rng.normal(0.0, 0.8, (FIELDS["taste group"], FIELDS["category"]))
    test = make_rows(200_000, rng, table)
    valid = make_rows(50_000, rng, table)
    big = make_rows(500_000, rng, table)
    truth_loss = log_loss(test[3], test[2])
    print(f"Planted experiment: test rows {len(test[2]):,}, positive rate {test[2].mean():.3f}; "
          f"the truth's own log loss {truth_loss:.4f}, AUC {impression_auc(test[3], test[2]):.3f}")
    print(f"  {'':<10}" + "".join(f"{'trained on ' + f'{n:,}' + ' rows':>32}" for n in (20_000, 500_000)))
    print(f"  {'':<10}" + f"{'epochs':>8}{'loss above truth':>17}{'AUC':>7}" * 2)
    trained = {}
    for kind in ("logistic", "mlp", "deepfm", "dlrm", "dcnv2"):
        line = f"  {kind:<10}"
        for n in (20_000, 500_000):
            model, ep = fit(kind, tuple(a[:n] for a in big[:3]), valid)
            trained[kind] = model
            p = predict(model, test)
            line += f"{ep:>8}{log_loss(p, test[2]) - truth_loss:>17.4f}{impression_auc(p, test[2]):>7.3f}"
        print(line, flush=True)

    # What did each model learn about network x length? Mean predicted logit, short (1 minute or
    # less) minus long (5 minutes or more) videos, on 2G and on wifi, against the truth.
    rows = test
    short, long_ = rows[1] <= np.log(1.0), rows[1] >= np.log(5.0)
    net = rows[0][:, list(FIELDS).index("network")]
    logit = lambda p: np.log(p / (1 - p))                                    # noqa: E731
    print("\n  short minus long videos, mean logit (models trained on 500,000 rows):   on 2G   on wifi")
    for kind in ("truth", "logistic", "mlp", "deepfm", "dlrm", "dcnv2"):
        p = rows[3] if kind == "truth" else predict(trained[kind], rows)      # the models trained on 500,000
        vals = [logit(p[short & (net == g)]).mean() - logit(p[long_ & (net == g)]).mean() for g in (0, 3)]
        print(f"  {kind:<68}{vals[0]:>8.2f}{vals[1]:>9.2f}")
    print(f"[{time.time() - t0:.0f} s]")
