"""Why some ranking architectures stop improving with more compute (section 12.3).

Chapter 7's three families of interaction layers, each grown in 5 steps, and measured
by held-out log loss against training compute per row (FLOPs counted by PyTorch's
FlopCounterMode on one forward pass, times 3 for the backward pass):

  MLP     - the embeddings side by side, then 3 hidden layers of width w
  DCN v2  - L full-rank cross layers (Listing 7.3), then an MLP of width w
  Wukong  - L Wukong layers (Listing 7.7): a factorisation-machine block, whose MLP has
            width w, beside a linear compression block; then an MLP of width w

Part 1, Tesselo's own rows. Chapter 7's heavy ranker, with its inputs and its click and
play heads, trained on the impressions of days 9 to 16 and checked on days 16 to 23 (up
to 5 epochs, the best kept), as in section 7.4. The loss is the click log loss plus the
play log loss of clicked rows. In Chapter 2's world a click depends on one dot product
of taste vectors, so a model needs no deep interactions here.

Part 2, a planted world with deep interactions (its own seed; nothing else uses it).
Each row has 12 fields, each with 20 values. The true click logit is

  -2.5 + sum of 6 tables of 2 fields + 4 tables of 3 fields + 2 tables of 4 fields

(the fields of each table, and its values, drawn at random; every table's values have
standard deviation 0.5). Fresh rows are drawn for every batch of 2,048, so no model sees
a row twice and none can overfit: 2,000 steps, about 4.1 million rows, the same for every
model. The measure is the test log loss above the truth's own log loss on 100,000 test rows
(0 is perfect), checked after 125, 250, 500, 1,000 and 2,000 steps.

Needs Python 3.12 or later and PyTorch. Run from the repository root (about 20 minutes):
    python chapter_12/ranker_scaling.py
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
from torch import nn
from torch.utils.flop_counter import FlopCounterMode

HERE = Path(__file__).resolve().parent
for _ch in ("chapter_07", "chapter_06", "chapter_09", "chapter_12"):
    sys.path.insert(0, str(HERE.parent / _ch))
from rankers import N_FIELDS, DCNv2, MLP, Ranker, Wukong, WukongLayer, mlp, predict, tensors, train  # noqa: E402
from ranking_data import DAY, Config, build, snapshot  # noqa: E402
from heavy_ranking import meta_for, rows_for  # noqa: E402
from metrics import log_loss  # noqa: E402
from seq_data import environment  # noqa: E402
from scaling_laws import fit_power_law  # noqa: E402

torch.set_num_threads(2)
STEPS = (1, 2, 3, 4, 5)
TESSELO = {"MLP": [(64,), (128,), (256,), (512,), (1024,)],
           "DCN v2": [(1, 64), (2, 128), (3, 256), (4, 512), (6, 1024)],
           "Wukong": [(1, 64), (2, 128), (3, 256), (4, 512), (6, 1024)]}
PLANTED = {"MLP": [(32,), (64,), (128,), (256,), (512,)],
           "DCN v2": [(1, 32), (2, 64), (3, 128), (4, 256), (6, 512)],
           "Wukong": [(1, 32), (2, 64), (3, 128), (4, 256), (6, 512)]}
FIELDS, VALUES, D, BATCH, N_STEPS = 12, 20, 16, 2_048, 2_000
CHECKS = (125, 250, 500, 1_000, 2_000)


def train_flops_per_row(model, example):
    """FLOPs of one forward pass per row, counted by PyTorch, times 3 for training."""
    with FlopCounterMode(display=False) as fc:
        with torch.no_grad():
            model(*example)
    n = len(example[0]["item"]) if isinstance(example[0], dict) else len(example[0])
    return 3 * fc.get_total_flops() / n


# ---------------------------------------------------------------- part 1: Chapter 7's ranker, grown
def tesselo_ranker(meta, family, size):
    r = Ranker(meta, "mlp", seed=0)
    if family == "MLP":
        w = size[0]
        r.body = MLP(meta, hidden=(w, w, w))
    elif family == "DCN v2":
        layers, w = size
        r.body = DCNv2(meta, n_cross=layers, hidden=(w, w))
    else:
        layers, w = size
        torch.manual_seed(0)
        r.body = Wukong(meta, n_layers=layers, k=4, hidden=(w, w))
        r.body.layers = nn.ModuleList([WukongLayer(N_FIELDS + 1, 4, hidden=w) for _ in range(layers)])
    torch.manual_seed(0)
    r.click, r.play = nn.Linear(r.body.out_dim, 1), nn.Linear(r.body.out_dim, 1)
    return r


def tesselo_part(t0):
    cfg = Config()
    users, items, served, events, orders, truth = build(cfg)
    snaps = {d: snapshot(events, served, users, items, d * DAY) for d in (9, 16)}
    w1 = rows_for(events, served, users, items, [(9, 16)])
    w2 = rows_for(events, served, users, items, [(16, 23)])
    b1 = tensors([snaps[9]], users, items, w1["user"], w1["item"], w1["snap"])
    b2 = tensors([snaps[16]], users, items, w2["user"], w2["item"], w2["snap"])
    meta = meta_for(users, items, b1["dense"].numpy())
    print(f"Part 1, Tesselo's rows (Chapter 7): {len(w1['click']):,} training impressions (days 9 to 16), "
          f"{len(w2['click']):,} validation impressions (days 16 to 23)  [{time.time() - t0:.0f} s]")
    print(f"  {'':<26}{'FLOPs/row':>11}  validation loss after each epoch{'':>12}{'best':>9}")
    out = {}
    for family, sizes in TESSELO.items():
        for size in sizes:
            model = tesselo_ranker(meta, family, size)
            idx = torch.arange(256)
            ex = ({k: v[idx] for k, v in b1.items()},)
            flops = train_flops_per_row(model, ex)
            losses = []

            def check(e):
                pc, pp = predict(model, b2, len(w2["click"]), position=torch.as_tensor(w2["position"]))
                c = w2["click"]
                losses.append(log_loss(pc, c) + log_loss(pp[c], w2["play"][c]))
            train(model, b1, w1, epochs=max(STEPS), seed=0, after_epoch=check)
            out[family, size] = (flops, min(losses))
            label = f"{family}, " + (f"width {size[0]}" if family == "MLP"
                                      else f"{size[0]} layer{'s' * (size[0] > 1)}, width {size[1]}")
            print(f"  {label:<26}{flops:>11,.0f}  " + " ".join(f"{x:.4f}" for x in losses)
                  + f"{min(losses):>9.4f}  [{time.time() - t0:.0f} s]", flush=True)
    return out


# ---------------------------------------------------------------- part 2: a planted world with deep interactions
class World:
    def __init__(self, seed=12_300):
        rng = np.random.default_rng(seed)
        self.tables = []
        for order, count in ((2, 6), (3, 4), (4, 2)):
            for _ in range(count):
                fields = np.sort(rng.choice(FIELDS, order, replace=False))
                self.tables.append((fields, rng.normal(0, 0.5, (VALUES,) * order)))

    def logit(self, X):
        z = np.full(len(X), -2.5)
        for fields, T in self.tables:
            z += T[tuple(X[:, f] for f in fields)]
        return z

    def rows(self, n, rng):
        X = rng.integers(0, VALUES, (n, FIELDS))
        p = 1 / (1 + np.exp(-self.logit(X)))
        return X, (rng.random(n) < p).astype(np.float32), p


class Planted(nn.Module):
    """Field embeddings (12 x 16) into one of the three families, then a logit."""
    def __init__(self, family, size, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.emb = nn.Embedding(FIELDS * VALUES, D)
        self.family = family
        flat = FIELDS * D
        if family == "MLP":
            w = size[0]
            self.body = mlp([flat, w, w, w])
        elif family == "DCN v2":
            layers, w = size
            self.cross = nn.ModuleList([nn.Linear(flat, flat) for _ in range(layers)])
            for c in self.cross:
                nn.init.zeros_(c.weight)
                nn.init.zeros_(c.bias)
            self.body = mlp([flat, w, w])
        else:
            layers, w = size
            k = FIELDS // 2
            self.layers = nn.ModuleList()
            for _ in range(layers):
                lay = nn.Module()
                lay.fmb = nn.Sequential(mlp([FIELDS * FIELDS, w]), nn.Linear(w, k * D))
                lay.lcb = nn.Linear(FIELDS, FIELDS - k, bias=False)
                lay.norm = nn.LayerNorm(D)
                self.layers.append(lay)
            self.k = k
            self.body = mlp([flat, w, w])
        self.out = nn.Linear(w, 1)

    def forward(self, X):
        E = self.emb(X + torch.arange(FIELDS) * VALUES)                     # batch x 12 x 16
        if self.family == "DCN v2":
            x0 = x = E.flatten(1)
            for c in self.cross:
                x = x0 * c(x) + x
            h = x
        elif self.family == "Wukong":
            for lay in self.layers:
                fm = lay.fmb(torch.bmm(E, E.transpose(1, 2)).flatten(1)).view(len(E), self.k, D)
                lc = lay.lcb(E.transpose(1, 2)).transpose(1, 2)
                E = lay.norm(torch.cat([fm, lc], dim=1) + E)
            h = E.flatten(1)
        else:
            h = E.flatten(1)
        return self.out(self.body(h))[:, 0]


def planted_part(t0):
    world = World()
    Xt, yt, pt = world.rows(100_000, np.random.default_rng(1))
    truth = log_loss(pt, yt.astype(bool))
    print(f"\nPart 2, the planted world: 12 fields of 20 values; 6 + 4 + 2 interaction tables; test rows 100,000 "
          f"(click rate {yt.mean():.3f}); the truth's own test log loss {truth:.4f}")
    print(f"  {'':<26}{'FLOPs/row':>11}  excess test loss after {', '.join(f'{c:,}' for c in CHECKS)} steps")
    out = {}
    Xt_t = torch.as_tensor(Xt)
    for family, sizes in PLANTED.items():
        for size in sizes:
            model = Planted(family, size)
            flops = train_flops_per_row(model, (Xt_t[:256],))
            rng = np.random.default_rng(2)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            curve = []
            for step in range(1, N_STEPS + 1):
                X, y, _ = world.rows(BATCH, rng)
                L = F.binary_cross_entropy_with_logits(model(torch.as_tensor(X)), torch.as_tensor(y))
                opt.zero_grad()
                L.backward()
                opt.step()
                if step in CHECKS:
                    with torch.no_grad():
                        p = torch.sigmoid(model(Xt_t)).numpy()
                    curve.append((flops * step * BATCH, log_loss(p, yt.astype(bool)) - truth))
            out[family, size] = (flops, curve)
            label = f"{family}, " + (f"width {size[0]}" if family == "MLP"
                                      else f"{size[0]} layer{'s' * (size[0] > 1)}, width {size[1]}")
            print(f"  {label:<26}{flops:>11,.0f}  " + " ".join(f"{v:.4f}" for _, v in curve)
                  + f"  [{time.time() - t0:.0f} s]", flush=True)
    print("\n  Excess loss after 2,000 steps against training FLOPs per row (power law in FLOPs per row):")
    for family in PLANTED:
        rs = [out[family, s] for s in PLANTED[family]]
        x, y = [f for f, _ in rs], [c[-1][1] for _, c in rs]
        l_inf, a, alpha = fit_power_law(x, y)
        print(f"  {family}: " + ", ".join(f"{f:,.0f}: {v:.4f}" for f, v in zip(x, y))
              + f"; fit L = {l_inf:.4f} + {a:.4g} x F^-{alpha:.4f}")
    return out


if __name__ == "__main__":
    t0 = time.time()
    print(environment())
    parts = sys.argv[1:] or ["tesselo", "planted"]
    if "tesselo" in parts:
        tesselo_part(t0)
    if "planted" in parts:
        planted_part(t0)
    print(f"[{time.time() - t0:.0f} s]")
