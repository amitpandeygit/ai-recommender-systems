"""Listing 8.4 - A small cross-encoder for reranking, against a bi-encoder (section 8.5).

Both models read the same tokens. A token is the sum of learned vectors for a
video's id, creator, category and language:
  the candidate         - its token, plus its three stored statistics (popularity,
                          click rate, plays per click)
  the user's history    - the tokens of their 10 highest-weighted videos, each with its weight
  the request           - the app language, and the size of the user's history
The bi-encoder (two towers, Chapter 6) encodes the request and the history on one
side and the candidate on the other, and scores their dot product, so every
candidate vector can be computed ahead and stored. The cross-encoder puts all the
tokens into one sequence through two self-attention layers, so every history token
can attend to the candidate and back; the token vectors can be stored, but the
attention layers and the score must run once for every (user, candidate) pair. A third
model is the cross-encoder with the eight dense features of Chapter 7 added to its
output layer, as a production reranker would have them.

All three are trained like Chapter 7's ranker: a click head (with the shallow
position tower) and a play head, epochs chosen on the validation week (1 to 4). On
the test week each reorders the same 50 candidates the shipped pre-ranker keeps,
from seeds 0, 1 and 2, and each seed is compared with the shipped heavy ranker
retrained from the same seed. The time to score 50 candidates for one request is
measured. Pretrained text cross-encoders, for search and the assistant, come with
Chapters 18 and 19; this one is trained from scratch on Tesselo's logs.

Needs Python 3.12 or later and PyTorch. Run from the repository root (about 12 minutes):
    python chapter_08/cross_encoder.py
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from slate_data import (KEPT, SHIP, Ranker, order_by, row_metrics, seed_paired, split, tensors, train,  # noqa: E402
                        world)
from rankers import score_pools, take  # noqa: E402

D = 32                        # length of every token vector
HIST = 10                     # history tokens per user
EPOCHS = (1, 2, 3, 4)
SEEDS = (0, 1, 2)


class Tokens(nn.Module):
    """Token vectors for videos (id + creator + category + language), the request
    and the history weights."""
    def __init__(self, meta, items):
        super().__init__()
        self.item = nn.Embedding(meta["n_items"], D)
        self.creator = nn.Embedding(meta["n_creators"], D)
        self.category = nn.Embedding(meta["n_categories"], D)
        self.lang = nn.Embedding(meta["n_languages"], D)
        self.ulang = nn.Embedding(meta["n_languages"], D)
        self.stats = nn.Linear(3, D)                              # candidate: popularity, click rate, plays per click
        self.hist_w = nn.Linear(1, D)                             # history: the weight of each video
        self.req = nn.Linear(2, D)                                # request: history size, has history
        self.kind = nn.Embedding(3, D)                            # which kind of token: request, candidate, history
        self.register_buffer("i_creator", torch.as_tensor(items["creator"]))
        self.register_buffer("i_category", torch.as_tensor(items["category"]))
        self.register_buffer("i_lang", torch.as_tensor(items["language"]))
        self.mean = torch.as_tensor(meta["dense_mean"])
        self.sd = torch.as_tensor(meta["dense_sd"])

    def video(self, i):
        return self.item(i) + self.creator(self.i_creator[i]) + self.category(self.i_category[i]) + self.lang(self.i_lang[i])

    def forward(self, b):
        dense = (b["dense"] - self.mean) / self.sd
        cand = self.video(b["item"]) + self.stats(dense[:, 1:4]) + self.kind.weight[1]
        req = self.ulang(b["ulang"]) + self.req(dense[:, 6:8]) + self.kind.weight[0]
        h = b["H"][b["u"], :HIST]
        hw = b["HW"][b["u"], :HIST]
        hist = self.video(h) + self.hist_w(torch.log1p(hw)[..., None]) + self.kind.weight[2]
        return req, cand, hist, hw > 0, dense


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        layer = nn.TransformerEncoderLayer(D, nhead=2, dim_feedforward=64, dropout=0.0, batch_first=True)
        self.layers = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)

    def forward(self, x, mask):
        return self.layers(x, src_key_padding_mask=~mask)


class Scorer(nn.Module):
    """kind: 'bi-encoder', 'cross-encoder' or 'cross-encoder + dense'."""
    def __init__(self, meta, items, kind, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.kind = kind
        self.tokens = Tokens(meta, items)
        self.encoder = Encoder()
        extra = 8 if kind == "cross-encoder + dense" else 0
        if kind == "bi-encoder":
            self.item_tower = nn.Sequential(nn.Linear(D, 64), nn.ReLU(), nn.Linear(64, 2 * D))
            self.user_tower = nn.Linear(D, 2 * D)
            self.bias = nn.Parameter(torch.zeros(2))
        else:
            self.head = nn.Sequential(nn.Linear(D + extra, 64), nn.ReLU(), nn.Linear(64, 2))
        self.position = nn.Embedding(11, 1)
        nn.init.zeros_(self.position.weight)

    def user_vector(self, req, hist, hmask):
        x = torch.cat([req[:, None], hist], dim=1)
        m = torch.cat([torch.ones(len(req), 1, dtype=torch.bool), hmask], dim=1)
        return self.user_tower(self.encoder(x, m)[:, 0])                # the request token's output

    def forward(self, b, position=None):
        """Listing 8.4. Click and play logits for each row of b."""
        req, cand, hist, hmask, dense = self.tokens(b)
        if self.kind == "bi-encoder":
            u = self.user_vector(req, hist, hmask)                       # the user side, without the candidate
            v = self.item_tower(cand)                                    # the candidate side, without the user
            out = torch.stack([(u[:, :D] * v[:, :D]).sum(1), (u[:, D:] * v[:, D:]).sum(1)], dim=1) + self.bias
        else:
            x = torch.cat([req[:, None], cand[:, None], hist], dim=1)    # one sequence: request, candidate, history
            m = torch.cat([torch.ones(len(req), 2, dtype=torch.bool), hmask], dim=1)
            z = self.encoder(x, m)[:, 1]                                 # the candidate token's output
            if self.kind == "cross-encoder + dense":
                z = torch.cat([z, dense], dim=1)
            out = self.head(z)
        click = out[:, 0]
        slot = position if position is not None else torch.ones_like(b["item"])
        click = click + self.position(slot)[:, 0] - self.position.weight[1, 0]
        return {"click": click, "play": out[:, 1]}


def fit(model, b, R, epochs, seed=0, lr=2e-3, batch=512, after_epoch=None):
    """Chapter 7's losses: click log loss on every row, play log loss on clicked rows."""
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    yc = torch.as_tensor(R["click"], dtype=torch.float32)
    yp = torch.as_tensor(R["play"], dtype=torch.float32)
    pos = torch.as_tensor(R["position"])
    n = len(yc)
    model.train()
    for e in range(1, epochs + 1):
        perm = torch.randperm(n, generator=gen)
        for a in range(0, n, batch):
            idx = perm[a:a + batch]
            out = model(take(b, idx), position=pos[idx])
            loss = F.binary_cross_entropy_with_logits(out["click"], yc[idx])
            c = yc[idx] > 0
            if c.any():
                loss = loss + F.binary_cross_entropy_with_logits(out["play"][c], yp[idx][c])
            opt.zero_grad()
            loss.backward()
            opt.step()
        if after_epoch is not None:
            model.eval()
            after_epoch(e)
            model.train()
    model.eval()
    return model


def held_out(model, b, R):
    """Validation loss: click log loss (true slot) plus play log loss on clicked rows."""
    pos = torch.as_tensor(R["position"])
    n = len(R["click"])
    with torch.no_grad():
        outs = [model(take(b, torch.arange(a, min(a + 10_000, n))), position=pos[a:a + 10_000])
                for a in range(0, n, 10_000)]
    click = torch.cat([o["click"] for o in outs])
    play = torch.cat([o["play"] for o in outs])
    yc = torch.as_tensor(R["click"], dtype=torch.float32)
    yp = torch.as_tensor(R["play"], dtype=torch.float32)
    c = yc > 0
    return float(F.binary_cross_entropy_with_logits(click, yc) + F.binary_cross_entropy_with_logits(play[c], yp[c]))


def kept_scores(model, sp, users, items, who):
    """Expected satisfied plays at slot 1 for every kept candidate."""
    kept, rows = sp[f"kept_{who}"], sp[who]
    out = np.full(kept.shape, -np.inf)
    with torch.no_grad():
        for a in range(0, len(rows), 128):
            r, k = rows[a:a + 128], kept[a:a + 128]
            uu, ii = np.repeat(r, k.shape[1]), k.ravel()
            ok = ii >= 0
            bb = tensors([sp["S"]], users, items, uu[ok], ii[ok], np.zeros(ok.sum(), dtype=np.int64))
            o = model(take(bb, torch.arange(int(ok.sum()))))
            s = np.full(len(ii), -np.inf)
            s[ok] = (torch.sigmoid(o["click"]) * torch.sigmoid(o["play"])).double().numpy()
            out[a:a + 128] = s.reshape(k.shape)
    return out


def request_ms(model, sp, users, items, heavy=False, repeats=200):
    """Median milliseconds to score one user's 50 kept candidates (inputs already
    built). For the bi-encoder the candidate vectors are computed ahead, as they
    would be stored: only the user side and 50 dot products are timed."""
    times = []
    with torch.no_grad():
        for r in range(repeats):
            ii = sp["kept_aud"][r][sp["kept_aud"][r] >= 0]
            bb = tensors([sp["S"]], users, items, np.full(len(ii), sp["aud"][r]), ii, np.zeros(len(ii), dtype=np.int64))
            if not heavy and model.kind == "bi-encoder":
                _, cand, _, _, _ = model.tokens(bb)
                V = model.item_tower(cand)                                # stored ahead in production
                one = take(bb, torch.arange(1))

                def call():
                    req, _, hist, hmask, _ = model.tokens(one)
                    u = model.user_vector(req, hist, hmask)
                    return (V[:, :D] * u[:, :D]).sum(1), (V[:, D:] * u[:, D:]).sum(1)
            else:
                def call():
                    return model(bb)
            call()                                                        # warm-up
            t = time.perf_counter()
            for _ in range(10):
                call()
            times.append((time.perf_counter() - t) / 10)
    return 1000 * float(np.median(times))


if __name__ == "__main__":
    t0 = time.time()
    W = world()
    users, items = W["users"], W["items"]
    val, test = split(W, "validation"), split(W, "test")
    Rh = W["weeks"]["validation, held out"]
    bh = tensors([W["snaps"][16]], users, items, Rh["user"], Rh["item"], np.zeros(len(Rh["user"]), dtype=np.int64))
    kinds = ("bi-encoder", "cross-encoder", "cross-encoder + dense")
    print(f"Section 8.5: tokens of {D} numbers; {HIST} history videos per user; two self-attention layers  "
          f"[{time.time() - t0:.0f} s]")
    print("\nValidation week: held-out loss (days 16 to 23) after each epoch")
    chosen = {}
    for kind in kinds:
        losses = []
        m = Scorer(val["meta"], items, kind)
        fit(m, val["b"], val["R"], max(EPOCHS), after_epoch=lambda e: losses.append(held_out(m, bh, Rh)))
        chosen[kind] = EPOCHS[int(np.argmin(np.round(losses, 9)))]
        print(f"  {kind:<24}" + "".join(f"{x:>9.4f}" for x in losses) + f"   chosen: {chosen[kind]}  "
              f"[{time.time() - t0:.0f} s]")

    print(f"\nTest week: each model reorders the {KEPT} candidates the shipped pre-ranker keeps; offline on "
          f"{len(test['rows']):,} test users, the rest on {len(test['aud']):,} consenting users")
    print(f"  {'':<26}{'NDCG@10':>8}{'coverage':>9}{'top-10% cr.':>12}{'ms / request':>14}   online, seeds 0, 1, 2"
          f"      mean   against  s.e. seeds  s.e. users")
    base_runs, base = [], None
    for seed in SEEDS:                                       # the shipped ranker, retrained from each seed
        h = test["heavy"] if seed == 0 else train(Ranker(test["meta"], SHIP["arch"], seed=seed), test["b"],
                                                  test["R"], epochs=SHIP["epochs"], seed=seed)
        res = row_metrics(order_by(test["kept_rows"], score_pools(h, test["S"], users, items, test["rows"],
                                                                  test["kept_rows"])),
                          order_by(test["kept_aud"], score_pools(h, test["S"], users, items, test["aud"],
                                                                 test["kept_aud"])), test, W, boredom_too=False)
        base_runs.append(res["online"])
        base = res if seed == 0 else base
    print(f"  {'heavy ranker (shipped)':<26}{base['NDCG@10']:>8.4f}{base['coverage']:>9.1%}"
          f"{base['top-10% creators']:>12.1%}{request_ms(test['heavy'], test, users, items, heavy=True):>14.3f}"
          f"   {base_runs[0].mean():.4f}  {base_runs[1].mean():.4f}  {base_runs[2].mean():.4f}  "
          f"{np.mean([o.mean() for o in base_runs]):.4f}  [{time.time() - t0:.0f} s]")
    for kind in kinds:
        onl, first, ms = [], None, None
        for seed in SEEDS:
            m = fit(Scorer(test["meta"], items, kind, seed=seed), test["b"], test["R"], chosen[kind], seed=seed)
            res = row_metrics(order_by(test["kept_rows"], kept_scores(m, test, users, items, "rows")),
                              order_by(test["kept_aud"], kept_scores(m, test, users, items, "aud")), test, W,
                              boredom_too=False)
            onl.append(res["online"])
            if seed == 0:
                first, ms = res, request_ms(m, test, users, items)
        d, se_seeds, se_users = seed_paired(onl, base_runs)
        print(f"  {kind:<26}{first['NDCG@10']:>8.4f}{first['coverage']:>9.1%}{first['top-10% creators']:>12.1%}"
              f"{ms:>14.3f}   {onl[0].mean():.4f}  {onl[1].mean():.4f}  {onl[2].mean():.4f}  "
              f"{np.mean([o.mean() for o in onl]):.4f}   {d:+.4f}  {se_seeds:>10.4f}  {se_users:>10.4f}  "
              f"[{time.time() - t0:.0f} s]")
    print("  (NDCG@10, coverage, top-10% creators and ms / request are seed 0's; ms / request: median time to score "
          "one user's 50 candidates on this machine, one reference run; the bi-encoder's candidate vectors are "
          "computed ahead. The last three columns compare each seed with the heavy ranker retrained from the same "
          "seed: the mean of the three differences, its standard error across seeds, and the per-user standard "
          "error of one run)")
