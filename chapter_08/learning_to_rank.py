"""Listing 8.2 - Learning to rank: one network, different losses (section 8.3).

Chapter 7's heavy ranker is trained pointwise: each impression row on its own,
with a click head and a play head. Learning to rank trains a score whose only job
is to put the right videos first. Here the network stays the shipped one (the MLP
with id vectors and the shallow position tower, rankers.py) and only the loss
changes. The relevance label is the satisfied play (a completion or a like).

  pointwise, two heads - Chapter 7's loss: log loss of the click, and of the play given a click
  pointwise, one head  - log loss of the satisfied play itself
  pairwise, RankNet    - inside each logged slate, every (satisfied, not satisfied) pair:
                         -log sigmoid(s_i - s_j)
  pairwise, BPR        - the same criterion, but each satisfied play is paired with an
                         eligible video drawn at random from those the user has not used or
                         enjoyed, as BPR does (Rendle and colleagues, 2009; Chapter 4 used it
                         for factorisation)
  listwise, softmax    - per slate, cross-entropy between the label shares y_i / sum(y) and the
                         softmax of the scores: the softmax cross-entropy loss (Bruch and
                         colleagues, 2019); original ListNet puts a softmax on the labels too
  LambdaRank           - RankNet's pairs, each weighted by how much NDCG would change if the
                         two swapped places in the current order
  LambdaMART           - LightGBM's lambdarank objective: gradient-boosted trees on the
                         eight dense features, with the slot as a feature (set to 1 to serve)

The number of epochs (1 to 4) is chosen for each loss on the validation week by
that loss on held-out slates (days 16 to 23), as Chapter 7 chose epochs. Then the
loss for the shipped ranker is chosen by validation NDCG@10 (Chapter 5's rule).
Each model is retrained on days 9 to 23 from seeds 0, 1 and 2 and orders the same
50 candidates the shipped pre-ranker keeps.

Needs Python 3.12 or later, PyTorch and LightGBM. Run from the repository root (about three minutes):
    python chapter_08/learning_to_rank.py
"""
import os
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "2"              # NumPy's linear algebra: the same thread count on every run
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from slate_data import (KEPT, SHIP, Ranker, order_by, predict, row_metrics, seed_paired, slates, split,  # noqa: E402
                        tensors, train, world)
from rankers import take  # noqa: E402
from ranking_data import pair_features  # noqa: E402
from metrics import ndcg_at_k  # noqa: E402

LOSSES = ("pointwise, one head", "pairwise, RankNet", "pairwise, BPR", "listwise, softmax", "LambdaRank")
EPOCHS = (1, 2, 3, 4)
SEEDS = (0, 1, 2)


def ltr_loss(kind, s, y, mask):
    """Listing 8.2. s: scores (slates x 10), y: labels, mask: real rows. RankNet,
    softmax and LambdaRank use only slates with at least one satisfied play."""
    has_pos = (y * mask).sum(1) > 0
    s, y, mask = s[has_pos], y[has_pos], mask[has_pos]
    if len(s) == 0:
        return None
    if kind == "listwise, softmax":
        logp = F.log_softmax(s.masked_fill(~mask, -1e9), dim=1)
        return -((y / y.sum(1, keepdim=True)) * logp).sum(1).mean()
    better = (y[:, :, None] > y[:, None, :]) & mask[:, :, None] & mask[:, None, :]   # pairs (i above j)
    if not better.any():
        return None
    diff = s[:, :, None] - s[:, None, :]
    loss = F.softplus(-diff)                                              # -log sigmoid(s_i - s_j)
    if kind == "LambdaRank":
        with torch.no_grad():                                             # |change in NDCG| if i and j swapped
            order = torch.argsort(s.masked_fill(~mask, -1e9), dim=1, descending=True)
            rank = torch.empty_like(order)
            rank.scatter_(1, order, torch.arange(s.shape[1]).expand_as(order))
            disc = 1.0 / torch.log2(rank.float() + 2.0)
            ideal = torch.sort(y, dim=1, descending=True).values / torch.log2(torch.arange(s.shape[1]) + 2.0)
            gain = (2 ** y - 1) / ideal.sum(1, keepdim=True)
            weight = (gain[:, :, None] - gain[:, None, :]).abs() * (disc[:, :, None] - disc[:, None, :]).abs()
        loss = loss * weight
        return loss[better].sum() / len(s)
    return loss[better].mean()


def train_ltr(meta, b, R, G, kind, S_list, users, items, epochs, seed=0, lr=2e-3, batch=128, after_epoch=None):
    """Train the shipped network's click logit as one ranking score. G holds the
    logged slates (row indices); BPR pairs each satisfied row with a random video."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = Ranker(meta, SHIP["arch"], seed=seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    y_all = torch.as_tensor(R["play"], dtype=torch.float32)
    pos_all = torch.as_tensor(R["position"])
    if kind == "pointwise, one head":
        units = np.arange(len(R["play"]))                                 # rows
    elif kind == "pairwise, BPR":
        units = np.flatnonzero(R["play"])                                 # satisfied rows
        known = known_positives(R)
    else:
        units = np.flatnonzero((G >= 0).any(1) & np.array([R["play"][g[g >= 0]].any() for g in G]))
    step = 1024 if kind == "pointwise, one head" else batch
    model.train()
    for e in range(1, epochs + 1):
        perm = units[torch.randperm(len(units), generator=gen).numpy()]
        for start in range(0, len(perm), step):
            u = perm[start:start + step]
            if kind == "pointwise, one head":
                out = model(take(b, torch.as_tensor(u)), position=pos_all[u])["click"]
                loss = F.binary_cross_entropy_with_logits(out, y_all[u])
            elif kind == "pairwise, BPR":
                neg = np.array([sample_negative(S_list[R["snap"][r]], R["user"][r], known, rng) for r in u])
                u, neg = u[neg >= 0], neg[neg >= 0]                       # a row with no negative is skipped
                if len(u) == 0:
                    continue
                s_pos = model(take(b, torch.as_tensor(u)), position=pos_all[u])["click"]
                bn = tensors(S_list, users, items, R["user"][u], neg, R["snap"][u])
                s_neg = model(take(bn, torch.arange(len(u))), position=pos_all[u])["click"]
                loss = F.softplus(-(s_pos - s_neg)).mean()
            else:
                g = torch.as_tensor(G[u])
                mask = g >= 0
                flat = g[mask]
                s = torch.zeros(g.shape)
                s[mask] = model(take(b, flat), position=pos_all[flat])["click"]
                y = torch.zeros(g.shape)
                y[mask] = y_all[flat]
                loss = ltr_loss(kind, s, y, mask)
                if loss is None:
                    continue
            opt.zero_grad()
            loss.backward()
            opt.step()
        if after_epoch is not None:
            model.eval()
            after_epoch(e, model)
            model.train()
    model.eval()
    return model


def known_positives(R):
    """Each user's satisfied plays in the training rows (all weeks of R): BPR treats
    them as observed, so they are never drawn as negatives."""
    known = {}
    for u, i in zip(R["user"][R["play"]], R["item"][R["play"]]):
        known.setdefault(int(u), set()).add(int(i))
    return known


def sample_negative(S, u, known, rng, tries=1_000):
    """An eligible video the user has neither used before the snapshot nor enjoyed in
    the training rows, drawn uniformly (BPR's sampler); -1 if there is none."""
    seen = known.get(int(u), set())
    for _ in range(tries):
        j = int(rng.integers(len(S["ok"])))
        if S["ok"][j] and not S["used"][u, j] and j not in seen:
            return j
    left = np.flatnonzero(S["ok"] & ~S["used"][u])
    left = left[~np.isin(left, list(seen))]
    return int(rng.choice(left)) if len(left) else -1


def ndcg10(top, sp):
    """Mean NDCG@10 of the rows over the week's test users."""
    return float(np.mean([ndcg_at_k(t.tolist(), {i: 1.0 for i in sp["pos"][u]}) for u, t in zip(sp["rows"], top)]))


def slate_ndcg(scores, y, G):
    """Mean NDCG of the logged slates with at least one satisfied play, when the slate
    is reordered by the model's scores (slot 1 for every row)."""
    out = []
    for g in G:
        g = g[g >= 0]
        if y[g].sum() == 0:
            continue
        order = g[np.argsort(-scores[g], kind="stable")]
        dcg = np.sum(y[order] / np.log2(np.arange(2, len(g) + 2)))
        ideal = np.sum(np.sort(y[g])[::-1] / np.log2(np.arange(2, len(g) + 2)))
        out.append(dcg / ideal)
    return float(np.mean(out))


def held_out_loss(kind, model, b, R, G):
    """The loss itself on held-out slates (for choosing the number of epochs)."""
    pos = torch.as_tensor(R["position"])
    with torch.no_grad():
        s = torch.as_tensor(np.concatenate([model(take(b, torch.arange(a, min(a + 20_000, len(R["play"])))),
                                                  position=pos[a:a + 20_000])["click"].numpy()
                                            for a in range(0, len(R["play"]), 20_000)]))
        y = torch.as_tensor(R["play"], dtype=torch.float32)
        if kind == "pointwise, one head":
            return float(F.binary_cross_entropy_with_logits(s, y))
        if kind == "pairwise, BPR":                                      # in-slate pairs stand in for sampled ones
            kind = "pairwise, RankNet"
        g = torch.as_tensor(G)
        mask = g >= 0
        S_, Y_ = torch.zeros(g.shape), torch.zeros(g.shape)
        S_[mask], Y_[mask] = s[g[mask]], y[g[mask]]
        return float(ltr_loss(kind, S_, Y_, mask))


def lgb_data(S_list, users, items, R, G):
    """LightGBM wants each slate's rows together: the eight dense features plus the slot."""
    order = np.concatenate([g[g >= 0] for g in G])
    X = np.zeros((len(order), 9), dtype=np.float32)
    for s, S in enumerate(S_list):
        m = R["snap"][order] == s
        X[m, :8] = pair_features(S, users, items, R["user"][order][m], R["item"][order][m])
    X[:, 8] = R["position"][order]
    return X, R["play"][order].astype(int), (G >= 0).sum(1)


def lgb_params(seed):
    return {"objective": "lambdarank", "metric": "ndcg", "eval_at": [10], "learning_rate": 0.05,
            "num_leaves": 31, "min_data_in_leaf": 50, "feature_fraction": 0.8, "bagging_fraction": 0.8,
            "bagging_freq": 1, "seed": seed, "deterministic": True, "num_threads": 2, "verbosity": -1}


def lgb_scores(model, S, users, items, rows, kept):
    out = np.full(kept.shape, -np.inf)
    for a in range(0, len(rows), 512):
        r, k = rows[a:a + 512], kept[a:a + 512]
        uu, ii = np.repeat(r, k.shape[1]), k.ravel()
        ok = ii >= 0
        X = np.ones((ok.sum(), 9), dtype=np.float32)                     # column 8: slot 1
        X[:, :8] = pair_features(S, users, items, uu[ok], ii[ok])
        s = np.full(len(ii), -np.inf)
        s[ok] = model.predict(X)
        out[a:a + 512] = s.reshape(k.shape)
    return out


def net_scores(model, S, users, items, rows, kept, two_heads=False):
    """Each kept candidate's score at slot 1: the click logit, or for Chapter 7's
    two-head model the expected satisfied plays P(click) x P(play | click)."""
    out = np.full(kept.shape, -np.inf)
    for a in range(0, len(rows), 256):
        r, k = rows[a:a + 256], kept[a:a + 256]
        uu, ii = np.repeat(r, k.shape[1]), k.ravel()
        ok = ii >= 0
        bb = tensors([S], users, items, uu[ok], ii[ok], np.zeros(ok.sum(), dtype=np.int64))
        s = np.full(len(ii), -np.inf)
        if two_heads:
            pc, pp = predict(model, bb, int(ok.sum()))
            s[ok] = pc.astype(np.float64) * pp
        else:
            with torch.no_grad():
                s[ok] = model(take(bb, torch.arange(int(ok.sum()))))["click"].double().numpy()
        out[a:a + 256] = s.reshape(k.shape)
    return out


def row_scores_ho(model, b, n, two_heads=False):
    """Scores of held-out rows at slot 1 (for in-slate NDCG)."""
    if two_heads:
        pc, pp = predict(model, b, n)
        return pc.astype(np.float64) * pp
    with torch.no_grad():
        return np.concatenate([model(take(b, torch.arange(a, min(a + 20_000, n))))["click"].double().numpy()
                               for a in range(0, n, 20_000)])


if __name__ == "__main__":
    t0 = time.time()
    W = world()
    users, items = W["users"], W["items"]
    val, test = split(W, "validation"), split(W, "test")
    print(f"Shipped pipeline rebuilt  [{time.time() - t0:.0f} s]")
    Gv = slates(val["R"])
    Rh = W["weeks"]["validation, held out"]
    bh = tensors([W["snaps"][16]], users, items, Rh["user"], Rh["item"], np.zeros(len(Rh["user"]), dtype=np.int64))
    Rh = dict(Rh, snap=np.zeros(len(Rh["user"]), dtype=np.int64))
    Gh = slates(Rh)
    n_pos = np.array([val["R"]["play"][g[g >= 0]].sum() for g in Gv])
    print(f"Section 8.3: logged slates, days 9 to 16: {len(Gv):,}, {np.mean((Gv >= 0).sum(1)):.2f} rows each; "
          f"{np.mean(n_pos > 0):.1%} hold a satisfied play, {np.mean(n_pos > 1):.1%} more than one")

    # ---- validation week: epochs per loss, then the loss by NDCG@10
    print("\nValidation week: the loss on held-out slates of days 16 to 23 after each epoch, and validation NDCG@10")
    print(f"  {'':<24}" + "".join(f"{'ep. ' + str(e):>10}" for e in EPOCHS) + f"{'chosen':>8}{'NDCG@10':>9}")
    chosen, val_nd = {}, {}
    nd = ndcg10(order_by(val["kept_rows"], val["heavy_rows"]), val)
    chosen["pointwise, two heads"] = SHIP["epochs"]
    val_nd["pointwise, two heads"] = nd
    print(f"  {'pointwise, two heads':<24}{'(Chapter 7: 1 epoch, chosen on the click and play losses)':<48}{nd:>9.4f}")
    for kind in LOSSES:
        losses = []
        train_ltr(val["meta"], val["b"], val["R"], Gv, kind, val["snaps"], users, items, max(EPOCHS),
                  after_epoch=lambda e, m: losses.append(held_out_loss(kind, m, bh, Rh, Gh)))
        ep = EPOCHS[int(np.argmin(np.round(losses, 9)))]
        m = train_ltr(val["meta"], val["b"], val["R"], Gv, kind, val["snaps"], users, items, ep)
        top = order_by(val["kept_rows"], net_scores(m, val["S"], users, items, val["rows"], val["kept_rows"]))
        nd = ndcg10(top, val)
        chosen[kind], val_nd[kind] = ep, nd
        print(f"  {kind:<24}" + "".join(f"{x:>10.4f}" for x in losses) + f"{ep:>8}{nd:>9.4f}")
    Xv, yv, gv = lgb_data(val["snaps"], users, items, val["R"], Gv)
    Xh, yh, gh = lgb_data([W["snaps"][16]], users, items, Rh, Gh)
    dv = lgb.Dataset(Xv, yv, group=gv)
    booster = lgb.train(lgb_params(0), dv, num_boost_round=1_000, valid_sets=[lgb.Dataset(Xh, yh, group=gh)],
                        callbacks=[lgb.early_stopping(50, verbose=False)])
    rounds = booster.best_iteration
    top = order_by(val["kept_rows"], lgb_scores(booster, val["S"], users, items, val["rows"], val["kept_rows"]))
    nd = ndcg10(top, val)
    chosen["LambdaMART"], val_nd["LambdaMART"] = rounds, nd
    print(f"  {'LambdaMART':<24}{'(trees chosen by early stopping on held-out slates: ' + str(rounds) + ')':<48}"
          f"{nd:>9.4f}")
    best = max(val_nd, key=lambda k: round(val_nd[k], 9))
    print(f"Chosen by validation NDCG@10: {best}   [{time.time() - t0:.0f} s]")

    # ---- test week: three seeds each, ordering the same 50 kept candidates
    G = slates(test["R"])
    R3 = W["weeks"]["test, held out"]
    b3 = tensors([W["snaps"][23]], users, items, R3["user"], R3["item"], np.zeros(len(R3["user"]), dtype=np.int64))
    R3 = dict(R3, snap=np.zeros(len(R3["user"]), dtype=np.int64))
    G3 = slates(R3)
    S = test["S"]
    print(f"\nTest week: each model orders the {KEPT} candidates the shipped pre-ranker keeps; offline on "
          f"{len(test['rows']):,} test users, online on {len(test['aud']):,} consenting users")
    print(f"  {'':<24}{'slate NDCG':>11}{'NDCG@10':>9}{'coverage':>9}{'top-10% cr.':>12}   online, seeds 0, 1, 2"
          f"      mean   against  s.e. seeds  s.e. users")
    base_runs = None
    for kind in ("pointwise, two heads",) + LOSSES + ("LambdaMART",):
        onl, first = [], None
        for seed in SEEDS:
            t_ = time.time()
            if kind == "pointwise, two heads":
                m = test["heavy"] if seed == 0 else train(Ranker(test["meta"], SHIP["arch"], seed=seed), test["b"],
                                                          test["R"], epochs=SHIP["epochs"], seed=seed)
                sc = lambda r, k: net_scores(m, S, users, items, r, k, two_heads=True)        # noqa: E731
                ho = row_scores_ho(m, b3, len(R3["play"]), two_heads=True)
            elif kind == "LambdaMART":
                X, y, g = lgb_data(test["snaps"], users, items, test["R"], G)
                m = lgb.train(lgb_params(seed), lgb.Dataset(X, y, group=g), num_boost_round=chosen[kind])
                sc = lambda r, k: lgb_scores(m, S, users, items, r, k)                       # noqa: E731
                X3 = np.ones((len(R3["play"]), 9), dtype=np.float32)
                X3[:, :8] = pair_features(S, users, items, R3["user"], R3["item"])
                ho = m.predict(X3)
            else:
                m = train_ltr(test["meta"], test["b"], test["R"], G, kind, test["snaps"], users, items,
                              chosen[kind], seed=seed)
                sc = lambda r, k: net_scores(m, S, users, items, r, k)                       # noqa: E731
                ho = row_scores_ho(m, b3, len(R3["play"]))
            top_r = order_by(test["kept_rows"], sc(test["rows"], test["kept_rows"]))
            top_a = order_by(test["kept_aud"], sc(test["aud"], test["kept_aud"]))
            res = row_metrics(top_r, top_a, test, W, boredom_too=False)
            onl.append(res["online"])
            if seed == 0:
                first = (slate_ndcg(ho, R3["play"].astype(float), G3), res)
        if kind == "pointwise, two heads":
            base_runs = onl
        sn, res = first
        print(f"  {kind:<24}{sn:>11.4f}{res['NDCG@10']:>9.4f}{res['coverage']:>9.1%}{res['top-10% creators']:>12.1%}"
              f"   {onl[0].mean():.4f}  {onl[1].mean():.4f}  {onl[2].mean():.4f}  {np.mean([o.mean() for o in onl]):.4f}"
              + ("" if kind == "pointwise, two heads" else
                 "   {:+.4f}  {:>15.4f}  {:>10.4f}".format(*seed_paired(onl, base_runs))))
    print("  (slate NDCG: the logged slates of days 23 to 30 with a satisfied play, reordered by the model; the "
          "last three columns compare each seed with the same seed of pointwise, two heads: the mean of the three "
          "differences, its standard error across seeds, and the per-user standard error of one run)")
    print(f"[{time.time() - t0:.0f} s]")
