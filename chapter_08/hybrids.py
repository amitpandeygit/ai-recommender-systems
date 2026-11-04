"""Listing 8.3 - Hybrids: blending several models' scores (section 8.4).

Tesselo has three scores for each (user, video) pair:
  A    - the shipped heavy ranker (the MLP with id vectors, one epoch): P(click) x P(play | click)
  C    - the MLP without id vectors (six epochs), Chapter 7's arm C, the same product
  iALS - Chapter 4's dot product, which is also one of the rankers' inputs
A blend turns them into one score. Three blends are compared:
  equal average - the mean of A's and C's expected satisfied plays
  stacking      - a logistic regression whose inputs are log A, log C and the iALS score
                  (plus one term per slot, for position), fitted to the satisfied plays
                  of held-out impressions
  feature-weighted linear stacking (Sill and colleagues, 2009) - the same, but each
                  model's weight is a linear function of meta-features of the request:
                  whether the user has history, how much, whether the video is in the
                  app language, and how popular the video is
The blends are fitted on the validation week: base models trained on days 9 to 16
predict the impressions of days 16 to 23, and the blend learns from those
predictions, never from rows its base models trained on. For the test week the base
models are retrained on days 9 to 23 and the fitted weights are kept. Every method
orders the same 50 candidates the shipped pre-ranker keeps.

Needs Python 3.12 or later and PyTorch. Run from the repository root (about two minutes):
    python chapter_08/hybrids.py
"""
import os
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "2"              # NumPy's linear algebra: the same thread count on every run
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from slate_data import (KEPT, Ranker, order_by, paired, predict, row_metrics, split, tensors, train,  # noqa: E402
                        world)
from metrics import log_loss  # noqa: E402

SECOND = {"arch": "mlp, no ids", "epochs": 6}           # Chapter 7's arm C
META = ["constant", "has history", "history size (log)", "same language", "popularity (log)"]


def base_predictions(A, C, b, n, position=None):
    """log A, log C and the iALS score for n rows (slot 1 unless positions are given)."""
    pa = np.multiply(*predict(A, b, n, position=position)).astype(np.float64)
    pc = np.multiply(*predict(C, b, n, position=position)).astype(np.float64)
    dense = b["dense"].numpy()
    return np.log(pa), np.log(pc), dense[:, 0].astype(np.float64), dense


def meta_features(dense):
    """Meta-features of each row, from the dense features (see ranking_data.DENSE)."""
    return np.stack([np.ones(len(dense)), dense[:, 7], dense[:, 6] / 3.0, dense[:, 4], dense[:, 1] / 5.0],
                    axis=1).astype(np.float64)


def design(la, lc, ials, dense, kind, slot=None):
    """Listing 8.3. The blend's inputs. 'stacking': the three scores; 'fwls': every
    score times every meta-feature. Slot terms (one per slot 2 to 10) absorb
    position in training; serving uses slot 1, where they are zero."""
    base = np.stack([la, lc, ials], axis=1)
    if kind == "fwls":
        m = meta_features(dense)
        base = (base[:, :, None] * m[:, None, :]).reshape(len(la), -1)
        base = np.concatenate([base, m[:, 1:]], axis=1)              # the meta-features alone, too
    X = np.concatenate([np.ones((len(la), 1)), base], axis=1)
    if slot is not None:
        X = np.concatenate([X, (slot[:, None] == np.arange(2, 11)[None, :]).astype(float)], axis=1)
    else:
        X = np.concatenate([X, np.zeros((len(la), 9))], axis=1)
    return X


def fit_logistic(X, y, ridge=1e-3, iters=25):
    """Logistic regression by Newton's method with a small ridge penalty."""
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ w)))
        g = X.T @ (p - y) + ridge * w
        H = (X * (p * (1 - p))[:, None]).T @ X + ridge * np.eye(X.shape[1])
        w -= np.linalg.solve(H, g)
    return w


def fit_blends(W, val, C_val):
    """Fit stacking and feature-weighted stacking on the validation week's held-out
    impressions (days 16 to 23), with base models trained on days 9 to 16."""
    users, items = W["users"], W["items"]
    Rh = W["weeks"]["validation, held out"]
    bh = tensors([W["snaps"][16]], users, items, Rh["user"], Rh["item"], np.zeros(len(Rh["user"]), dtype=np.int64))
    la, lc, ials, dense = base_predictions(val["heavy"], C_val, bh, len(Rh["user"]),
                                           position=torch.as_tensor(Rh["position"]))
    y = Rh["play"].astype(float)
    w = {k: fit_logistic(design(la, lc, ials, dense, k, Rh["position"]), y) for k in ("stacking", "fwls")}
    return w, (la, lc, ials, dense, y)


def blend_scores(w, kind, A, C, sp, users, items, kept, rows):
    """A blend's score at slot 1 for every kept candidate."""
    uu, ii = np.repeat(rows, kept.shape[1]), kept.ravel()
    ok = ii >= 0
    b = tensors([sp["S"]], users, items, uu[ok], ii[ok], np.zeros(ok.sum(), dtype=np.int64))
    la, lc, ia, d = base_predictions(A, C, b, int(ok.sum()))
    return to_matrix(ok, design(la, lc, ia, d, kind) @ w[kind], kept.shape)


def kept_inputs(A, C, sp, users, items, who):
    """Base predictions for every kept candidate of every user, at slot 1."""
    kept = sp[f"kept_{who}"]
    rows = sp[who]
    uu, ii = np.repeat(rows, kept.shape[1]), kept.ravel()
    ok = ii >= 0
    b = tensors([sp["S"]], users, items, uu[ok], ii[ok], np.zeros(ok.sum(), dtype=np.int64))
    return ok, base_predictions(A, C, b, int(ok.sum()))


def to_matrix(ok, values, shape):
    out = np.full(ok.shape, -np.inf)
    out[ok] = values
    return out.reshape(shape)


if __name__ == "__main__":
    t0 = time.time()
    W = world()
    users, items = W["users"], W["items"]
    val, test = split(W, "validation"), split(W, "test")
    C_val = train(Ranker(val["meta"], SECOND["arch"]), val["b"], val["R"], epochs=SECOND["epochs"])
    C_test = train(Ranker(test["meta"], SECOND["arch"]), test["b"], test["R"], epochs=SECOND["epochs"])
    print(f"Base models trained  [{time.time() - t0:.0f} s]")

    # ---- fit the blends on the validation week's held-out impressions (days 16 to 23)
    w, (la, lc, ials, dense, y) = fit_blends(W, val, C_val)
    print(f"\nSection 8.4: blends fitted on {len(y):,} held-out impressions of days 16 to 23 "
          f"({int(y.sum()):,} satisfied plays), base models trained on days 9 to 16")
    print("  stacking weights: " + ", ".join(f"{n} {v:+.3f}" for n, v in
                                             zip(["intercept", "log A", "log C", "iALS score"], w["stacking"][:4])))
    print("  feature-weighted stacking, weight of each score by meta-feature:")
    print(f"    {'':<10}" + "".join(f"{m:>20}" for m in META))
    for k, s in enumerate(("log A", "log C", "iALS")):
        print(f"    {s:<10}" + "".join(f"{w['fwls'][1 + 5 * k + j]:>+20.3f}" for j in range(5)))
    pop = float(dense[:, 1].mean())
    for hs in (1.0, 3.0):                       # how the weights move with the size of the user's history
        m = np.array([1.0, 1.0, hs / 3.0, 1.0, pop / 5.0])
        print(f"    history size (log) {hs:.1f}, video in the app language, average popularity: weight on log A "
              f"{w['fwls'][1:6] @ m:+.3f}, on log C {w['fwls'][6:11] @ m:+.3f}, on iALS {w['fwls'][11:16] @ m:+.3f}")
    print(f"[{time.time() - t0:.0f} s]")

    # ---- test week: held-out log loss, offline and online
    R3 = W["weeks"]["test, held out"]
    b3 = tensors([W["snaps"][23]], users, items, R3["user"], R3["item"], np.zeros(len(R3["user"]), dtype=np.int64))
    s3 = torch.as_tensor(R3["position"])
    la3, lc3, ia3, d3 = base_predictions(test["heavy"], C_test, b3, len(R3["user"]), position=s3)
    y3 = R3["play"].astype(float)
    p_blend = {k: 1 / (1 + np.exp(-(design(la3, lc3, ia3, d3, k, R3["position"]) @ w[k]))) for k in w}
    print(f"\nTest week: held-out impressions of days 23 to 30 ({len(y3):,} rows), satisfied play log loss: "
          f"A {log_loss(np.exp(la3), y3):.4f}, C {log_loss(np.exp(lc3), y3):.4f}, "
          f"equal average {log_loss((np.exp(la3) + np.exp(lc3)) / 2, y3):.4f}, "
          f"stacking {log_loss(p_blend['stacking'], y3):.4f}, feature-weighted {log_loss(p_blend['fwls'], y3):.4f}")
    cache = {who: kept_inputs(test["heavy"], C_test, test, users, items, who) for who in ("rows", "aud")}

    def scores(name, who):
        ok, (la_, lc_, ia_, d_) = cache[who]
        if name == "A":
            v = la_
        elif name == "C":
            v = lc_
        elif name == "iALS score":
            v = ia_
        elif name == "equal average":
            v = (np.exp(la_) + np.exp(lc_)) / 2
        else:
            v = design(la_, lc_, ia_, d_, name) @ w[name]
        return to_matrix(ok, v, test[f"kept_{who}"].shape)

    print(f"  each method orders the {KEPT} kept candidates; offline on {len(test['rows']):,} test users, "
          f"the rest on {len(test['aud']):,} consenting users")
    print(f"  {'':<34}{'hit@10':>7}{'NDCG@10':>8}{'online':>8}{'coverage':>9}{'top-10% cr.':>12}   against A")
    base = None
    for name, label in (("A", "A: shipped heavy ranker"), ("C", "C: MLP without id vectors"),
                        ("iALS score", "iALS score alone"), ("equal average", "equal average of A and C"),
                        ("stacking", "stacking"), ("fwls", "feature-weighted linear stacking")):
        tr = order_by(test["kept_rows"], scores(name, "rows"))
        ta = order_by(test["kept_aud"], scores(name, "aud"))
        res = row_metrics(tr, ta, test, W, boredom_too=False)
        if base is None:
            base = res["online"]
        d, se = paired(res["online"], base)
        print(f"  {label:<34}{res['hit@10']:>7.3f}{res['NDCG@10']:>8.3f}{res['online'].mean():>8.4f}"
              f"{res['coverage']:>9.1%}{res['top-10% creators']:>12.1%}   {d:+.4f} ({se:.4f})")
    print(f"[{time.time() - t0:.0f} s]")
