"""Scaling laws for generative sequence models: size, data and history length (sections 12.4 and 12.5).

In the scale world (scale_world.py), HSTU and SASRec (large_models.py) learn to predict
each user's next video click. They train on 90% of the users' clicks before day 23; the
other 10% of users are held out. The validation loss is the mean cross-entropy (nats)
of the held-out users' clicks before day 23, and it chooses the epoch; the test loss is
the same for their clicks in days 23 to 30. Each click is predicted at the moment of
the request that led to it, from the clicks made before that moment in its window.
Lower is better. Two references: predicting every click with the training clicks'
popularity, and the cross-entropy of a uniform guess over all videos.

  size     - 7 sizes (width d and layers) from d 16 with 1 layer to d 256 with 4 layers,
             6 epochs each, with every click and windows of 64 (Figure 12.5). Every
             epoch of every run is a point (compute, validation loss); the lowest loss
             at or below each compute forms the frontier, fitted with a power law in
             compute (scaling_laws.py), and the best test loss of each size is fitted
             with a power law in non-embedding weights (as Kaplan and colleagues
             counted them: the video and position vectors are left out).
  data     - one size (d 64, 2 layers) on 1/32, 1/16, 1/8, 1/4, 1/2 and all of the
             training users (nested; seed 1), each trained until its validation loss
             has not improved for 2 epochs (at most 40), so that every run reaches its
             best; power law in clicks.
  length   - the same size with windows of 2, 4, 8, 16, 32 and 64 clicks, 6 epochs;
             power law in window length.

Compute is train_flops (Listing 12.3) times the training clicks seen.

Needs Python 3.12 or later, PyTorch and SciPy. Run from the repository root (about 2.75
hours on 2 CPUs; add size, data or length to run one part):
    python chapter_12/seq_scaling.py
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
sys.path.insert(0, str(HERE.parent / "chapter_09"))
from scale_world import DAY, load, sequences  # noqa: E402
from large_models import SeqLM, fit, loss, scored, train_flops, windows  # noqa: E402
from scaling_laws import doubling, fit_power_law, frontier  # noqa: E402
from seq_data import environment  # noqa: E402

KINDS = ("HSTU", "SASRec")
SIZES = ((16, 1), (32, 1), (32, 2), (64, 2), (128, 2), (128, 4), (256, 4))
FRACTIONS = (1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0)
LENGTHS = (2, 4, 8, 16, 32, 64)
MID = (64, 2)


def clicks(data):
    return scored(data)


def references(D, test):
    """Test loss of the training clicks' popularity (add-one smoothed), and of a uniform guess."""
    counts = np.bincount(np.concatenate(D["train"][0]), minlength=D["n_videos"]) + 1.0
    logp = np.log(counts / counts.sum())
    y = np.concatenate([y[ok] for y, ok in zip(test[4], test[5])])
    return -logp[y].mean(), np.log(D["n_videos"])


def law_line(label, x, y, unit):
    l_inf, a, alpha = fit_power_law(x, y)
    print(f"  {label}: L = {l_inf:.4f} + {a:.4g} x {unit}^-{alpha:.4f}; each doubling multiplies the removable "
          f"loss by {doubling(alpha):.4f}")
    return l_inf, a, alpha


def run(D, kind, d, layers, k, train, valid, test, epochs, t0, label="", patience=None):
    m = SeqLM(D["n_videos"], kind, d, layers, max_len=k)
    m, curve = fit(m, train, valid, epochs, patience=patience)
    test_loss = loss(m, test)
    per = train_flops(m, k) * clicks(train)
    params = sum(p.numel() for p in m.parameters())
    best = int(np.argmin(curve)) + 1
    print(f"  {kind:<7}{label:<14}{m.non_embedding_params():>12,}{params:>11,}{train_flops(m, k):>13,}"
          f"{per:>11.3g}  " + " ".join(f"{v:.4f}" for v in curve) + f"{best:>6}{test_loss:>9.4f}"
          f"  [{time.time() - t0:.0f} s]", flush=True)
    return {"curve": curve, "test": test_loss, "per_epoch": per, "params": params,
            "non_embedding": m.non_embedding_params()}


if __name__ == "__main__":
    t0 = time.time()
    parts = sys.argv[1:] or ["size", "data", "length"]
    print(environment())
    W = load()
    D = sequences(W)
    k = 64
    train = windows(*D["train"], k)
    valid = windows(*D["held"], k, hi=23 * DAY)
    test = windows(*D["held"], k, lo=23 * DAY)
    print(f"Scale world: {D['n_videos']:,} videos; {clicks(train):,} training clicks of "
          f"{len(D['train'][0]):,} users; held-out users {D['held_users']:,}: {clicks(valid):,} validation clicks "
          f"(before day 23), {clicks(test):,} test clicks (days 23 to 30)  [{time.time() - t0:.0f} s]")
    pop, uni = references(D, test)
    print(f"  test loss of popularity {pop:.4f}; of a uniform guess over {D['n_videos']:,} videos {uni:.4f}")
    head = (f"  {'':<21}{'non-emb.':>12}{'params':>11}{'FLOPs/click':>13}{'FLOPs/ep.':>11}  "
            f"validation loss after each epoch{'':>16}{'best':>6}{'test':>9}")

    if "size" in parts:
        print(f"\nModel size: every training click, windows of {k}, 6 epochs")
        print(head)
        res = {}
        for kind in KINDS:
            for d, L in SIZES:
                res[kind, d, L] = run(D, kind, d, L, k, train, valid, test, 6, t0, f"d {d}, {L} layer" + "s" * (L > 1))
        print("\nScaling laws over model size (validation loss for the frontier, test loss for non-embedding weights):")
        for kind in KINDS:
            pts = [(r["per_epoch"] * (e + 1), v) for (kd, _, _), r in res.items() if kd == kind
                   for e, v in enumerate(r["curve"])]
            fr = frontier(pts)
            print(f"  {kind} frontier ({len(fr)} of {len(pts)} points): "
                  + ", ".join(f"{c:.3g}:{v:.4f}" for c, v in fr))
            law_line(f"{kind}, loss against compute", [c for c, _ in fr], [v for _, v in fr], "C")
            rs = [r for (kd, _, _), r in res.items() if kd == kind]
            law_line(f"{kind}, test loss against non-embedding weights", [r["non_embedding"] for r in rs],
                     [r["test"] for r in rs], "N")

    if "data" in parts:
        order = np.random.default_rng(1).permutation(len(D["train"][0]))
        print(f"\nTraining data: d {MID[0]}, {MID[1]} layers, windows of {k}, until 2 epochs without improvement "
              f"(at most 40), the first part of one random order of the training users")
        print(head)
        res = {}
        for kind in KINDS:
            for f in FRACTIONS:
                keep = order[:int(round(f * len(order)))]
                part = windows(*([a[i] for i in keep] for a in D["train"]), k)
                res[kind, f] = run(D, kind, *MID, k, part, valid, test, 40, t0, f"{f:.4g} of users", patience=2)
                res[kind, f]["clicks"] = clicks(part)
        print("\nScaling laws over training clicks (test loss):")
        for kind in KINDS:
            rs = [res[kind, f] for f in FRACTIONS]
            print(f"  {kind}: " + ", ".join(f"{r['clicks']:,} clicks {r['test']:.4f}" for r in rs))
            law_line(f"{kind}, test loss against training clicks", [r["clicks"] for r in rs],
                     [r["test"] for r in rs], "D")

    if "length" in parts:
        print(f"\nHistory length: d {MID[0]}, {MID[1]} layers, every training click, 6 epochs; each click is "
              f"predicted from the earlier clicks in its window")
        print(head)
        res = {}
        for kind in KINDS:
            for n in LENGTHS:
                tr, va, te = (windows(*D["train"], n), windows(*D["held"], n, hi=23 * DAY),
                              windows(*D["held"], n, lo=23 * DAY))
                res[kind, n] = run(D, kind, *MID, n, tr, va, te, 6, t0, f"window {n}")
        print("\nScaling laws over history length (test loss):")
        for kind in KINDS:
            rs = [res[kind, n] for n in LENGTHS]
            print(f"  {kind}: " + ", ".join(f"{n}: {r['test']:.4f}" for n, r in zip(LENGTHS, rs)))
            law_line(f"{kind}, test loss against window length", list(LENGTHS), [r["test"] for r in rs], "k")
    print(f"[{time.time() - t0:.0f} s]")
