"""A foundation model that teaches a small model (section 12.6).

Meta's GEM is trained on every ads surface and passes what it learned to the smaller
models that serve each surface. This is the same idea at Tesselo's scale, in the scale
world (scale_world.py):

  teacher  - HSTU, d 128 with 4 layers, the best size of section 12.5, trained on every
             training user's clicks before day 23 (6 epochs, the best on validation
             kept): the "foundation" model
  student  - HSTU, d 32 with 1 layer, small enough for Tesselo's latency budget, which
             sees only one surface's data: the clicks of 1/8 of the training users
             (the first eighth of one random order, seed 1)

The student is trained three ways, each until its validation loss has not improved for
2 epochs (at most 40), with the best epoch kept:
  alone            - on its own clicks, with the usual cross-entropy
  distilled        - on its own clicks, with the teacher's softened prediction over every
                     video as a second target (large_models.distill_loss: weight 0.5,
                     temperature 2)
  with all clicks  - alone, but on every training user's clicks: what more data would
                     give the small model without a teacher

The measure is the test loss of section 12.5 (held-out users' clicks in days 23 to 30),
and the share of the gap between the student alone and the teacher that each closes.

Needs Python 3.12 or later and PyTorch. Run from the repository root (about 25 minutes):
    python chapter_12/distill.py
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
from large_models import SeqLM, fit, loss, train_flops, windows  # noqa: E402
from seq_data import environment  # noqa: E402

K = 64
TEACHER, STUDENT = (128, 4), (32, 1)
SHARE = 1 / 8


def line(label, m, curve, test, t0):
    print(f"  {label:<34}{sum(p.numel() for p in m.parameters()):>11,}{train_flops(m, K) / 3:>14,.0f}  "
          + " ".join(f"{v:.4f}" for v in curve) + f"{test:>9.4f}  [{time.time() - t0:.0f} s]", flush=True)


if __name__ == "__main__":
    t0 = time.time()
    print(environment())
    D = sequences(load())
    train = windows(*D["train"], K)
    valid = windows(*D["held"], K, hi=23 * DAY)
    test = windows(*D["held"], K, lo=23 * DAY)
    order = np.random.default_rng(1).permutation(len(D["train"][0]))
    keep = order[:int(round(SHARE * len(order)))]
    own = windows([D["train"][0][i] for i in keep], [D["train"][1][i] for i in keep], K)
    n_own = int(sum(s.sum() for s in own[2]))
    n_all = int(sum(s.sum() for s in train[2]))
    print(f"Teacher data: {n_all:,} training clicks; the student's surface: {len(keep):,} users, {n_own:,} clicks  "
          f"[{time.time() - t0:.0f} s]")
    print(f"  {'':<34}{'params':>11}{'FLOPs/click':>14}  validation loss after each epoch{'':>30}{'test':>9}")
    teacher, curve = fit(SeqLM(D["n_videos"], "HSTU", *TEACHER, max_len=K), train, valid, 6)
    t_test = loss(teacher, test)
    line(f"teacher, d {TEACHER[0]}, {TEACHER[1]} layers", teacher, curve, t_test, t0)
    res = {}
    for label, data, teach in (("alone", own, None), ("distilled", own, teacher), ("with all clicks", train, None)):
        m, curve = fit(SeqLM(D["n_videos"], "HSTU", *STUDENT, max_len=K), data, valid, 40, teacher=teach,
                       patience=2)
        res[label] = loss(m, test)
        line(f"student, {label}", m, curve, res[label], t0)
    gap = res["alone"] - t_test
    print(f"\nGap between the student alone and the teacher: {gap:.4f} nats. Closed by distillation: "
          f"{res['alone'] - res['distilled']:.4f} ({(res['alone'] - res['distilled']) / gap:.1%}); by 8 times the data "
          f"without a teacher: {res['alone'] - res['with all clicks']:.4f} "
          f"({(res['alone'] - res['with all clicks']) / gap:.1%})")
    print("(FLOPs/click: one forward pass per predicted click, what serving costs; test: cross-entropy of the held-out "
          f"users' clicks in days 23 to 30)  [{time.time() - t0:.0f} s]")
