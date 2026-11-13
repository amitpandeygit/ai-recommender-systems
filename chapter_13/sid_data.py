"""Shared data for Chapter 13: Chapter 12's scale world, each video's content, and the
videos held back as new (sections 13.2 to 13.5).

The scaling world of Chapter 12 (chapter_12/scale_world.py) has 3,480 videos and 1.1
million training clicks. This module adds what each video looks like, built as in
Chapter 11: its title as TF-IDF over its words (chapter_11/content_models.py), and the
simulated outputs of pretrained picture, video and sound encoders (chapter_11/
content_data.py, Listing 11.1), side by side, each block unit length (Listing 11.3).

It also holds back 10% of the videos, drawn at random (seed 13), as new: their clicks
before day 23 are removed for every user, as if they were uploaded on day 23. They
appear only in the test clicks of days 23 to 30, where a model must score videos it
never trained on.

  world()                 - the scale world (Chapter 12), with its items rebuilt so that
                            their titles and hidden encoders' inputs are known
  content(W)              - one content vector per video (videos numbered 0 to V - 1),
                            its title weights learned from the videos before day 23
  NEW_SHARE, new_videos   - the share of videos held back, and which they are
  data(W, k)              - training, validation and test windows (Chapter 12's
                            request-time windows), with the new videos' early clicks
                            removed, the user of each test window, and the training
                            clicks of each video
  co_clicks(D)            - pairs of videos clicked one after the other in training
  cached(name, fn)        - runs fn once and keeps its result in cache/, so that an
                            interrupted script resumes
  slices(train_clicks, new) - each video's slice: new, rare (a warm video in the
                            bottom half by training clicks) or common

Run from the repository root to print the data's sizes (about 1 minute):
    python chapter_13/sid_data.py
"""
import os
import pickle
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "2"
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _ch in ("chapter_12", "chapter_11", "chapter_09", "chapter_03", "chapter_02"):
    sys.path.insert(0, str(ROOT / _ch))
from scale_world import DAY, USERS, VISITS, load, sequences  # noqa: E402
from large_models import scored, windows  # noqa: E402
from tesselo_data import Config, make_world  # noqa: E402
from tesselo_shop import add_attributes  # noqa: E402
from content_data import modalities  # noqa: E402
from content_models import fuse  # noqa: E402

NEW_SHARE = 0.10
K = 64                                                   # Chapter 12's window
CACHE = HERE / "cache"


def cached(name, fn):
    """Runs fn() once and keeps its result, with the seconds it took, in cache/<name>.pkl,
    so that an interrupted script resumes where it stopped. Delete the folder to start
    afresh. Returns (result, seconds)."""
    f = CACHE / f"{name}.pkl"
    if f.exists():
        with open(f, "rb") as h:
            return pickle.load(h)
    t = time.time()
    out = (fn(), time.time() - t)
    CACHE.mkdir(exist_ok=True)
    with open(f.with_suffix(".tmp"), "wb") as h:
        pickle.dump(out, h)
    f.with_suffix(".tmp").rename(f)
    return out


def world():
    """Chapter 12's scale world, plus its items as Chapter 2's generator made them (same
    configuration and seed), so that titles and the hidden inputs of the simulated
    encoders are known. Checks that they are the same items as the saved world's."""
    W = load()
    cfg = Config(num_users=USERS, sessions_per_user=VISITS)
    users, items, _ = make_world(cfg, np.random.default_rng(cfg.seed))
    add_attributes(cfg, users, items)
    assert (items["category"] == W["category"]).all() and (items["creator"] == W["creator"]).all()
    W["cfg"], W["items"] = cfg, items
    W["video"] = np.flatnonzero(~items["is_product"])
    return W


def title_tfidf(docs, fit):
    """Unit-length TF-IDF rows, with the vocabulary and the word weights learned from the
    documents marked by fit only; words seen only in other documents are dropped."""
    vocab = {}
    for d, f in zip(docs, fit):
        if f:
            for t in d:
                vocab.setdefault(t, len(vocab))
    X = np.zeros((len(docs), len(vocab)))
    for r, d in enumerate(docs):
        for t in d:
            if t in vocab:
                X[r, vocab[t]] += 1
    df = (X[fit] > 0).sum(0)
    X = X * np.log(fit.sum() / df)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def content(W):
    """Each video's content vector: title TF-IDF, picture, video and sound, side by side,
    each block unit length and weighted equally; a video without sound has a zero sound
    block (chapter_11/content_models.fuse). The TF-IDF's words and weights are learned
    from the videos that exist before day 23 only, not from the new videos' titles."""
    v = W["video"]
    old = np.ones(len(v), dtype=bool)
    old[new_videos(W)] = False
    title = title_tfidf([list(map(str, W["items"]["title"][i])) for i in v], old).astype(np.float32)
    parts = [(title, np.ones(len(v), dtype=bool))]
    for name, (x, has) in modalities(W).items():
        parts.append((x[v], has[v]))
    return fuse(parts, [1.0] * len(parts))


def new_videos(W):
    """The held-back videos, numbered 0 to V - 1 as in Chapter 12's sequences."""
    n = len(W["video"])
    return np.sort(np.random.default_rng(13).choice(n, int(round(NEW_SHARE * n)), replace=False))


def data(W, k=K):
    """Chapter 12's training, validation and test windows, with every click on a new
    video before day 23 removed. Returns the windows, the number of training clicks of
    each video, and the user of each test window."""
    renum = np.full(len(W["is_product"]), -1)
    renum[W["video"]] = np.arange(len(W["video"]))
    new = np.zeros(len(W["video"]), dtype=bool)
    new[new_videos(W)] = True
    keep = ~(new[renum[W["item"]]] & (W["time"] < 23 * DAY))
    V = dict(W)
    for name in ("user", "item", "time", "request_time", "session"):
        V[name] = W[name][keep]
    D = sequences(V)
    train = windows(*D["train"], k)
    valid = windows(*D["held"], k, hi=23 * DAY)
    test = windows(*D["held"], k, lo=23 * DAY)
    clicks = np.bincount(np.concatenate(D["train"][0]), minlength=len(W["video"]))
    per_user = [len(windows([s], [t], [m], k, lo=23 * DAY)[0]) for s, t, m in zip(*D["held"])]
    test_user = np.repeat(np.arange(len(per_user)), per_user)              # the user of each test window
    assert len(test_user) == len(test[0])
    return {"train": train, "valid": valid, "test": test, "test_user": test_user, "clicks": clicks,
            "n_videos": D["n_videos"], "removed": int((~keep).sum())}


def co_clicks(D):
    """Pairs of videos one user clicked one after the other in the training windows."""
    pairs = np.vstack([np.stack([x[:-1], x[1:]], 1) for x in D["train"][0] if len(x) > 1])
    return pairs[pairs[:, 0] != pairs[:, 1]]


def slices(clicks, new):
    """0 = common, 1 = rare (a warm video in the bottom half by training clicks), 2 = new."""
    s = np.zeros(len(clicks), dtype=np.int64)
    warm = np.flatnonzero(~new)
    rare = warm[np.argsort(clicks[warm], kind="stable")[: len(warm) // 2]]
    s[rare] = 1
    s[new] = 2
    return s


SLICE_NAMES = ("common", "rare", "new")


if __name__ == "__main__":
    t0 = time.time()
    W = world()
    X = content(W)
    new = np.zeros(len(W["video"]), dtype=bool)
    new[new_videos(W)] = True
    D = data(W)
    s = slices(D["clicks"], new)
    print(f"Videos: {len(W['video']):,}; content vector of {X.shape[1]} numbers (title TF-IDF {X.shape[1] - 160}, "
          f"picture 64, video 64, sound 32); held back as new: {new.sum():,}  [{time.time() - t0:.0f} s]")
    taste = W["items"]["vec"][W["video"]]
    A = np.hstack([X, np.ones((len(X), 1), dtype=np.float32)])
    B = np.linalg.solve(A.T @ A + 0.1 * np.eye(A.shape[1]), A.T @ taste)          # ridge regression
    r2 = 1 - ((A @ B - taste) ** 2).sum() / ((taste - taste.mean(0)) ** 2).sum()
    print(f"  a linear map from the content vector explains {r2:.3f} of the variance of the hidden taste "
          f"(16 numbers, known only to the simulator)")
    print(f"  clicks on new videos before day 23 removed: {D['removed']:,}; training clicks predicted: "
          f"{scored(D['train']):,}; validation: {scored(D['valid']):,}; test: {scored(D['test']):,}")
    y = np.concatenate([yy[ok] for yy, ok in zip(D["test"][4], D["test"][5])])
    for j, name in enumerate(SLICE_NAMES):
        v = s == j
        c = D["clicks"][v]
        print(f"  {name:<7} videos {v.sum():>5,}; training clicks per video: median {np.median(c):,.0f}, "
              f"range {c.min():,} to {c.max():,}; test clicks {np.isin(y, np.flatnonzero(v)).sum():,}")
