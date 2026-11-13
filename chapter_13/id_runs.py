"""Random IDs, content and semantic IDs inside Chapter 12's HSTU (sections 13.2, 13.4 and 13.5).

Every arm is HSTU of width 64 with 2 layers (id_models.ItemLM), trained on the scale
world's training clicks with the new videos' early clicks removed (sid_data.data), until
its validation loss has not improved for 2 epochs (at most 30), with the best epoch kept.
Only the way it reads a video changes:

  arms    - random ID; content; semantic IDs from RQ-KMeans, from RQ-VAE and from RQ-VAE
            with co-clicks (build_ids.py, sids.npz); the last plus a random ID, plus
            content, and trained with its fine rows sometimes dropped
  length  - collaborative-aware semantic IDs of 3 codes of 16, 64 or 256

The measures are on the held-out users' test clicks of days 23 to 30, split by the
clicked video: common, rare (a warm video in the bottom half by training clicks) and
new (never in training). For each: the cross-entropy (nats; lower is better) and
recall@20 (the share of clicks whose video is among the 20 the model scores highest of
all 3,480). The differences on new videos come with 95% intervals from 1,000 bootstrap
draws of whole users.

Needs build_ids.py's sids.npz (and, for length, its cached sweep). Each arm's results are
kept in chapter_13/cache, so an interrupted run resumes; the times shown are each arm's
own. Run from the repository root (about 1.5 hours on 2 CPUs; add arms or length to run
one part):
    python chapter_13/id_runs.py
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
from sid_data import SLICE_NAMES, cached, co_clicks, content, data, new_videos, slices, world  # noqa: E402
from quantise import disambiguate, train_rqvae  # noqa: E402
from id_models import ItemLM, evaluate, table_rows  # noqa: E402
from large_models import fit  # noqa: E402
from seq_data import environment  # noqa: E402

FILE = HERE / "sids.npz"
TOP = 20


def run(name, label, D, new, sl, **kw):
    """Trains one arm (kept in the cache, sid_data.cached) and prints its line."""
    def train():
        m = ItemLM(D["n_videos"], unseen=new, **kw)
        m, curve = fit(m, D["train"], D["valid"], 30, patience=2)
        return {"eval": evaluate(m, D["test"]), "curve": curve, "rows": table_rows(m)}
    out, sec = cached(name, train)
    r = out["eval"]
    s = sl[r["video"]]
    cells = "".join(f"{r['ce'][s == j].mean():>10.4f}{(r['rank'][s == j] < TOP).mean():>8.4f}" for j in range(3))
    print(f"  {label:<32}{out['rows']:>7,}{len(out['curve']):>4}{int(np.argmin(out['curve'])) + 1:>5}{cells}"
          f"{r['ce'].mean():>9.4f}  [{sec:.0f} s]", flush=True)
    return r


def boot(a, b, user, mask, n=1000, seed=0):
    """The mean of a - b over the clicks in mask, and its 95% interval from n draws of
    whole users with replacement."""
    u = user[mask]
    d = np.bincount(u, weights=(a - b)[mask], minlength=user.max() + 1)
    c = np.bincount(u, minlength=user.max() + 1).astype(float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(d), size=(n, len(d)))
    est = d[draws].sum(1) / c[draws].sum(1)
    return d.sum() / c.sum(), *np.percentile(est, [2.5, 97.5])


HEAD = (f"  {'':<32}{'rows':>7}{'ep':>4}{'best':>5}" + "".join(f"{n + ' CE':>10}{'R@20':>8}" for n in SLICE_NAMES)
        + f"{'all CE':>9}")


if __name__ == "__main__":
    t0 = time.time()
    parts = sys.argv[1:] or ["arms", "length"]
    print(environment())
    W = world()
    X = content(W)
    D = data(W)
    new = np.zeros(D["n_videos"], dtype=bool)
    new[new_videos(W)] = True
    sl = slices(D["clicks"], new)
    Z = np.load(FILE)
    y = np.concatenate([yy[ok] for yy, ok in zip(D["test"][4], D["test"][5])])
    print(f"Scale world: {D['n_videos']:,} videos, {new.sum():,} of them new; test clicks: "
          + ", ".join(f"{n} {np.isin(y, np.flatnonzero(sl == j)).sum():,}" for j, n in enumerate(SLICE_NAMES)))
    if "arms" in parts:
        print("\nHow HSTU reads a video (width 64, 2 layers):")
        print(HEAD)
        R = {}
        R["random ID"] = run("arm_random", "random ID", D, new, sl)
        R["content"] = run("arm_content", "content", D, new, sl, use_id=False, content=X)
        R["SID RQ-KMeans"] = run("arm_rq_kmeans", "semantic ID, RQ-KMeans", D, new, sl, use_id=False,
                                 codes=Z["rq_kmeans"])
        R["SID RQ-VAE"] = run("arm_rq_vae", "semantic ID, RQ-VAE", D, new, sl, use_id=False, codes=Z["rq_vae"])
        R["SID co-clicks"] = run("arm_co_clicks", "semantic ID, RQ-VAE + co-clicks", D, new, sl, use_id=False,
                                 codes=Z["rq_vae_co_clicks"])
        R["SID co-clicks + ID"] = run("arm_co_clicks_id", "  the same + random ID", D, new, sl,
                                      codes=Z["rq_vae_co_clicks"])
        R["SID co-clicks + content"] = run("arm_co_clicks_content", "  the same + content", D, new, sl,
                                           use_id=False, codes=Z["rq_vae_co_clicks"], content=X)
        R["SID co-clicks, rows dropped"] = run("arm_co_clicks_drop", "  the same, fine rows dropped", D, new, sl,
                                               use_id=False, codes=Z["rq_vae_co_clicks"], drop=0.5)
        print(f"  (rows: item vectors learned; ep: epochs run; best: the epoch kept; CE: test cross-entropy; R@20: "
              f"recall@{TOP}; [s]: training time)")
        user = D["test_user"][R["random ID"]["window"]]
        is_new = sl[R["random ID"]["video"]] == 2
        print("\nNew videos, difference in cross-entropy (95% interval, 1,000 draws of whole users):")
        for base in ("random ID", "content"):
            for arm in ("SID RQ-KMeans", "SID RQ-VAE", "SID co-clicks", "SID co-clicks + ID", "SID co-clicks + content",
                        "SID co-clicks, rows dropped", "content"):
                if arm != base:
                    m, lo, hi = boot(R[arm]["ce"], R[base]["ce"], user, is_new)
                    print(f"  {arm} minus {base}: {m:+.4f} ({lo:+.4f} to {hi:+.4f})")
    if "length" in parts:
        pairs = co_clicks(D)
        print("\nCollaborative-aware semantic IDs of 3 codes, with 16, 64 or 256 codes per level:")
        print(HEAD)
        for k in (16, 64, 256):
            if k == 64:
                codes = Z["rq_vae_co_clicks"]
            else:
                codes = disambiguate(cached(f"sweep_{k}_3", lambda: train_rqvae(X, k, 3, pairs=pairs)[1:])[0][0])
            name = "arm_co_clicks" if k == 64 else f"length_{k}"                    # the same as arms'
            run(name, f"{k} codes per level", D, new, sl, use_id=False, codes=codes)
    print(f"[{time.time() - t0:.0f} s]")
