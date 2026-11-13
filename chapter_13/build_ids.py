"""Building semantic IDs for Tesselo's videos, and what makes a good one (sections 13.3 to 13.5).

  example   - the worked example of section 13.3: two levels of residual quantisation
              of one small vector, step by step
  build     - semantic IDs of 3 codes of 64, plus the code that keeps IDs unique, made
              three ways from each video's content (sid_data.content):
                RQ-KMeans on the content vectors
                RQ-VAE on the content vectors (TIGER)
                RQ-VAE with the co-click loss (collaborative-aware, as in PLUM)
              For each: how far the reconstruction is from the content, how many codes
              of each level are used and how evenly (perplexity), collisions, and how
              much of the simulator's hidden taste the first code explains; and how
              many new videos share each prefix with a trained video. Saves the IDs
              in sids.npz for id_runs.py and Chapter 14.
  collapse  - the same RQ-VAE with and without its two guards against codebook
              collapse: codebooks started from k-means, and unused codes moved
  sweep     - collaborative-aware IDs with 16, 64 or 256 codes per level and 2, 3 or 4
              levels: collisions, IDs, table rows and the taste the first code explains

Each RQ-VAE's codes are kept in chapter_13/cache, so an interrupted run resumes; the
times shown are each training's own. Run from the repository root (about 30 minutes on 2
CPUs; add a part's name to run one):
    python chapter_13/build_ids.py
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
from sid_data import cached, co_clicks, content, data, new_videos, world  # noqa: E402
from quantise import disambiguate, report, residual_quantise, rq_kmeans, train_rqvae  # noqa: E402
from id_models import prefixes  # noqa: E402
from seq_data import environment  # noqa: E402

FILE = HERE / "sids.npz"
K, L = 64, 3


def example():
    x = np.array([0.9, 0.3])
    books = [np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]),
             np.array([[0.1, 0.1], [-0.1, 0.2], [0.0, -0.2]])]
    print("Worked example: x = (0.9, 0.3)")
    r = x.copy()
    for level, C in enumerate(books, 1):
        d = ((r - C) ** 2).sum(1)
        c = int(d.argmin())
        print(f"  level {level}: residual ({r[0]:.2f}, {r[1]:.2f}); squared distances to the codes "
              + ", ".join(f"{v:.2f}" for v in d) + f"; code {c}, vector ({C[c][0]:.2f}, {C[c][1]:.2f})")
        r = r - C[c]
    codes, rec = residual_quantise(x[None], books)
    print(f"  ID {tuple(int(c) for c in codes[0])}; reconstruction ({rec[0][0]:.2f}, {rec[0][1]:.2f}); "
          f"what is left ({r[0]:.2f}, {r[1]:.2f})")


def line(label, codes, taste, cat, err=None, chance=False, sec=None):
    r = report(codes, K, taste, cat)
    full = disambiguate(codes)
    if chance:                                   # shuffled codes: only the checks against taste mean anything
        print(f"  {label:<28}{'':>58}{r['taste_explained'][0]:>9.3f}{r['first_code_category']:>9.3f}")
        return full
    print(f"  {label:<28}{'' if err is None else f'{err:.4f}':>8}  {'/'.join(map(str, r['used'])):>9}  "
          f"{'/'.join(f'{p:.1f}' for p in r['perplexity']):>15}{r['unique_ids']:>8,}{r['colliding_items']:>8,}"
          f"{r['largest_group']:>6}{r['taste_explained'][0]:>9.3f}{r['first_code_category']:>9.3f}"
          f"{sum(prefixes(full)[1]):>8,}" + ("" if sec is None else f"  [{sec:.0f} s]"), flush=True)
    return full


def rqvae(name, X, k, levels, **kw):
    """train_rqvae's codes and error, kept in the cache (sid_data.cached)."""
    (codes, err), sec = cached(name, lambda: train_rqvae(X, k, levels, **kw)[1:])
    return codes, err, sec


if __name__ == "__main__":
    t0 = time.time()
    parts = sys.argv[1:] or ["example", "build", "collapse", "sweep"]
    print(environment())
    if "example" in parts:
        example()
    W = world()
    X = content(W)
    v = W["video"]
    taste, cat = W["items"]["vec"][v], W["items"]["category"][v]
    D = data(W)
    pairs = co_clicks(D)
    head = (f"  {'':<28}{'error':>8}  {'used':>9}  {'perplexity':>15}{'IDs':>8}{'shared':>8}{'max':>6}"
            f"{'taste':>9}{'category':>9}{'rows':>8}")
    print(f"\n{len(v):,} videos; content vectors of {X.shape[1]} numbers; {len(pairs):,} co-click pairs from the "
          f"training clicks; {len(new_videos(W)):,} new videos")
    if "build" in parts:
        print(f"\nSemantic IDs of {L} codes of {K}:")
        print(head)
        rng = np.random.default_rng(0)
        codes, rec, _ = rq_kmeans(X, K, L)
        line("chance (codes shuffled)", rng.permutation(codes), taste, cat, chance=True)
        km = line("RQ-KMeans, content", codes, taste, cat, float(((rec - X) ** 2).sum() / (X ** 2).sum()))
        codes, err, sec = rqvae("rqvae_content", X, K, L)
        vae = line("RQ-VAE, content", codes, taste, cat, err, sec=sec)
        codes, err, sec = rqvae("rqvae_co_clicks", X, K, L, pairs=pairs)
        col = line("RQ-VAE, content + co-clicks", codes, taste, cat, err, sec=sec)
        np.savez_compressed(FILE, rq_kmeans=km, rq_vae=vae, rq_vae_co_clicks=col)
        print(f"  (error: squared reconstruction error over the content's own; used: codes used at each level; "
              f"perplexity: exp of the entropy of code use, {K} if all are used equally; IDs: distinct sequences of "
              f"{L} codes; shared: videos whose {L} codes another video also has; max: the most videos with one "
              f"sequence; taste: share of the variance of the hidden taste explained by the first code's groups; "
              f"category: share of videos in the most common category of their first code; rows: embedding rows "
              f"of Listing 13.5 with the extra code; [s]: training time)")
        new = np.zeros(len(v), dtype=bool)
        new[new_videos(W)] = True
        print("New videos whose prefix of 1, 2, 3 or 4 codes a trained video also has (rows of Listing 13.5 that "
              "a new video shares):")
        for name, c in (("RQ-KMeans", km), ("RQ-VAE", vae), ("RQ-VAE + co-clicks", col)):
            g, _ = prefixes(c)
            share = [np.isin(g[new, j], g[~new, j]).mean() for j in range(g.shape[1])]
            print(f"  {name:<20}" + "".join(f"{x:>7.3f}" for x in share))
    if "collapse" in parts:
        print(f"\nCodebook collapse: RQ-VAE on content, {L} codes of {K}, 3,000 steps:")
        print(head)
        for init, reset in ((False, False), (True, False), (False, True), (True, True)):
            name = "rqvae_content" if init and reset else f"collapse_{int(init)}{int(reset)}"   # the same as build's
            codes, err, sec = rqvae(name, X, K, L, kmeans_init=init, reset_dead=reset)
            line(f"k-means start {'yes' if init else 'no'}, reset {'yes' if reset else 'no'}", codes, taste, cat, err,
                 sec=sec)
    if "sweep" in parts:
        print("\nCodebook size and ID length: RQ-VAE on content + co-clicks:")
        print(f"  {'codes':>6}{'levels':>8}{'IDs':>8}{'shared':>8}{'max':>6}{'taste':>9}{'chance':>9}{'rows':>8}")
        rng = np.random.default_rng(0)
        for k in (16, 64, 256):
            for lv in (2, 3, 4):
                name = "rqvae_co_clicks" if (k, lv) == (K, L) else f"sweep_{k}_{lv}"            # the same as build's
                codes, err, sec = rqvae(name, X, k, lv, pairs=pairs)
                r = report(codes, k, taste, cat)
                rows = sum(prefixes(disambiguate(codes))[1])
                chance = report(rng.permutation(codes), k, taste)["taste_explained"][0]
                print(f"  {k:>6}{lv:>8}{r['unique_ids']:>8,}{r['colliding_items']:>8,}{r['largest_group']:>6}"
                      f"{r['taste_explained'][0]:>9.3f}{chance:>9.3f}{rows:>8,}  [{sec:.0f} s]", flush=True)
    print(f"[{time.time() - t0:.0f} s]")
