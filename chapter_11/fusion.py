"""Pictures, video and sound: each modality alone, and two ways to fuse them (section 11.4).

The next-video test of section 11.3 (requests of days 23 to 30, each answered from the
clicks the server had received before it), with every video scored by the similarity
of its content to the request's history. The content is the title's Qwen3-Embedding
vector and the simulated outputs of three pretrained encoders (content_data.modalities):
the thumbnail (image), the video frames (video) and the soundtrack (audio); 10% of
videos have no usable sound. For each modality, the request's profile is the mean of
the vectors of the history's videos that have it, scaled to length 1, and a video's
similarity is the cosine between its vector and that profile.

  one modality        - that modality's similarity only (a video without it scores 0)
  late fusion, mean   - the mean of the similarities of the modalities the video has
  late fusion, sum    - the sum of the four similarities, a missing one counting as 0
  early fusion        - Listing 11.3: the four vectors side by side, each with weight 1,
                        renormalised over the modalities the video has; the profile is
                        the mean of the history's fused vectors (Listing 11.2)
  early fusion, missing sound filled - the same, but a video without sound is given
                        the mean of the sound vectors of the videos that have it, not
                        scaled to length 1, so its sound similarity to any profile is
                        the average sound similarity of the videos with sound

"No sound" is the share of the videos shown that have no usable sound (9.6% of the
videos in Chapter 2's world). Nothing is fitted, so there is no validation choice.

Needs Python 3.12 or later, PyTorch and text_embeddings.npz. Run from the repository
root (about 4 minutes):
    python chapter_11/fusion.py
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
from content_data import MODELS, load, modalities, text_embeddings  # noqa: E402
from content_models import fuse, profiles, scorer  # noqa: E402
from graph_data import chapter8_rows, environment, ndcg, online, paired, phase, top10  # noqa: E402
from slate_data import split  # noqa: E402
import baselines as B  # noqa: E402

DAY = 86_400


def unit_rows(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def modality_profiles(X, has, H):
    """Each request's profile in one modality: the mean vector of the history's videos
    that have the modality, scaled to length 1 (zero if none has it)."""
    keep = np.where((H >= 0) & has[np.maximum(H, 0)], H, -1)
    return unit_rows(profiles(X, keep))


def line(label, W, R, top, base, no_sound):
    on = online(W, R, top)
    later = R["place"] >= 1
    shown = top >= 0
    quiet = no_sound[np.where(shown, top, 0)][shown].mean()
    against = "" if base is None else "{:+.4f} ({:.4f})".format(*paired(on, base, R["user"])[:2])
    print(f"  {label:<36}{ndcg(R, top):>9.4f}{on.mean():>9.4f}{on[later].mean():>9.4f}{against:>18}{quiet:>10.1%}")
    return on


if __name__ == "__main__":
    t0 = time.time()
    print(environment())
    W = load()
    items, users = W["items"], W["users"]
    E, idx = text_embeddings(W, MODELS[-1])
    T = idx["titles"]
    text = E[T[np.arange(len(T)), items["language"]]]
    parts = {"text": (text, np.ones(len(text), dtype=bool))} | modalities(W)
    video = ~items["is_product"]
    no_sound = video & ~parts["audio"][1]
    print(f"Chapter 2's world: {video.sum():,} videos; " + ", ".join(
        f"{k} {v[0].shape[1]} numbers ({v[1][video].mean():.1%} of videos)" for k, v in parts.items())
        + f"  [{time.time() - t0:.0f} s]")

    sp = split(W, "test")
    R, H, _, bad = phase(W, 23, sp["S"])
    pop = B.popularity(W, 23 * DAY)
    print(f"\nNext-video test: {len(R['user']):,} requests of days 23 to 30; each video scored by the similarity of "
          f"its content to the request's history  [{time.time() - t0:.0f} s]")
    print(f"  {'':<36}{'NDCG@10':>9}{'online':>9}{'later':>9}{'against Chapter 8':>18}{'no sound':>10}")
    ch8, _, _ = chapter8_rows(W, sp, R, bad)
    base = line("Chapter 8's ranker", W, R, ch8, None, no_sound)
    P = {k: modality_profiles(x, h, H) for k, (x, h) in parts.items()}
    sims = {k: scorer(P[k], parts[k][0]) for k in parts}
    has = np.stack([h for _, h in parts.values()], axis=1).astype(np.float32)
    res = {}
    for k in parts:
        res[k] = line(k + " only", W, R, top10(sims[k], bad, tie=pop), base, no_sound)

    def late(a, b, mean):
        s = sum(sims[k](a, b) for k in parts)
        return s / has.sum(axis=1)[None, :] if mean else s
    res["late mean"] = line("late fusion, mean", W, R, top10(lambda a, b: late(a, b, True), bad, tie=pop), base,
                            no_sound)
    res["late sum"] = line("late fusion, sum", W, R, top10(lambda a, b: late(a, b, False), bad, tie=pop), base,
                           no_sound)
    F = fuse(list(parts.values()), [1.0] * len(parts))
    res["early"] = line("early fusion (Listing 11.3)", W, R, top10(scorer(profiles(F, H), F), bad, tie=pop), base,
                        no_sound)
    sound, heard = parts["audio"]
    average = sound[heard].mean(axis=0, keepdims=True)
    filled = parts | {"audio": (np.where(no_sound[:, None], average, sound).astype(np.float32), heard | no_sound)}
    F = fuse(list(filled.values()), [1.0] * len(filled))
    res["filled"] = line("early fusion, missing sound filled", W, R, top10(scorer(profiles(F, H), F), bad, tie=pop),
                         base, no_sound)
    for a_, b_ in (("late mean", "text"), ("early", "text"), ("late mean", "late sum"), ("filled", "early")):
        print(f"  {a_} against {b_}: " + "{:+.4f} ({:.4f}) [{:+.4f}, {:+.4f}]".format(
            *paired(res[a_], res[b_], R["user"])))
    print("  (online: expected satisfied plays per row; later: requests after the first of their visit; against: "
          "paired difference over the same requests, standard error from 1,000 bootstrap draws of whole users; "
          f"no sound: share of the videos shown that have no usable sound)  [{time.time() - t0:.0f} s]")
