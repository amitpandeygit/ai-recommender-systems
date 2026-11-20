"""Generating the next video code by code, and constrained beam search (sections 14.2 to 14.4).

Every model reads the history with Chapter 13's HSTU (width 64, 2 layers), each video
through its collaborative-aware semantic ID (build_ids.py's rq_vae_co_clicks, 3 codes of
64 plus the code that keeps IDs unique), and is trained on the scale world's training
clicks with the new videos' early clicks removed (chapter_13/sid_data.data), until its
validation loss has not improved for 2 epochs (at most 30), keeping the best epoch.

  example   - the worked example of section 14.3: beam search over a trie by hand
  train     - the reference, Chapter 13's model scoring every video with a full softmax;
              the generator (Listing 14.1) on the same IDs; the generator on RQ-KMeans
              IDs (OneRec's tokenizer); and Chapter 13's content model (for section 14.4)
  test      - on the held-out users' test clicks of days 23 to 30, split into common,
              rare and new videos: cross-entropy and recall of the reference and of the
              generator's exact ranking; the probability the generator gives to IDs that
              name no video; how often beam search without the trie writes such IDs; the
              beam width (chosen on the validation clicks) against recall, agreement with
              the exact ranking, and cost per request; the RQ-KMeans generator; and new
              videos, alone and with LIGER's fix (generated candidates plus every new
              video, ranked by the content model)

Each trained model is kept in chapter_14/cache, so an interrupted run resumes; the times
shown are each training's own. Run from the repository root (about 1 hour 25 minutes on 2
CPUs; add a part's name to run one part):
    python chapter_14/beam_runs.py
"""
import os
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "2"
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
for _ch in ("chapter_13", "chapter_12", "chapter_09"):
    sys.path.insert(0, str(ROOT / _ch))
from sid_data import SLICE_NAMES, content, data, new_videos, slices, world  # noqa: E402
from id_models import ItemLM, evaluate, hit_rate  # noqa: E402
from large_models import fit, request_batch, tensors  # noqa: E402
from seq_data import environment  # noqa: E402
from gen_models import Generator, Trie, all_logprob, beam_search, fit_generator  # noqa: E402

CACHE = HERE / "cache"
SIDS = ROOT / "chapter_13" / "sids.npz"
WIDTHS = (10, 16, 32, 64, 128)
TOP = (10, 20)


def cached(name, fn):
    """Runs fn() once and keeps its result, with the seconds it took, in cache/<name>.pkl."""
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


def setup():
    W = world()
    D = data(W)
    new = np.zeros(D["n_videos"], dtype=bool)
    new[new_videos(W)] = True
    Z = np.load(SIDS)
    return W, D, new, slices(D["clicks"], new), Z


def trained(name, make, train_fn):
    """A model built by make() with the weights train_fn(model) learned, kept in the cache."""
    def run():
        m = make()
        m, curve = train_fn(m)
        return {"state": m.state_dict(), "curve": curve}
    out, sec = cached(name, run)
    m = make()
    m.load_state_dict(out["state"])
    m.eval()
    return m, out["curve"], sec


def reference(D, new, codes):
    return trained("reference", lambda: ItemLM(D["n_videos"], use_id=False, codes=codes, unseen=new),
                   lambda m: fit(m, D["train"], D["valid"], 30, patience=2))


def generator(name, D, new, codes, weights=None, cls=Generator):
    return trained(name, lambda: cls(D["n_videos"], codes, new),
                   lambda m: fit_generator(m, D["train"], D["valid"], weights=weights))


def content_model(D, new, X):
    return trained("content", lambda: ItemLM(D["n_videos"], use_id=False, content=X, unseen=new),
                   lambda m: fit(m, D["train"], D["valid"], 30, patience=2))


@torch.no_grad()
def states(model, test, batch=256):
    """The encoder's state at every scored click, and the clicked video, in order."""
    model.eval()
    H, Y = [], []
    for a in range(0, len(test[0]), batch):
        part = [p[a:a + batch] for p in test]
        x, t, tn, where, y = tensors(*part, model.pad)
        H.append(model.encode(x, t, tn)[where])
        Y.append(y)
    return torch.cat(H), torch.cat(Y)


def ranks(scores, y):
    """How many videos score above the clicked one, and how many tie with it."""
    s = scores.gather(1, y[:, None])
    return {"above": (scores > s).sum(1).numpy(), "ties": ((scores == s).sum(1) - 1).numpy()}


@torch.no_grad()
def exact(model, H, Y, trie, constrained=True, batch=128):
    """The generator's exact ranking of every video for each click: cross-entropy of the
    clicked video, its rank, and the probability left on IDs that name no video."""
    ce, above, ties, mass, top = [], [], [], [], []
    for a in range(0, len(H), batch):
        lp = all_logprob(model, H[a:a + batch], trie, constrained)
        y = Y[a:a + batch]
        ce.append((-lp.gather(1, y[:, None])[:, 0]).numpy())
        r = ranks(lp, y)
        above.append(r["above"])
        ties.append(r["ties"])
        mass.append(torch.exp(lp).sum(1).numpy())
        top.append(lp.topk(10, dim=1).indices)
    return {"ce": np.concatenate(ce), "above": np.concatenate(above), "ties": np.concatenate(ties),
            "mass": np.concatenate(mass), "top10": torch.cat(top)}


@torch.no_grad()
def beams(model, H, trie, width, batch=512, last=None):
    out = []
    for a in range(0, len(H), batch):
        w = (width,) * (len(model.sizes) - 1) + ((last or width),)
        out.append(beam_search(model, H[a:a + batch], trie, w)[0])
    return torch.cat(out)


def found(v, Y, k):
    return (v[:, :k] == Y[:, None]).any(1).numpy().astype(float)


@torch.no_grad()
def cost(model, width=None):
    """Forward FLOPs and milliseconds per request for the output stage alone: scoring
    every video (width None) or a constrained beam search, from 64 requests' states."""
    x, t, tn = request_batch(model.n_videos, 64, 64)
    model.uploaded = True
    h = model.encode(x, t, tn)[:, -1]
    run = (lambda: model.logits(h)) if width is None else \
        (lambda: beam_search(model, h, model._trie, (width,) * len(model.sizes)))
    with FlopCounterMode(display=False) as fc:
        run()
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        run()
        times.append(time.perf_counter() - t0)
    model.uploaded = False
    return fc.get_total_flops() / 64, 1_000 * float(np.median(times)) / 64


@torch.no_grad()
def encoder_cost(model):
    x, t, tn = request_batch(model.n_videos, 64, 64)
    with FlopCounterMode(display=False) as fc:
        model.encode(x, t, tn)
    return fc.get_total_flops() / 64


def example():
    """Section 14.3: two levels of three codes, beam width 2, and a trie in which the
    ID (1, 2) names no video."""
    p1 = np.array([0.5, 0.3, 0.2])
    p2 = {0: np.array([0.1, 0.6, 0.3]), 1: np.array([0.2, 0.1, 0.7]), 2: np.array([0.5, 0.4, 0.1])}
    valid = {(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 0), (2, 1)}
    print("Worked example: beam width 2, level-1 probabilities 0.5, 0.3, 0.2; ID (1, 2) names no video")
    for use_trie in (False, True):
        keep = np.argsort(-p1)[:2]
        print(f"  {'with' if use_trie else 'without'} the trie: level 1 keeps codes {keep.tolist()}")
        cands = []
        for c1 in keep:
            q = p2[int(c1)].copy()
            if use_trie:
                ok = np.array([(int(c1), c2) in valid for c2 in range(3)])
                q = np.where(ok, q, 0.0) / q[ok].sum()
            for c2 in range(3):
                if q[c2] > 0:
                    cands.append(((int(c1), c2), p1[c1] * q[c2]))
        cands.sort(key=lambda z: -z[1])
        print("    " + ", ".join(f"{c} {s:.3f}" for c, s in cands))
        print(f"    kept: {[c for c, _ in cands[:2]]}; names a video: {[c in valid for c, _ in cands[:2]]}")


def line(label, ce, rec, sl, extra=""):
    """One row of the table: each slice's cross-entropy (blank for a beam, which gives no
    probability to most videos) and recall at each k, then the cross-entropy of all."""
    f = lambda x: f"{'':>9}" if np.isnan(x) else f"{x:>9.4f}"                     # noqa: E731
    cells = "".join(f(ce[sl == j].mean()) + "".join(f"{r[sl == j].mean():>8.4f}" for r in rec) for j in range(3))
    print(f"  {label:<36}{cells}{f(ce.mean())}{extra}", flush=True)


HEAD = (f"  {'':<36}" + "".join(f"{n + ' CE':>9}{'R@10':>8}{'R@20':>8}" for n in SLICE_NAMES) + f"{'all CE':>9}")


if __name__ == "__main__":
    t0 = time.time()
    parts = sys.argv[1:] or ["example", "train", "test"]
    print(environment())
    if "example" in parts:
        example()
    W, D, new, sl_v, Z = setup()
    codes = Z["rq_vae_co_clicks"]
    print(f"\nSemantic IDs (chapter_13/sids.npz, rq_vae_co_clicks): {codes.shape[1]} codes, of "
          f"{', '.join(str(int(codes[:, j].max()) + 1) for j in range(codes.shape[1]))} values; "
          f"{len(np.unique(codes[:, :-1], axis=0)):,} distinct sequences of 3 codes for {len(codes):,} videos")
    ref, c_ref, s_ref = reference(D, new, codes)
    gen, c_gen, s_gen = generator("generator", D, new, codes)
    km, c_km, s_km = generator("generator_rq_kmeans", D, new, Z["rq_kmeans"])
    cm, c_cm, s_cm = content_model(D, new, content(W))
    if "train" in parts:
        print("\nTraining (validation loss after each epoch, nats):")
        for label, c, s in (("reference, full softmax", c_ref, s_ref), ("generator", c_gen, s_gen),
                            ("generator, RQ-KMeans IDs", c_km, s_km), ("content model", c_cm, s_cm)):
            print(f"  {label:<26} " + " ".join(f"{v:.4f}" for v in c) + f"; best epoch {int(np.argmin(c)) + 1}  "
                  f"[{s:.0f} s]")
    if "test" not in parts:
        sys.exit()
    old_trie = Trie(codes, ~new, late=new)                    # before day 23: the new videos do not exist
    trie = Trie(codes, np.ones(len(codes), dtype=bool), late=new)
    Hv, Yv = states(gen, D["valid"])
    ev = exact(gen, Hv, Yv, old_trie)
    rec_exact_v = (ev["top10"] == Yv[:, None]).any(1).float().mean().item()
    print(f"\nChoosing the beam width on the validation clicks ({len(Yv):,}): exact top-10 recall "
          f"{rec_exact_v:.4f}")
    chosen = None
    for B in WIDTHS:
        r = found(beams(gen, Hv, old_trie, B), Yv, 10).mean()
        print(f"  beam {B:>4}: recall@10 {r:.4f}")
        if chosen is None and r >= rec_exact_v - 0.005:
            chosen = B
    print(f"  chosen (the smallest within 0.005 of the exact ranking): {chosen}")
    (CACHE / "beam_width.txt").write_text(str(chosen))

    ref.uploaded = True
    r_ref = evaluate(ref, D["test"])
    sl = sl_v[r_ref["video"]]
    H, Y = states(gen, D["test"])
    assert np.array_equal(Y.numpy(), r_ref["video"])
    e = exact(gen, H, Y, trie)
    print(f"\nTest clicks of days 23 to 30: {len(Y):,}; " +
          ", ".join(f"{n} {(sl == j).sum():,}" for j, n in enumerate(SLICE_NAMES)))
    print(HEAD)
    line("reference: full softmax over 3,480", r_ref["ce"], [hit_rate(r_ref, k) for k in TOP], sl)
    line("generator, exact ranking (trie)", e["ce"], [hit_rate(e, k) for k in TOP], sl)
    hm, ym = H[sl == 2], Y[sl == 2]
    bres = {}
    for B in WIDTHS:
        v = beams(gen, H, trie, B, last=max(B, 20))
        bres[B] = v
        agree = (v[:, :10, None] == e["top10"][:, None, :]).any(2).float().mean(1).numpy()
        line(f"generator, beam {B}", np.full(len(Y), np.nan), [found(v, Y, k) for k in TOP], sl,
             f"   agreement with exact top 10 {agree.mean():.4f}")
    plain = exact(gen, H, Y, trie, constrained=False)
    print(f"  (CE: test cross-entropy, nats; R@k: recall@k, ties broken at random for the full rankings; a beam "
          f"of width B returns its best max(B, 20); agreement: share of the exact top 10 that the beam's top 10 holds)")
    print(f"\nWithout the trie: probability on IDs that name no video, mean over test clicks "
          f"{1 - plain['mass'].mean():.4f}; plain cross-entropy {plain['ce'].mean():.4f} against "
          f"{e['ce'].mean():.4f} with the trie")
    v = beams(gen, H, None, 128, last=100)
    for k in (10, 20, 100):
        print(f"  beam 128 without the trie, its top {k}: share that names no video "
              f"{(v[:, :k] < 0).float().mean().item():.4f}; clicks with at least one such ID "
              f"{(v[:, :k] < 0).any(1).float().mean().item():.4f}")

    print("\nCost per request (64 requests of 64 clicks; FLOPs counted by FlopCounterMode):")
    ef = encoder_cost(gen)
    print(f"  encoder (both models, the same HSTU): {ef:,.0f} FLOPs")
    gen._trie = trie
    f, ms = cost(ref)
    print(f"  reference, a score for each of 3,480 videos: {f:,.0f} FLOPs, {ms:.3f} ms")
    for B in WIDTHS:
        f, ms = cost(gen, B)
        print(f"  generator, beam {B:>3} over 4 levels: {f:,.0f} FLOPs, {ms:.3f} ms")

    print("\nOther IDs: the generator on RQ-KMeans IDs (content only, OneRec's tokenizer):")
    print(HEAD)
    trie_km = Trie(Z["rq_kmeans"], np.ones(len(codes), dtype=bool), late=new)
    Hk, Yk = states(km, D["test"])
    ek = exact(km, Hk, Yk, trie_km)
    line("RQ-KMeans generator, exact ranking", ek["ce"], [hit_rate(ek, k) for k in TOP], sl)
    vk = beams(km, Hk, trie_km, chosen, last=20)
    line(f"RQ-KMeans generator, beam {chosen}", np.full(len(Y), np.nan), [found(vk, Yk, k) for k in TOP], sl)

    print("\nNew videos (section 14.4):")
    v = bres[chosen]
    newset = torch.as_tensor(new)
    print(f"  share of the beam's top 10 that are new videos, over all test clicks: "
          f"{newset[v[:, :10].clamp(min=0)].float().mean().item():.4f} (new videos are {new.mean():.3f} of the "
          f"catalogue and {(sl == 2).mean():.4f} of the test clicks)")
    r_cm = evaluate(cm, D["test"])
    print(HEAD)
    line("content model, full softmax", r_cm["ce"], [hit_rate(r_cm, k) for k in TOP], sl)
    Hc, _ = states(cm, D["test"])
    cm.uploaded = True                                         # the test: the new videos exist
    with torch.no_grad():
        lc = torch.cat([cm.logits(Hc[a:a + 512]) for a in range(0, len(Hc), 512)])
    for K in (50,):
        cand = beams(gen, H, trie, max(chosen, K), last=K)
        allowed = torch.zeros(len(Y), len(codes), dtype=torch.bool)
        allowed[torch.arange(len(Y))[:, None], cand.clamp(min=0)] = True
        allowed[:, newset] = True
        s = lc.masked_fill(~allowed, -math.inf)
        top = s.topk(20, dim=1).indices
        line(f"LIGER: beam's {K} + all new, by content", np.full(len(Y), np.nan), [found(top, Y, k) for k in TOP], sl,
             f"   candidates per click {allowed.sum(1).float().mean().item():.0f}")
        print(f"  ({K} generated candidates hold the clicked video for {found(cand, Y, K).mean():.4f} of the clicks)")
    print(f"[{time.time() - t0:.0f} s]")
