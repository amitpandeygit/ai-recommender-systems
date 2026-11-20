"""The data of sections 24.4 to 24.6: Tesselo's "New for you" slot, request by request.

Every test of these sections runs in Chapter 2's world with Chapter 9's test contract, as
Chapters 9 to 11 did: models are fitted on what had arrived by the cutoff (day 23 for the
test week, day 16 for the validation week), and each request of the following seven days
is answered with what the server knew before it was served. week(name) builds, for every
request of the week:

  R, H, bad          - the requests, their histories and the videos each may not show
                       (Chapter 9's requests, histories and exclude_mask)
  cols               - the new videos: below Chapter 8's history threshold at the cutoff and
                       not removed (1,716 on the test week)
  static             - today's "New for you" choice: Chapter 11's content model (content
                       mapped into LightGCN's space, the user's language first)
  short              - the content model's 100 best new videos that the request may show,
                       best first (indices into cols): the candidates of every bandit
  content, same      - their content score (without the language bonus) and whether each is
                       in the user's app language
  hidden             - the heavy ranker's last hidden layer (64 numbers) for each candidate,
                       from the cutoff's feature snapshot (for the neural-linear bandit)
  _p, _p_best        - hidden truth: the exact chance of a satisfied play at slot 4 for each
                       candidate, and for the best new video the request may show
  arrive             - when an outcome of the request reaches the server (bandit_world)
  kept, rel, ad, new8 - for the honest test (test week): Chapter 8's 50 kept candidates and
                       their heavy-ranker scores, the ad, and Chapter 8's own new video

Results are kept in chapter_24/cache (one file per week), so they are built once.

Needs Python 3.12 or later, PyTorch, pyroaring and Chapter 11's text_embeddings.npz.
Imported by the other Chapter 24 scripts; run from the repository root to build both
weeks (about 30 minutes):
    python chapter_24/new_video_data.py
"""
import os
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "2"
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
for _ch in ("chapter_11", "chapter_10", "chapter_09", "chapter_08", "chapter_07"):
    sys.path.insert(0, str(ROOT / _ch))
from content_data import MODELS, load, modalities, text_embeddings  # noqa: E402
from content_models import fuse  # noqa: E402
from cold_start import CHOSEN, fit_map, lightgcn_video, mapped  # noqa: E402
from graph_data import DAY, exclude_mask, histories, requests  # noqa: E402
from cold_items import new_video_choice  # noqa: E402
from seq_data import eligible_kept, ranked_pools  # noqa: E402
from pipeline_test import first_allowed, new_candidates  # noqa: E402
from slate_data import rules, split, top_creators  # noqa: E402
from filtering import RuleSets  # noqa: E402
from rankers import score_pools, tensors  # noqa: E402
from slates import ad_setup, best_ads  # noqa: E402
from bandit_world import arrival_times, expected_play, visit_delays  # noqa: E402

CACHE = HERE / "cache"
SHORT = 100                    # candidates per request: the content model's best allowed new videos
SLOT = 4                       # Chapter 8's slot for the new video
WEEKS = {"validation": 16, "test": 23}


def hidden_layer(model, S, users, items, u, i, chunk=50_000):
    """The heavy ranker's last hidden layer for pairs (u[k], i[k]), from snapshot S,
    in chunks of pairs to keep memory small."""
    out = np.empty((len(u), model.body.out_dim), dtype=np.float16)
    with torch.no_grad():
        for a in range(0, len(u), chunk):
            b = min(a + chunk, len(u))
            bb = tensors([S], users, items, u[a:b], i[a:b], np.zeros(b - a, dtype=np.int64))
            if getattr(model.body, "uses_vectors", True):
                vectors, dense = model.inputs(bb)
            else:
                vectors, dense = None, model.inputs.dense(bb)
            out[a:b] = model.body(bb, vectors, dense).numpy().astype(np.float16)
    return out


def rss():
    """Resident memory of this process in GB (Linux), for the progress lines."""
    try:
        return int(open("/proc/self/statm").read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1e9
    except OSError:
        return float("nan")


def build(W, name, X):
    t0 = time.time()
    day = WEEKS[name]
    users, items = W["users"], W["items"]
    n = len(items["vec"])
    sp = split(W, name)
    print(f"  {name}: rankers trained [{time.time() - t0:.0f} s, {rss():.1f} GB]", flush=True)
    S = sp["S"]
    R = requests(W, day, day + 7)
    H, _ = histories(W, R)
    bad = exclude_mask(W, S, R, H)
    print(f"  {name}: requests and rules [{time.time() - t0:.0f} s, {rss():.1f} GB]", flush=True)
    rl = rules(W["cfg"], users, items)
    cols = np.flatnonzero(W["video"] & ~S["ok"] & ~rl["removed"])
    bn = bad[:, cols]
    U, V = lightgcn_video(W, day * DAY, R, H, 0, kg=False)             # Chapter 11's chosen content model
    est = np.flatnonzero(W["video"] & S["ok"])
    P, Vc = U, mapped(X[cols], fit_map(X[est], V[est], CHOSEN["penalty"]))
    print(f"  {name}: content model [{time.time() - t0:.0f} s, {rss():.1f} GB]", flush=True)
    same_all = users["language"][R["user"]][:, None] == items["language"][cols][None, :]
    nr = len(R["user"])
    short = np.full((nr, SHORT), -1)
    content = np.zeros((nr, SHORT), dtype=np.float32)
    static = np.full(nr, -1)
    eff = W["truth"](R["user"], R["time"], R["intent"]).astype(np.float64)
    p_best = np.zeros(nr)
    v_sum, v_n = np.zeros(len(cols)), np.zeros(len(cols))           # each new video's value in its own language
    for a in range(0, nr, 2_048):
        b = min(a + 2_048, nr)
        c = (P[a:b] @ Vc.T).astype(np.float32)
        s = np.where(bn[a:b], -np.inf, c + 1_000.0 * same_all[a:b])
        top = np.argsort(-s, axis=1, kind="stable")[:, :SHORT]
        okk = np.isfinite(np.take_along_axis(s, top, axis=1))
        short[a:b] = np.where(okk, top, -1)
        content[a:b] = np.where(okk, np.take_along_axis(c, top, axis=1), 0.0)
        static[a:b] = new_video_choice(c + 1_000.0 * same_all[a:b], bn[a:b], cols)
        pall = expected_play(eff[a:b, None, :], items, cols[None, :], SLOT, W["cfg"])
        p_best[a:b] = np.where(bn[a:b], -1.0, pall).max(axis=1)
        own = same_all[a:b] & ~bn[a:b]
        v_sum += np.where(own, pall, 0.0).sum(axis=0)
        v_n += own.sum(axis=0)
    has = short >= 0
    vid = np.where(has, cols[np.maximum(short, 0)], 0)
    p = np.where(has, expected_play(eff[:, None, :], items, vid, SLOT, W["cfg"]), 0.0)
    same = np.where(has, users["language"][R["user"]][:, None] == items["language"][vid], False)
    uu = np.repeat(R["user"], SHORT)[has.ravel()]
    hid = np.zeros((nr * SHORT, sp["heavy"].body.out_dim), dtype=np.float16)
    hid[has.ravel()] = hidden_layer(sp["heavy"], S, users, items, uu, vid.ravel()[has.ravel()])
    print(f"  {name}: {nr:,} requests, {len(cols):,} new videos, {has.sum(axis=1).mean():.1f} candidates per request "
          f"[{time.time() - t0:.0f} s]", flush=True)
    rc = visit_delays(W["served"], W["events"])
    out = {"user": R["user"], "time": R["time"], "session": R["session"], "request": R["request"],
           "cols": cols, "short": short, "content": content, "same": same, "static": static,
           "hidden": hid.reshape(nr, SHORT, -1), "_p": p, "_p_best": p_best, "_eff": eff,
           "arrive": arrival_times(R["time"], R["session"], rc),
           "engaged_before": S["engaged"][cols], "language": users["language"][R["user"]],
           "_video_value": v_sum / np.maximum(v_n, 1)}
    if name == "test":                                                # the rest of the row, for section 24.6
        rs = RuleSets(rl, [np.flatnonzero(S["used"][u]) for u in range(len(users["vec"]))], n, "numpy")
        uq, inv = np.unique(R["user"], return_inverse=True)
        kept = eligible_kept(ranked_pools(sp, W, rs, uq), inv, bad)
        rel = np.full(kept.shape, -np.inf)
        for a in range(0, nr, 2_048):
            b = min(a + 2_048, nr)
            hp = score_pools(sp["heavy"], S, users, items, R["user"][a:b], np.maximum(kept[a:b], 0))
            rel[a:b] = np.where(kept[a:b] >= 0, hp, -np.inf)
        new8 = first_allowed(new_candidates(W, S, uq, rl)[inv], bad)
        ads = ad_setup(W, W["cfg"], rl)
        ad = best_ads(W, S, uq, ads)[0][:, 0][inv]
        pos_items = [sorted(x) for x in R["pos"]]
        out.update({"kept": kept, "rel": rel, "ad": ad, "new8": new8, "bid": ads["bid"], "bad": bad,
                    "ok": S["ok"], "top_creators": np.array(sorted(top_creators(W, S))),
                    "pos_ptr": np.r_[0, np.cumsum([len(x) for x in pos_items])],
                    "pos_item": np.array([i for x in pos_items for i in x], dtype=np.int64),
                    "static": np.where(static >= 0, static, new8)})
    print(f"  {name}: done [{time.time() - t0:.0f} s]", flush=True)
    return out


def week(name):
    """One week's data, from the cache if it is there."""
    f = CACHE / f"new_video_{name}.npz"
    if not f.exists():
        build_all()
    z = np.load(f)
    return {k: z[k] for k in z.files}


def content_inputs(W):
    """Chapter 11's fused content vector for every item (text, pictures, video, sound)."""
    n = len(W["items"]["vec"])
    E, idx = text_embeddings(W, MODELS[-1])
    text = E[idx["titles"][np.arange(n), W["items"]["language"]]]
    return fuse([(text, np.ones(n, dtype=bool))] + list(modalities(W).values()), [1.0] * 4)


def build_week(name):
    W = load()
    X = content_inputs(W)
    np.savez(CACHE / f"new_video_{name}.npz", **build(W, name, X))


def build_all():
    """Each week in a process of its own, to keep memory small."""
    import subprocess
    CACHE.mkdir(exist_ok=True)
    for name in WEEKS:
        if not (CACHE / f"new_video_{name}.npz").exists():
            subprocess.run([sys.executable, "-u", __file__, "--week", name], check=True)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--week"]:
        build_week(sys.argv[2])
        sys.exit()
    build_all()
    for name in WEEKS:
        d = week(name)
        print(f"{name}: requests {len(d['user']):,}; new videos {len(d['cols']):,}; best candidate in the shortlist "
              f"{d['_p'].max(axis=1).mean():.4f}, best allowed new video {d['_p_best'].mean():.4f}, static choice "
              f"{d['_p'][:, 0].mean():.4f} (expected satisfied plays at slot {SLOT})")
