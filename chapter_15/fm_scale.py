"""Netflix's foundation model ideas, tested in Chapter 12's scale world (section 15.3).

  tokens     - how often a user clicks the video they clicked just before, which is
               what merging adjacent actions into one token (Netflix's tokenisation)
               would remove, and the worked example of a merge
  staleness  - Netflix computes its members' embeddings in batch and serves them until
               the next batch. At Tesselo a user's embedding is computed once a day, at
               midnight, and used for every request of that day. Chapter 12's HSTU
               (width 64, 2 layers) is trained five ways and tested twice: with the
               embedding computed at each request ("fresh", as in Chapter 12) and at the
               start of the request's day ("daily"). The five ways:
                 next click                  - Chapter 12's loss, each click predicted at
                                               its request
                 next click, daily moments    - the same loss, each click predicted at the
                                               start of its day, as it will be served
                 next 8 clicks, half-life h   - Listing 15.1: the next 8 clicks from one
                                               output, weighted by 0.5^(time after the
                                               prediction / h), for h = 1, 6 and 24 hours
  coldstart  - Netflix blends each title's ID embedding with embeddings of its metadata,
               by the title's age, and masks the ID embedding at random in training. On
               Chapter 13's test of common, rare and 348 new videos (chapter_13/
               sid_data.py), with Chapter 13's content vectors, five ways to make a
               video's vector inside the same HSTU:
                 random ID, content                    - Chapter 13's two references
                 ID + content                          - the two added
                 ID + content, ID masked 10%           - each use of an ID vector replaced,
                                                         with probability 0.1, by one
                                                         learned vector for unknown videos
                 ID + content, gated, ID masked 10%    - Listing 15.2: and the ID part
                                                         weighted by a learned gate on the
                                                         video's training clicks
               The masking rate is the share of the catalogue that is new on day 23 in
               this world (Chapter 13's NEW_SHARE).

Every model trains until its validation loss has not improved for 2 epochs (at most 30)
and keeps its best epoch. The validation loss of the staleness runs is the daily one,
on the held-out users' clicks before day 23. The test is the held-out users' clicks of
days 23 to 30: cross-entropy over all 3,480 videos (nats; lower is better) and recall@20,
with 95% intervals of differences from 1,000 bootstrap draws of whole users.

Results are kept in chapter_15/cache, so an interrupted run resumes. Needs Chapter 12's
scale world (chapter_12/scale_world.py) and, for coldstart, Chapter 13's code. Run from
the repository root (about 1 hour 30 minutes on 2 CPUs; add a part's name to run one):
    python chapter_15/fm_scale.py
"""
import os
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "2"
import pickle
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
for _ch in ("chapter_13", "chapter_12", "chapter_09"):
    sys.path.insert(0, str(HERE.parent / _ch))
from fm_models import DAY, GatedItemLM, day_moments, fit_future, future_data  # noqa: E402
from large_models import SeqLM, fit, windows  # noqa: E402
from scale_world import load, sequences  # noqa: E402
from id_models import ItemLM, evaluate, hit_rate  # noqa: E402
from seq_data import environment  # noqa: E402

K, D_MODEL, LAYERS = 64, 64, 2
J = 8
HALF_LIVES = {"1 hour": 3_600.0, "6 hours": 21_600.0, "24 hours": 86_400.0}
CACHE = HERE / "cache"
TOP = 20


def cached(name, fn):
    """Runs fn() once and keeps its result, with the seconds it took, in
    chapter_15/cache/<name>.pkl. Returns (result, seconds)."""
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


def boot(a, b, user, mask=None, n=1000, seed=0):
    """The mean of a - b over the clicks in mask, and its 95% interval from n draws of
    whole users with replacement (Chapter 13's id_runs.boot)."""
    mask = np.ones(len(a), dtype=bool) if mask is None else mask
    u = user[mask]
    d = np.bincount(u, weights=(a - b)[mask], minlength=user.max() + 1)
    c = np.bincount(u, minlength=user.max() + 1).astype(float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(d), size=(n, len(d)))
    est = d[draws].sum(1) / c[draws].sum(1)
    return d.sum() / c.sum(), *np.percentile(est, [2.5, 97.5])


def boot_users(a, ua, b, ub, n=1000, seed=0):
    """The mean of a over its clicks minus the mean of b over its clicks, when a and b
    score the same clicks in different orders (ua and ub: the user of each), with a 95%
    interval from n draws of whole users with replacement."""
    m = max(ua.max(), ub.max()) + 1
    d = np.bincount(ua, weights=a, minlength=m) - np.bincount(ub, weights=b, minlength=m)
    c = np.bincount(ua, minlength=m).astype(float)
    assert (c == np.bincount(ub, minlength=m)).all()
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, m, size=(n, m))
    est = d[draws].sum(1) / c[draws].sum(1)
    return d.sum() / c.sum(), *np.percentile(est, [2.5, 97.5])


def window_users(S, T, M, lo):
    """The user of each window of windows(S, T, M, K, lo=lo), in order."""
    per = [len(windows([s], [t], [m], K, lo=lo)[0]) for s, t, m in zip(S, T, M)]
    return np.repeat(np.arange(len(per)), per)


# ---------------------------------------------------------------- tokens
def tokens(W):
    u, i = W["user"], W["item"]
    same = u[1:] == u[:-1]
    repeat = same & (i[1:] == i[:-1])
    key = u.astype(np.int64) * 10_000 + i
    again = 1 - len(np.unique(key)) / len(key)
    print(f"Video clicks of consenting users: {len(i):,}; a user clicks the same video they clicked just before for "
          f"{repeat.sum():,} of {same.sum():,} clicks after their first ({repeat.sum() / same.sum():.1%}); "
          f"{again:.1%} of all clicks are of a video the user clicked at some time before")
    # the worked example: one user's history, with adjacent clicks of the same video merged
    s = np.array([7, 7, 3, 7, 7, 7, 5])
    keep = np.r_[True, s[1:] != s[:-1]]
    runs = np.diff(np.r_[np.flatnonzero(keep), len(s)])
    print(f"  worked example: clicks {s.tolist()} -> tokens {s[keep].tolist()} with counts {runs.tolist()}")


# ---------------------------------------------------------------- staleness
def staleness(D):
    S, T, R = D["train"]
    HS, HT, HR = D["held"]
    daily_valid = windows(HS, HT, day_moments(HR), K, hi=23 * DAY)
    fresh_test = windows(HS, HT, HR, K, lo=23 * DAY)
    daily_test = windows(HS, HT, day_moments(HR), K, lo=23 * DAY)
    users = {"fresh": window_users(HS, HT, HR, 23 * DAY), "daily": window_users(HS, HT, day_moments(HR), 23 * DAY)}
    arms = {"next click": ("plain", R), "next click, daily moments": ("plain", day_moments(R))}
    arms |= {f"next {J} clicks, half-life {h}": ("future", hl) for h, hl in HALF_LIVES.items()}
    out = {}
    print(f"\nStaleness: HSTU, width {D_MODEL}, {LAYERS} layers; tested with the user's embedding computed at the "
          f"request (fresh) and at the start of the request's day (daily)")
    print(f"  {'':<34}{'ep':>4}{'best':>5}{'valid (daily)':>15}{'fresh CE':>10}{'R@20':>8}{'daily CE':>10}{'R@20':>8}")
    for label, (kind, arg) in arms.items():
        def train():
            m = SeqLM(D["n_videos"], "HSTU", D_MODEL, LAYERS, max_len=K)
            if kind == "plain":
                m, curve = fit(m, windows(S, T, arg, K), daily_valid, 30, patience=2)
            else:
                m, curve = fit_future(m, future_data(S, T, R, K, J, arg), daily_valid, 30, patience=2)
            return {"curve": curve, "fresh": evaluate(m, fresh_test), "daily": evaluate(m, daily_test)}
        r, sec = cached("stale_" + label.replace(" ", "_").replace(",", ""), train)
        out[label] = r
        cells = "".join(f"{r[w]['ce'].mean():>10.4f}{hit_rate(r[w], TOP).mean():>8.4f}" for w in ("fresh", "daily"))
        print(f"  {label:<34}{len(r['curve']):>4}{int(np.argmin(r['curve'])) + 1:>5}{min(r['curve']):>15.4f}{cells}",
              flush=True)
    print("  (valid: daily next-click cross-entropy on the held-out users' clicks before day 23, which chooses the "
          "epoch; CE: test cross-entropy, nats; R@20: recall@20 of all 3,480 videos)")
    print("What a day of staleness costs each model: daily minus fresh test cross-entropy, over the same "
          f"{len(out['next click']['fresh']['ce']):,} test clicks (95% interval, 1,000 draws of whole users):")
    for label, r in out.items():
        m_, lo, hi = boot_users(r["daily"]["ce"], users["daily"][r["daily"]["window"]], r["fresh"]["ce"],
                                users["fresh"][r["fresh"]["window"]])
        print(f"  {label}: {m_:+.4f} ({lo:+.4f} to {hi:+.4f})")
    base = out["next click"]
    print("Differences in test cross-entropy against next click (95% interval, 1,000 draws of whole users):")
    for w in ("fresh", "daily"):
        user = users[w][base[w]["window"]]
        for label in list(arms)[1:]:
            m_, lo, hi = boot(out[label][w]["ce"], base[w]["ce"], user)
            print(f"  {w}: {label} minus next click: {m_:+.4f} ({lo:+.4f} to {hi:+.4f})")
    return out


# ---------------------------------------------------------------- cold start
def coldstart():
    from sid_data import NEW_SHARE, SLICE_NAMES, content, data, new_videos, slices, world
    W = world()
    X = content(W)
    D = data(W)
    new = np.zeros(D["n_videos"], dtype=bool)
    new[new_videos(W)] = True
    sl = slices(D["clicks"], new)
    arms = {
        "random ID": lambda: ItemLM(D["n_videos"], unseen=new),
        "content": lambda: ItemLM(D["n_videos"], use_id=False, content=X, unseen=new),
        "ID + content": lambda: GatedItemLM(D["n_videos"], X, D["clicks"], new, gate=False),
        f"ID + content, ID masked {NEW_SHARE:.0%}": lambda: GatedItemLM(D["n_videos"], X, D["clicks"], new,
                                                                     mask=NEW_SHARE, gate=False),
        f"ID + content, gated, ID masked {NEW_SHARE:.0%}": lambda: GatedItemLM(D["n_videos"], X, D["clicks"], new,
                                                                            mask=NEW_SHARE),
    }
    print(f"\nCold start: Chapter 13's test, {D['n_videos']:,} videos, {new.sum():,} of them new on day 23")
    print(f"  {'':<38}{'ep':>4}{'best':>5}" + "".join(f"{n + ' CE':>10}{'R@20':>8}" for n in SLICE_NAMES)
          + f"{'all CE':>9}")
    R = {}
    for label, make in arms.items():
        def train():
            m, curve = fit(make(), D["train"], D["valid"], 30, patience=2)
            out = {"eval": evaluate(m, D["test"]), "curve": curve}
            if isinstance(m, GatedItemLM):
                out["gate"] = (float(m.a.detach()), float(m.b.detach()))
            return out
        r, sec = cached("cold_" + label.replace(" ", "_").replace(",", "").replace("%", ""), train)
        e = r["eval"]
        s = sl[e["video"]]
        cells = "".join(f"{e['ce'][s == j].mean():>10.4f}{hit_rate(e, TOP)[s == j].mean():>8.4f}" for j in range(3))
        print(f"  {label:<38}{len(r['curve']):>4}{int(np.argmin(r['curve'])) + 1:>5}{cells}{e['ce'].mean():>9.4f}",
              flush=True)
        R[label] = r
    gated = R[list(arms)[-1]]["gate"]
    q = np.percentile(np.log1p(D["clicks"][~new]), [0, 50, 100])
    g = 1 / (1 + np.exp(-(gated[0] * q + gated[1])))
    n = np.expm1(q)
    print(f"  gate: sigmoid({gated[0]:.3f} x log(1 + clicks) {'+' if gated[1] >= 0 else '-'} {abs(gated[1]):.3f}); "
          f"at 0 clicks {1 / (1 + np.exp(-gated[1])):.3f}; at the fewest, median and most training clicks of a warm "
          f"video ({n[0]:,.0f}, {n[1]:,.0f} and {n[2]:,.0f}) {g[0]:.3f}, {g[1]:.3f}, {g[2]:.3f}")
    s = sl[R["random ID"]["eval"]["video"]]
    print("  (CE: test cross-entropy; R@20: recall@20, ties broken at random; test clicks: "
          + ", ".join(f"{(s == j).sum():,} {name}" for j, name in enumerate(SLICE_NAMES)) + ")")
    user = D["test_user"][R["random ID"]["eval"]["window"]]
    print("Difference in test cross-entropy, by slice (95% interval, 1,000 draws of whole users):")
    for base in ("random ID", "content"):
        for label in list(arms)[2:]:
            for j, name in enumerate(SLICE_NAMES):
                mask = sl[R[base]["eval"]["video"]] == j
                m_, lo, hi = boot(R[label]["eval"]["ce"], R[base]["eval"]["ce"], user, mask)
                print(f"  {name}: {label} minus {base}: {m_:+.4f} ({lo:+.4f} to {hi:+.4f})")
    a_, b_ = list(arms)[-1], list(arms)[-2]
    for j, name in enumerate(SLICE_NAMES):
        mask = sl[R[b_]["eval"]["video"]] == j
        m_, lo, hi = boot(R[a_]["eval"]["ce"], R[b_]["eval"]["ce"], user, mask)
        print(f"  {name}: {a_} minus {b_}: {m_:+.4f} ({lo:+.4f} to {hi:+.4f})")
    return R


if __name__ == "__main__":
    t0 = time.time()
    parts = sys.argv[1:] or ["tokens", "staleness", "coldstart"]
    print(environment())
    W = load()
    if "tokens" in parts:
        tokens(W)
    if "staleness" in parts:
        staleness(sequences(W))
    if "coldstart" in parts:
        coldstart()
    print(f"[{time.time() - t0:.0f} s]")
