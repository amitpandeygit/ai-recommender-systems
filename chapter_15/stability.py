"""Keeping embeddings stable across retraining (section 15.7), in Chapter 9's planted world.

Netflix publishes its foundation model's user and item embeddings every day, and other
teams' models read them as features. A retrained model's embeddings can mean something
else coordinate by coordinate, even when every dot product is as good as before. This
script measures that at Tesselo, with the foundation model of fm_pipeline.py (HSTU,
width 128, 4 layers) and three versions of it:

  A  - pre-trained from scratch on the clicks before day 16 (seed 0): the version a
       downstream team built on
  B  - pre-trained from scratch on the clicks before day 23, with another seed (1): next
       period's version, as after Netflix's monthly pre-training
  C  - A, warm-started and trained for one more epoch on the clicks before day 23: a
       daily fine-tune

Users' embeddings are computed once a day, at midnight, from the clicks before it
(fm_pipeline's daily embeddings); items' are the model's video vectors.

  offline    - every version is stabilised (Listing 15.7) when it is published, on that
               night's embeddings of every consenting user with a history (A at midnight
               of day 16; B and C at midnight of day 23, aligned with A), and the maps are
               then frozen. For every video, the cosine between its vector in A and in B
               (or C), raw and stabilised; the same for each test user's embedding at their
               first test request; how many of each video's 10 nearest videos two
               versions share, raw and stabilised; and how many of each test user's 10
               highest-scoring videos they share, which no map can change
  consumer   - a downstream model (an MLP on the user's and the video's embeddings and the
               heavy ranker's logit, predicting a satisfied play) trained on the impressions
               before day 16, with its epochs chosen on the validation week and its blend
               fitted there as in Listing 9.7. There are two: one trained on A's raw
               embeddings and one on A's stabilised embeddings. In the test week they are
               served with:
                 version A kept                - each consumer with the version it was
                                                 trained on (raw A, or stabilised A)
                 B, raw / C, raw               - the raw consumer with the new version
                                                 swapped in unchanged
                 B, stabilised / C, stabilised - the stabilised consumer with the new
                                                 version stabilised onto A
                 retrained on B                - a new consumer trained on B's raw
                                                 embeddings of the impressions before day
                                                 23, with the raw consumer's epochs and blend
               inside Chapter 9's honest test: Chapter 8's production row, checked, with
               expected satisfied plays, paired differences against A, guardrails. Its
               labels are those the server had by each cutoff (fm_pipeline.observed_rows).

Needs fm_pipeline.py's cache (the foundation model) and PyTorch. Run from the repository
root (about 5 minutes on 2 CPUs, after fm_pipeline.py):
    python chapter_15/stability.py
"""
import os
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "2"
import copy
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
for _ch in ("chapter_12", "chapter_09"):
    sys.path.insert(0, str(HERE.parent / _ch))
from fm_models import stabilise  # noqa: E402
from fm_pipeline import FM, K, cached, day_requests, observed_rows, request_tensors  # noqa: E402
from large_models import fit  # noqa: E402
from worth_it import HSTUSource, fit_hstu, timed_histories, timed_sequences  # noqa: E402
from seq_data import (DAY, WORLDS, eligible_kept, environment, exclude_mask, forbidden_shown, histories,  # noqa: E402
                      load, ndcg, outcomes, paired, ranked_pools, requests)
from pipeline_test import assemble_rows, features, first_allowed, new_candidates  # noqa: E402
from slate_data import rules, split, top_creators  # noqa: E402
from filtering import RuleSets  # noqa: E402
from honest_test import AD_SLOT, CREATOR_CAP, NEW_SLOT  # noqa: E402
from hybrids import fit_logistic  # noqa: E402
from rankers import score_pools  # noqa: E402
from slates import ad_setup, best_ads, check_rows  # noqa: E402
from two_tower import guardrails  # noqa: E402

torch.set_num_threads(2)


@torch.no_grad()
def daily_users(src, W, R):
    """Each request's user embedding, computed at the start of its day from the clicks
    the server had then (zeros, and False, for a request with no such click)."""
    Rd = day_requests(R)
    H, T = timed_histories(W, Rd)
    rows, x, t, tn, last = request_tensors(src, H, T, Rd["time"])
    out = np.zeros((len(H), src.model.d), dtype=np.float32)
    src.model.eval()
    for a in range(0, len(rows), 512):
        h = src.model.encode(x[a:a + 512], t[a:a + 512], tn[a:a + 512])
        out[rows[a:a + 512]] = h[torch.arange(len(h)), last[a:a + 512]].numpy()
    has = np.zeros(len(H), dtype=bool)
    has[rows] = True
    return out, has


@torch.no_grad()
def item_vectors(src):
    src.model.eval()
    return src.model.vectors(torch.arange(len(src.video))).numpy()


class Consumer(nn.Module):
    """A downstream team's model that reads published embeddings as features: an MLP on
    the user's vector, the video's vector and the heavy ranker's logit."""
    def __init__(self, d, hidden=64, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.net = nn.Sequential(nn.Linear(2 * d + 1, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, u, v, heavy):
        return self.net(torch.cat([u, v, heavy[:, None]], 1)).squeeze(1)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_consumer(U, V, hl, y, epochs=20, valid=None, patience=2, seed=0):
    """Adam on the binary cross-entropy of satisfied plays. With valid (the same four
    arrays for the validation week), the epoch with the lowest validation loss is kept and
    training stops after `patience` epochs without a new best; without it, `epochs` are run."""
    m = Consumer(U.shape[1], seed=seed)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    g = torch.Generator().manual_seed(seed)
    U, V, hl, y = (torch.as_tensor(a, dtype=torch.float32) for a in (U, V, hl, y))
    if valid is not None:
        valid = [torch.as_tensor(a, dtype=torch.float32) for a in valid]
    best, best_state, curve = math.inf, None, []
    for _ in range(epochs):
        m.train()
        for b in torch.split(torch.randperm(len(y), generator=g), 512):
            L = F.binary_cross_entropy_with_logits(m(U[b], V[b], hl[b]), y[b])
            opt.zero_grad()
            L.backward()
            opt.step()
        if valid is None:
            continue
        m.eval()
        with torch.no_grad():
            curve.append(F.binary_cross_entropy_with_logits(m(*valid[:3]), valid[3]).item())
        if curve[-1] < best:
            best, best_state = curve[-1], copy.deepcopy(m.state_dict())
        elif len(curve) - 1 - int(np.argmin(curve)) >= patience:
            break
    if best_state is not None:
        m.load_state_dict(best_state)
    m.eval()
    return m, curve


@torch.no_grad()
def consume(m, U, V, hl):
    return m(*(torch.as_tensor(a, dtype=torch.float32) for a in (U, V, hl))).numpy()


def cos_rows(a, b):
    return (a * b).sum(1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-12)


def neighbour_overlap(SA, SB, k=10):
    """For each video, the share of its k nearest other videos (by the similarity matrices
    SA and SB) that the two versions have in common."""
    SA, SB = SA.copy(), SB.copy()
    np.fill_diagonal(SA, -np.inf)
    np.fill_diagonal(SB, -np.inf)
    return top10_overlap(SA, SB, k)


def top10_overlap(A, B, k=10):
    """For each row, the share of its k largest entries in A that are also among B's."""
    ta, tb = np.argsort(-A, axis=1)[:, :k], np.argsort(-B, axis=1)[:, :k]
    return float(np.mean([len(np.intersect1d(x, z)) / k for x, z in zip(ta, tb)]))


if __name__ == "__main__":
    t0 = time.time()
    print(environment())
    name = WORLDS[1]
    W = load(name)
    users, items = W["users"], W["items"]
    n = len(items["vec"])
    fmA, _, epochs, _ = cached("pipe_fm", lambda: fit_hstu(W, *FM, True))

    def version_b():
        src = HSTUSource(W, *FM, True, seed=1)
        src.model = fit(src.model, src.data(*timed_sequences(W, 23 * DAY)), None, epochs)[0]
        return src

    def version_c():
        src = copy.deepcopy(fmA)
        src.model = fit(src.model, src.data(*timed_sequences(W, 23 * DAY)), None, 1)[0]
        return src

    fmB = cached("stab_B", version_b)
    fmC = cached("stab_C", version_c)
    print(f"Versions: A (day 16, seed 0, {epochs} epochs), B (day 23, seed 1, {epochs} epochs, from scratch), "
          f"C (A plus 1 epoch on day 23's clicks, warm start)  [{time.time() - t0:.0f} s]", flush=True)

    # ---- offline: what moves
    R = requests(W, 23, 30)
    VA, VB, VC = item_vectors(fmA), item_vectors(fmB), item_vectors(fmC)
    UA, hasA = daily_users(fmA, W, R)
    UB, _ = daily_users(fmB, W, R)
    UC, _ = daily_users(fmC, W, R)
    first = np.unique(R["user"], return_index=True)[1]          # one request per test user, for user measures
    first = first[hasA[first]]

    def anchor(src, day):
        """The embeddings published at midnight of `day`: every consenting user with a history then."""
        U, has = daily_users(src, W, {"user": W["consenting"], "time": np.full(len(W["consenting"]), day * DAY)})
        return U[has]
    # each version's maps are fitted when it is published, on that night's embeddings, and then frozen
    A_items, _, MTA, MWA = stabilise(VA, anchor(fmA, 16))
    sv = {}
    nA = VA / np.linalg.norm(VA, axis=1, keepdims=True)
    nAs = A_items / np.linalg.norm(A_items, axis=1, keepdims=True)
    for label, src, V, U in (("B", fmB, VB, UB), ("C", fmC, VC, UC)):
        _, _, MT, MW = stabilise(V, anchor(src, 23), A_items)
        sv[label] = (MT, MW)
        Ts = V @ MT
        cos_raw_i, cos_st_i = cos_rows(VA, V).mean(), cos_rows(A_items, Ts).mean()
        cos_raw_u = cos_rows(UA[first], U[first]).mean()
        cos_st_u = cos_rows(UA[first] @ MWA, U[first] @ MW).mean()
        nV = V / np.linalg.norm(V, axis=1, keepdims=True)
        nVs = Ts / np.linalg.norm(Ts, axis=1, keepdims=True)
        print(f"A against {label}: mean cosine of a video's vector, raw {cos_raw_i:+.3f}, stabilised {cos_st_i:+.3f}; "
              f"of a test user's embedding at their first test request, raw {cos_raw_u:+.3f}, "
              f"stabilised {cos_st_u:+.3f}; shared of each video's 10 nearest videos, raw "
              f"{neighbour_overlap(nA @ nA.T, nV @ nV.T):.3f}, stabilised "
              f"{neighbour_overlap(nAs @ nAs.T, nVs @ nVs.T):.3f}; shared of each test user's "
              f"10 highest-scoring videos {top10_overlap(UA[first] @ VA.T, U[first] @ V.T):.3f}")
        dot_err = np.abs((U[first] @ MW) @ Ts.T - U[first] @ V.T).max()
        print(f"  stabilisation changes no score: largest change in a user-video dot product {dot_err:.2e}")
    print("  (maps fitted on the embeddings published at midnight of day 16 for A, and of day 23 for B and C, for "
          "every consenting user with a history, then frozen; nearest videos by cosine, raw or after the maps)")

    # ---- the consumer: trained on A's (raw or stabilised) embeddings of the impressions before day 16
    sp16, sp23 = split(W, "validation"), split(W, "test")

    def impressions(start, end, sp, cutoff):
        Rq = requests(W, start, end)
        Hq, ivq = histories(W, Rq)
        r, c, yy = observed_rows(W, Rq, cutoff)
        hl = logit(score_pools(sp["heavy"], sp["S"], users, items, Rq["user"][r], c[:, None])[:, 0])
        return Rq, Hq, ivq, r, c, yy, hl

    def stab(U, V, maps):
        return (U, V) if maps is None else (U @ maps[1], V @ maps[0])

    tr = impressions(0, 16, sp16, 16 * DAY)
    va = impressions(16, 23, sp16, 23 * DAY)
    UAtr, _ = daily_users(fmA, W, tr[0])
    UAva, _ = daily_users(fmA, W, va[0])
    Rv, Hv, ivv, rv, cv, yv, hlv = va
    cons, wts, curves = {}, {}, {}
    for label, maps in (("raw", None), ("stabilised", (MTA, MWA))):
        U, V = stab(UAtr, VA, maps)
        Uv, Vv = stab(UAva, VA, maps)
        cons[label], curves[label] = cached(f"stab_consumer_{label}_valid", lambda: fit_consumer(
            U[tr[3]], V[fmA.renum[tr[4]]], tr[6], tr[5], valid=(Uv[rv], Vv[fmA.renum[cv]], hlv, yv)))
        sc = consume(cons[label], Uv[rv], Vv[fmA.renum[cv]], hlv)
        wts[label] = fit_logistic(features(1 / (1 + np.exp(-hlv)), sc, ivv[rv]), yv)
    # the consumer retrained on B's embeddings of the impressions before day 23, with the raw consumer's epochs
    e_raw = int(np.argmin(curves["raw"])) + 1
    tr23 = impressions(0, 23, sp23, 23 * DAY)
    UBtr, _ = daily_users(fmB, W, tr23[0])
    cons["retrained on B"] = cached("stab_consumer_B_valid", lambda: fit_consumer(
        UBtr[tr23[3]], VB[fmB.renum[tr23[4]]], tr23[6], tr23[5], epochs=e_raw)[0])
    print(f"Consumers trained on {len(tr[5]):,} impressions before day 16 ({int(tr[5].sum()):,} satisfied); "
          + "; ".join(f"{k}: validation log loss " + " ".join(f"{v:.4f}" for v in c) + f", chosen "
                      f"{int(np.argmin(c)) + 1} epochs" for k, c in curves.items())
          + f"; blends fitted on {len(yv):,} validation impressions  [{time.time() - t0:.0f} s]", flush=True)

    # ---- the test week
    S = sp23["S"]
    H, iv = histories(W, R)
    bad = exclude_mask(W, S, R, H)
    rl = rules(W["cfg"], users, items)
    rs = RuleSets(rl, [np.flatnonzero(S["used"][u]) for u in range(len(users["vec"]))], n, "numpy")
    uu, inv = np.unique(R["user"], return_inverse=True)
    kept = eligible_kept(ranked_pools(sp23, W, rs, uu), inv, bad)
    new = first_allowed(new_candidates(W, S, uu, rl)[inv], bad)
    ads = ad_setup(W, W["cfg"], rl)
    ad = best_ads(W, S, uu, ads)[0][:, 0][inv]
    arms = {"A: Chapter 8's row": (None, None, None, None),
            "consumer, version A kept": ("raw", UA, VA, None),
            "stabilised consumer, A kept": ("stabilised", UA, VA, (MTA, MWA)),
            "consumer, B swapped in raw": ("raw", UB, VB, None),
            "consumer, B stabilised": ("stabilised", UB, VB, sv["B"]),
            "consumer, C swapped in raw": ("raw", UC, VC, None),
            "consumer, C stabilised": ("stabilised", UC, VC, sv["C"]),
            "consumer retrained on B": ("retrained on B", UB, VB, None)}
    rows_out = {k: [] for k in arms}
    for a in range(0, len(R["user"]), 2_048):
        b = min(a + 2_048, len(R["user"]))
        k50 = kept[a:b]
        hpp = score_pools(sp23["heavy"], S, users, items, R["user"][a:b], np.maximum(k50, 0))
        hpp = np.where(k50 >= 0, hpp, np.nan)
        for label, (ck, U, V, maps) in arms.items():
            if ck is None:
                rel = np.where(k50 >= 0, np.nan_to_num(hpp, nan=-np.inf), -np.inf)
            else:
                Uq, Vq = stab(U[a:b], V, maps)
                vid = np.maximum(fmA.renum[np.maximum(k50, 0)], 0)
                hl = logit(np.nan_to_num(hpp, nan=0.5))
                s2 = consume(cons[ck], np.repeat(Uq, K, 0), Vq[vid.ravel()], hl.ravel()).reshape(k50.shape)
                w = wts["raw" if ck == "retrained on B" else ck]
                sc = features(np.nan_to_num(hpp, nan=0.5), s2, iv[a:b, None]) @ w
                rel = np.where(k50 >= 0, np.nan_to_num(sc, nan=-np.inf), -np.inf)
            rows_out[label].append(assemble_rows(k50, rel, new[a:b], ad[a:b], items))
    tc = top_creators(W, S)
    print(f"Test week: {len(R['user']):,} requests  [{time.time() - t0:.0f} s]")
    print(f"  {'':<30}{'NDCG@10':>9}{'coverage':>10}{'top-10% cr.':>12}{'online':>9}{'against A':>18}"
          f"{'95% interval':>20}{'rule breaks':>13}{'not allowed':>13}")
    res, base = {}, None
    for label, parts in rows_out.items():
        top = np.concatenate(parts)
        on = outcomes(W, R, top)[1].sum(axis=1)
        base = on if base is None else base
        vids = np.where((top >= 0) & ~items["is_product"][np.maximum(top, 0)], top, -1)
        cov, cr = guardrails(vids, S["ok"], items["creator"], tc, n)
        breaks = check_rows(top, items, CREATOR_CAP, [(new, NEW_SLOT), (ad, AD_SLOT)])[0]
        shown_bad = forbidden_shown(top, bad, items)[0]
        d = ci = ""
        if on is not base:
            m_, se, lo, hi = paired(on, base, R["user"])
            d, ci = f"{m_:+.4f} ({se:.4f})", f"[{lo:+.4f}, {hi:+.4f}]"
        print(f"  {label:<30}{ndcg(R, top):>9.4f}{cov:>10.1%}{cr:>12.1%}{on.mean():>9.4f}{d:>18}{ci:>20}"
              f"{breaks:>13}{shown_bad:>13}")
        res[label] = on
    print("Direct comparisons (paired over the same requests, users resampled):")
    for a_, b_ in (("consumer, B swapped in raw", "consumer, version A kept"),
                   ("consumer, B stabilised", "stabilised consumer, A kept"),
                   ("consumer, B stabilised", "consumer, B swapped in raw"),
                   ("consumer retrained on B", "consumer, B stabilised"),
                   ("consumer, C swapped in raw", "consumer, version A kept"),
                   ("consumer, C stabilised", "stabilised consumer, A kept"),
                   ("stabilised consumer, A kept", "consumer, version A kept")):
        print(f"  {a_} minus {b_}: " + "{:+.4f} ({:.4f}) [{:+.4f}, {:+.4f}]".format(*paired(res[a_], res[b_],
                                                                                            R["user"])))
    print(f"(the consumer's score enters Listing 9.7's blend in place of the sequence model's; the blend of the "
          f"retrained consumer is the raw consumer's; against A: paired difference, standard error and 95% interval "
          f"from 1,000 bootstrap draws of whole users)  [{time.time() - t0:.0f} s]")
