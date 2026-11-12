"""When scaling is worth the cost (section 12.7).

Part 1, what serving costs. For each size of section 12.5, the forward FLOPs of one
request (a history of 64 clicks, then a score for every video), counted by PyTorch's
FlopCounterMode, and the time per request measured on this machine's 2 CPUs, in
batches of 64 requests (the median of 5 timings).

Part 2, the honest test in Chapter 9's planted world, where a visit's intent and
slow drift make the order of clicks matter (Chapter 2's world has neither, and
Chapter 9 found no sequence model useful there). It is Chapter 9's "blended score"
arm (section 9.7): every row is Chapter 8's production row, with section 8.2's rules,
the creator cap, the "New for you" slot and the ad assembled at once and checked, and
the 50 organic videos ordered by a logistic regression of the heavy ranker's logit and
a sequence model's log probability (with the visit flag terms of Listing 9.7), fitted
on the validation week's held-out impressions. The arms differ only in the sequence
model:

  A: Chapter 8's row     - the heavy ranker alone
  + Chapter 9's model    - section 9.4's choice in this world (seq_data.CHOICE)
  + HSTU, small, ids     - d 32, 1 layer, reading video ids only, as in section 12.5
  + HSTU, small          - d 32, 1 layer
  + HSTU, large          - d 128, 4 layers

Each HSTU reads the last 50 clicks with their times, and the moment of the request.
Except for the "ids" arm, a video's vector adds learned vectors for its creator,
category and language to the vector of its id, as Chapter 9's models do, so that the
HSTU arms read the same inputs as Chapter 9's model. Each HSTU's epochs (up to 30,
the number Chapter 9's model was given, stopping once 2 epochs in a row have not
beaten the best) are chosen on the validation week: fitted on the clicks the server
had by day 16, scored on the clicks of days 16 to 23. It is then refitted on the
clicks it had by day 23 for that many epochs. Differences against A are paired over
the same requests, with standard errors and 95% intervals from 1,000 bootstrap draws
of whole users. Each arm's serving cost is its sequence model's FLOPs and time
per request, measured as in part 1.

Needs Python 3.12 or later, PyTorch and pyroaring. Run from the repository root (about
five minutes):
    python chapter_12/worth_it.py
"""
import os
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "2"
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "chapter_09"))
from large_models import SeqLM, fit, windows  # noqa: E402
from seq_scaling import SIZES  # noqa: E402
from seq_data import (CHOICE, DAY, WORLDS, available_clicks, eligible_kept, environment, exclude_mask,  # noqa: E402
                      forbidden_shown, histories, load, ndcg, outcomes, paired, ranked_pools, requests,
                      training_sequences)
from seq_models import SeqModel, score  # noqa: E402
from seq_models import fit as fit9  # noqa: E402
from pipeline_test import (assemble_rows, features, first_allowed, heldout_rows, log_softmax_rows,  # noqa: E402
                           new_candidates)
from slate_data import rules, split  # noqa: E402
from filtering import RuleSets  # noqa: E402
from honest_test import AD_SLOT, CREATOR_CAP, NEW_SLOT  # noqa: E402
from hybrids import fit_logistic  # noqa: E402
from rankers import score_pools  # noqa: E402
from slates import ad_setup, best_ads, check_rows  # noqa: E402

K = 50
HSTU_ARMS = {"+ HSTU, small, ids": (32, 1, False), "+ HSTU, small": (32, 1, True), "+ HSTU, large": (128, 4, True)}


# ---------------------------------------------------------------- serving cost
def request_batch(n_videos, n=64, k=64, seed=0):
    """A made-up batch of n requests, each with k earlier clicks and their times."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, n_videos, (n, k), generator=g)
    t = torch.sort(torch.rand(n, k, generator=g) * 30 * DAY, dim=1).values
    tn = torch.cat([t[:, 1:], t[:, -1:] + 60.0], 1)
    return x, t, tn


@torch.no_grad()
def serve(model, x, t, tn):
    """The scores of every video for each request: the last position's output."""
    return model.logits(model.encode(x, t, tn)[:, -1])


def serving_cost(model, k=64, n=64):
    """Forward FLOPs and milliseconds per request (median of 5 timings of a batch of n).
    The FLOPs are counted in training mode, where dropout adds none: in evaluation mode
    PyTorch runs SASRec's attention as one fused operation that FlopCounterMode does
    not count. The timings are made in evaluation mode, as a server would run."""
    x, t, tn = request_batch(model.n_videos, n, k)
    model.train()
    with FlopCounterMode(display=False) as fc:
        serve(model, x, t, tn)
    model.eval()
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        serve(model, x, t, tn)
        times.append(time.perf_counter() - t0)
    return fc.get_total_flops() / n, 1_000 * float(np.median(times)) / n


# ---------------------------------------------------------------- sequences with times, under Chapter 9's contract
def timed_sequences(W, cutoff):
    """Each user's clicked videos the server could use at the cutoff, with their times."""
    C = W["clicks"]
    starts = np.searchsorted(C["user"], np.arange(len(W["users"]["vec"]) + 1))
    seqs, times = [], []
    for u in range(len(W["users"]["vec"])):
        idx = available_clicks(C, starts[u], starts[u + 1], cutoff)
        if len(idx) >= 2:
            seqs.append(C["item"][idx])
            times.append(C["t"][idx])
    return seqs, times


def timed_histories(W, R, length=K):
    """For each request, the last `length` clicks the server could use (left-padded with
    -1) and their times."""
    C = W["clicks"]
    starts = np.searchsorted(C["user"], np.arange(len(W["users"]["vec"]) + 1))
    H = np.full((len(R["user"]), length), -1)
    T = np.zeros((len(R["user"]), length))
    for r, (u, t) in enumerate(zip(R["user"], R["time"])):
        idx = available_clicks(C, starts[u], starts[u + 1], t)[-length:]
        if len(idx):
            H[r, -len(idx):] = C["item"][idx]
            T[r, -len(idx):] = C["t"][idx]
    return H, T


class HSTUSource:
    """An HSTU over the world's videos, renumbered 0 to V - 1."""
    def __init__(self, W, d, layers, with_parts, seed=0):
        items = W["items"]
        self.video = np.flatnonzero(~items["is_product"])
        self.renum = np.full(len(items["vec"]), -1)
        self.renum[self.video] = np.arange(len(self.video))
        self.n_items = len(items["vec"])
        parts = {k: items[k][self.video] for k in ("creator", "category", "language")} if with_parts else None
        self.model = SeqLM(len(self.video), "HSTU", d, layers, max_len=K, seed=seed, parts=parts)

    def data(self, seqs, times, lo=-np.inf, hi=np.inf):
        return windows([self.renum[s] for s in seqs], times, K, lo, hi)

    @torch.no_grad()
    def log_probs(self, H, T, t_req):
        """Requests x items: the log probability of every video as the next click
        (products -inf); a request with no history gets the same score for every video."""
        out = np.full((len(H), self.n_items), -np.inf, dtype=np.float32)
        out[:, self.video] = -np.log(len(self.video))
        n = (H >= 0).sum(1)
        for a in range(0, len(H), 512):
            rows = np.flatnonzero(n[a:a + 512] > 0) + a
            if not len(rows):
                continue
            L = int(n[rows].max())
            x = np.full((len(rows), L), self.model.pad)
            t = np.zeros((len(rows), L))
            tn = np.zeros((len(rows), L))
            for j, r in enumerate(rows):
                m = n[r]
                x[j, :m] = self.renum[H[r, -m:]]
                t[j, :m] = T[r, -m:]
                tn[j, :m] = np.r_[T[r, -m:][1:], t_req[r]]          # the request is the moment predicted
                t[j, m:], tn[j, m:] = t[j, m - 1], t_req[r]
            h = self.model.encode(torch.as_tensor(x), torch.as_tensor(t, dtype=torch.float32),
                                  torch.as_tensor(tn, dtype=torch.float32))
            last = torch.as_tensor(n[rows] - 1)
            lp = torch.log_softmax(self.model.logits(h[torch.arange(len(rows)), last]), dim=-1).numpy()
            out[np.ix_(rows, self.video)] = lp
        return out


def fit_hstu(W, d, layers, with_parts):
    """Epochs chosen on the validation week (fitted on day 16, scored on days 16 to 23),
    then refitted on day 23 for that many epochs."""
    src = HSTUSource(W, d, layers, with_parts)
    s16, t16 = timed_sequences(W, 16 * DAY)
    s23, t23 = timed_sequences(W, 23 * DAY)
    src.model, curve = fit(src.model, src.data(s16, t16), src.data(s23, t23, 16 * DAY, 23 * DAY), 30,
                           patience=2)
    best = int(np.argmin(curve)) + 1
    v16 = src
    src = HSTUSource(W, d, layers, with_parts)
    src.model = fit(src.model, src.data(s23, t23), None, best)[0]
    return v16, src, best, curve


# ---------------------------------------------------------------- the honest test
if __name__ == "__main__":
    t0 = time.time()
    print(environment())
    print("Part 1: serving cost per request (64 earlier clicks; a score for every one of 3,480 videos)")
    print(f"  {'':<24}{'params':>11}{'FLOPs/request':>15}{'ms/request':>12}")
    for kind in ("HSTU", "SASRec"):
        for d, L in SIZES:
            m = SeqLM(3_480, kind, d, L, max_len=64)
            f, ms = serving_cost(m)
            print(f"  {kind + ', d ' + str(d) + ', ' + str(L) + ' layer' + 's' * (L > 1):<24}"
                  f"{sum(p.numel() for p in m.parameters()):>11,}{f:>15,.0f}{ms:>12.3f}", flush=True)

    name = WORLDS[1]
    W = load(name)
    users, items = W["users"], W["items"]
    n = len(items["vec"])
    video = ~items["is_product"]
    print(f"\nPart 2: the honest test in Chapter 9's {name}  [{time.time() - t0:.0f} s]")
    kind, loss_name, ep = CHOICE[name]
    models16, models23, cost = {}, {}, {}
    models16["+ Chapter 9's model"] = SeqModel(n, kind, items)
    models16["+ Chapter 9's model"] = fit9(models16["+ Chapter 9's model"], training_sequences(W, 16 * DAY)[0], ep,
                                           loss_name)
    models23["+ Chapter 9's model"] = fit9(SeqModel(n, kind, items), training_sequences(W, 23 * DAY)[0], ep, loss_name)
    print(f"  Chapter 9's model: {kind}, {loss_name}, {ep} epochs")
    for label, (d, L, with_parts) in HSTU_ARMS.items():
        models16[label], models23[label], best, curve = fit_hstu(W, d, L, with_parts)
        print(f"  {label[2:]} (d {d}, {L} layer{'s' * (L > 1)}): validation loss " + " ".join(f"{v:.4f}" for v in curve)
              + f"; chosen {best} epochs  [{time.time() - t0:.0f} s]", flush=True)

    def second(label, m, H, T, t_req):
        if label == "+ Chapter 9's model":
            return log_softmax_rows(score(m, H), video)
        return m.log_probs(H, T, t_req)

    # serving cost of each arm's sequence model (Chapter 9's reads 50 clicks, no times)
    x50 = torch.randint(0, 3_000, (64, K))
    with FlopCounterMode(display=False) as fc:
        score(models23["+ Chapter 9's model"], x50.numpy())
    times = []
    for _ in range(5):
        t1 = time.perf_counter()
        score(models23["+ Chapter 9's model"], x50.numpy())
        times.append(time.perf_counter() - t1)
    cost["+ Chapter 9's model"] = (fc.get_total_flops() / 64, 1_000 * float(np.median(times)) / 64)
    for label in HSTU_ARMS:
        cost[label] = serving_cost(models23[label].model, k=K)

    # blends on the validation week
    sp = split(W, "validation")
    Rv = requests(W, 16, 23)
    Hv, ivv = histories(W, Rv)
    Hv2, Tv = timed_histories(W, Rv)
    rows, cand, y = heldout_rows(W, Rv, Hv, ivv)
    hp = score_pools(sp["heavy"], sp["S"], users, items, Rv["user"][rows], cand[:, None])[:, 0]
    wts = {}
    for label, m in models16.items():
        lp = np.empty(len(rows))
        for a in range(0, len(Rv["user"]), 2_048):
            sel = (rows >= a) & (rows < a + 2_048)
            s = second(label, m, Hv[a:a + 2_048], Tv[a:a + 2_048], Rv["time"][a:a + 2_048])
            lp[sel] = s[rows[sel] - a, cand[sel]]
        wts[label] = fit_logistic(features(hp, lp, ivv[rows]), y)
    print(f"  blends fitted on {len(y):,} validation impressions ({int(y.sum()):,} satisfied)  "
          f"[{time.time() - t0:.0f} s]")

    # the test week
    sp = split(W, "test")
    S = sp["S"]
    R = requests(W, 23, 30)
    H, iv = histories(W, R)
    H2, T = timed_histories(W, R)
    assert np.array_equal(H, H2)
    bad = exclude_mask(W, S, R, H)
    rl = rules(W["cfg"], users, items)
    rs = RuleSets(rl, [np.flatnonzero(S["used"][u]) for u in range(len(users["vec"]))], n, "numpy")
    uu, inv = np.unique(R["user"], return_inverse=True)
    kept = eligible_kept(ranked_pools(sp, W, rs, uu), inv, bad)
    new = first_allowed(new_candidates(W, S, uu, rl)[inv], bad)
    ads = ad_setup(W, W["cfg"], rl)
    ad = best_ads(W, S, uu, ads)[0][:, 0][inv]
    arms = {"A: Chapter 8's row": []} | {label: [] for label in models23}
    for a in range(0, len(R["user"]), 2_048):
        b = min(a + 2_048, len(R["user"]))
        k50 = kept[a:b]
        hpp = score_pools(sp["heavy"], S, users, items, R["user"][a:b], np.maximum(k50, 0))
        hpp = np.where(k50 >= 0, hpp, np.nan)
        rel = np.where(k50 >= 0, np.nan_to_num(hpp, nan=-np.inf), -np.inf)
        arms["A: Chapter 8's row"].append(assemble_rows(k50, rel, new[a:b], ad[a:b], items))
        for label, m in models23.items():
            sec = second(label, m, H[a:b], T[a:b], R["time"][a:b])
            s2 = np.where(k50 >= 0, np.take_along_axis(sec, np.maximum(k50, 0), axis=1), 0.0)
            sc = features(np.nan_to_num(hpp, nan=0.5), s2, iv[a:b, None]) @ wts[label]
            rel = np.where(k50 >= 0, np.nan_to_num(sc, nan=-np.inf), -np.inf)
            arms[label].append(assemble_rows(k50, rel, new[a:b], ad[a:b], items))
    later = R["place"] >= 1
    print(f"Test week: {len(R['user']):,} requests ({later.mean():.1%} later in their visit)  [{time.time() - t0:.0f} s]")
    print(f"  {'':<22}{'NDCG@10':>9}{'online':>9}{'later':>9}{'against A':>18}{'95% interval':>20}"
          f"{'rule breaks':>13}{'not allowed':>13}{'FLOPs/request':>15}{'ms/request':>12}")
    base, res = None, {}
    for label, parts in arms.items():
        top = np.concatenate(parts)
        on = outcomes(W, R, top)[1].sum(axis=1)
        base = on if base is None else base
        breaks = check_rows(top, items, CREATOR_CAP, [(new, NEW_SLOT), (ad, AD_SLOT)])[0]
        shown_bad = forbidden_shown(top, bad, items)[0]
        if on is base:
            d = ci = ""
        else:
            m_, se, lo, hi = paired(on, base, R["user"])
            d, ci = f"{m_:+.4f} ({se:.4f})", f"[{lo:+.4f}, {hi:+.4f}]"
        f, ms = cost.get(label, (0.0, 0.0))
        print(f"  {label:<22}{ndcg(R, top):>9.4f}{on.mean():>9.4f}{on[later].mean():>9.4f}{d:>18}{ci:>20}"
              f"{breaks:>13}{shown_bad:>13}{f:>15,.0f}{ms:>12.3f}")
        res[label] = on
    for a_, b_ in (("+ HSTU, small", "+ HSTU, small, ids"), ("+ HSTU, small", "+ Chapter 9's model"),
                   ("+ HSTU, large", "+ HSTU, small")):
        print(f"  {a_[2:]} against {b_[2:]}: " + "{:+.4f} ({:.4f}) [{:+.4f}, {:+.4f}]".format(
            *paired(res[a_], res[b_], R["user"])))
    print("  (online: expected satisfied plays per row; later: requests after the first of their visit; against A: "
          "paired difference, standard error and 95% interval from 1,000 bootstrap draws of whole users; "
          f"FLOPs and ms: the sequence model's forward pass per request on this machine)  [{time.time() - t0:.0f} s]")
