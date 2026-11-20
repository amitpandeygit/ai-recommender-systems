"""Three ways to use a foundation model, distillation, and the honest test, in Chapter 9's
planted world (sections 15.5, 15.6 and 15.8).

The foundation model is Chapter 12's large HSTU (width 128, 4 layers, each video's
vector the sum of vectors for its ID, creator, category and language), pre-trained on
every consenting user's clicks to predict the next click, with its epochs chosen on the
validation week (chapter_12/worth_it.fit_hstu). The downstream task is the one Tesselo's
row cares about: will a video shown in a request be satisfied (completed or liked)?
It has far fewer labels than the clicks: Chapter 9's impressions, the videos of a
request that were on screen or clicked.

Each arm gives one second score, which the blend of Listing 9.7 combines with the heavy
ranker's logit, fitted on the validation week's impressions; the arms differ only in it:

  + Chapter 9's model          - GRU4Rec, Chapter 9's choice in this world (the
                                 production blend since Chapter 12)
  + FM, request time           - the foundation model's own prediction of the next click,
                                 read at each request (Chapter 12's "+ HSTU, large")
  + FM embeddings, daily       - way 1: the user's embedding computed at the start of the
                                 request's day, in batch, scored against the model's item
                                 vectors; the request itself only reads stored vectors
  + FM subgraph                - way 2, Listing 15.3: the foundation model's layers inside
                                 a downstream model that scores (request, video) pairs for a
                                 satisfied play, trained on impressions, the layers
                                 fine-tuned with a learning rate 10 times smaller
  + FM fine-tuned              - way 3: the foundation model itself, trained further to
                                 predict the next satisfied play instead of the next click
  control: subgraph, no pre-training   - way 2's model from random weights
  control: fine-tuned, no pre-training - way 3's model from random weights
  + student                    - section 15.6: a small model (HSTU width 32, 1 layer, and a
                                 head) trained on the impressions alone
  + student, distilled, candidates - Listing 15.4: the same, also matching the teacher's
                                 scores over each training request's 50 candidates
  + student, distilled, all videos - the same over every video (Chapter 12's way); the
                                 teacher's scores are saved once (Listing 15.5)

The teacher is whichever of ways 2 and 3 has the lower validation log loss in its blend.
Every model is fitted on what had arrived by day 16 to choose its epochs and blend, and
refitted on day 23 for the test (Chapter 9's contract). The test is Chapter 9's honest
test: Chapter 8's production row with all its rules, checked; expected satisfied plays
per row; paired differences against A with 95% intervals from 1,000 bootstrap draws of
whole users; Chapter 5's guardrails; each app language; and each arm's compute per
request.

Results are kept in chapter_15/cache. Needs PyTorch and pyroaring. Run from the
repository root (about 50 minutes on 2 CPUs):
    python chapter_15/fm_pipeline.py
"""
import os
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "2"
import copy
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.flop_counter import FlopCounterMode

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
for _ch in ("chapter_12", "chapter_09"):
    sys.path.insert(0, str(HERE.parent / _ch))
from large_models import SeqLM, fit, windows  # noqa: E402
from worth_it import fit_hstu, timed_histories, timed_sequences  # noqa: E402
from seq_data import (CHOICE, DAY, HITS, WORLDS, available_clicks, eligible_kept, environment,  # noqa: E402
                      exclude_mask, forbidden_shown, histories, load, ndcg, outcomes, paired, ranked_pools, requests,
                      training_sequences)
from seq_models import SeqModel, score  # noqa: E402
from seq_models import fit as fit9  # noqa: E402
from pipeline_test import assemble_rows, features, first_allowed, heldout_rows, log_softmax_rows, new_candidates  # noqa: E402
from slate_data import rules, split, top_creators  # noqa: E402
from ranking_data import LANGUAGES  # noqa: E402
from filtering import RuleSets  # noqa: E402
from honest_test import AD_SLOT, CREATOR_CAP, NEW_SLOT  # noqa: E402
from hybrids import fit_logistic  # noqa: E402
from rankers import score_pools  # noqa: E402
from slates import ad_setup, best_ads, check_rows  # noqa: E402
from two_tower import guardrails  # noqa: E402

torch.set_num_threads(2)
K = 50                                                 # clicks read, and candidates per request
FM, STUDENT = (128, 4), (32, 1)
CACHE = HERE / "cache"
KD_WEIGHT, KD_TEMP = 0.5, 2.0


def cached(name, fn):
    """Runs fn() once and keeps its result in chapter_15/cache/<name>.pkl."""
    f = CACHE / f"{name}.pkl"
    if f.exists():
        with open(f, "rb") as h:
            return pickle.load(h)
    out = fn()
    CACHE.mkdir(exist_ok=True)
    with open(f.with_suffix(".tmp"), "wb") as h:
        pickle.dump(out, h)
    f.with_suffix(".tmp").rename(f)
    return out


# ---------------------------------------------------------------- the requests' histories as tensors
def request_tensors(src, H, T, t_req):
    """Right-padded histories of the requests that have at least one click (rows), their
    times, the moment each position predicts at (the request for the last one) and the
    position of the last click."""
    n = (H >= 0).sum(1)
    rows = np.flatnonzero(n > 0)
    L = int(max(n[rows].max(), 1)) if len(rows) else 1
    x = np.full((len(rows), L), src.model.pad)
    t = np.zeros((len(rows), L))
    tn = np.zeros((len(rows), L))
    for j, r in enumerate(rows):
        m = n[r]
        x[j, :m] = src.renum[H[r, -m:]]
        t[j, :m] = T[r, -m:]
        tn[j, :m] = np.r_[T[r, -m:][1:], t_req[r]]
        t[j, m:], tn[j, m:] = t[j, m - 1], t_req[r]
    return rows, torch.as_tensor(x), torch.as_tensor(t, dtype=torch.float32), \
        torch.as_tensor(tn, dtype=torch.float32), torch.as_tensor(n[rows] - 1)


class PairScorer(nn.Module):
    """Listing 15.3. A downstream model that holds a sequence model as a subgraph: the
    history is encoded once per request, and a small head scores each candidate video
    from the request's vector h, the video's vector v and their product. With the
    foundation model as `seq`, this is Netflix's subgraph; with a small new model, it is
    section 15.6's student."""
    def __init__(self, seq, hidden=64, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.seq = seq
        d = seq.d
        self.head = nn.Sequential(nn.Linear(3 * d, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def user(self, x, t, tn, last):
        """One vector per request: the sequence model's output at its last click."""
        return self.seq.encode(x, t, tn)[torch.arange(len(x)), last]

    def pairs(self, h, v):
        """h and v broadcast to the same shape (..., d); one logit per pair."""
        h, v = torch.broadcast_tensors(h, v)
        return self.head(torch.cat([h, v, h * v], -1)).squeeze(-1)


class Requests:
    """Requests of days [start_day, end_day) for a downstream model: each request's history
    (the clicks the server had when it was served, as in Chapter 9's contract), its
    impressions with their satisfied plays and, if given, its 50 candidates.

    For speed, the histories are encoded in shared rows: one row per user holds the user's
    clicks in order, and a request whose history is the first n of them (and n <= 50)
    reads position n - 1 of that row, as Chapter 12's windows read one row for many
    clicks. A request whose history is not such a prefix (a click that arrived late), or
    is longer than 50 clicks, gets a row of its own with its last 50 clicks. In a shared
    row, the moment a position predicts at is the earliest request that reads it, so a
    later request reading the same position sees the same output; the serving code
    (pair_scores) encodes every request on its own, at its own moment."""
    def __init__(self, W, src, start_day, end_day, pools=None):
        R = requests(W, start_day, end_day)
        H, iv = histories(W, R)
        C = W["clicks"]
        starts = np.searchsorted(C["user"], np.arange(len(W["users"]["vec"]) + 1))
        rows_x, rows_t, rows_m = [], [], []
        self.req_row = np.full(len(R["user"]), -1)
        self.req_pos = np.full(len(R["user"]), -1)
        own = 0
        for u in np.unique(R["user"]):
            ks = np.flatnonzero(R["user"] == u)
            ks = ks[np.argsort(R["time"][ks], kind="stable")]
            full = available_clicks(C, starts[u], starts[u + 1], end_day * DAY)[:K]
            moment = np.r_[C["t"][full][1:], C["t"][full][-1:]] if len(full) else np.zeros(0)
            used = -1
            for k in ks:
                av = available_clicks(C, starts[u], starts[u + 1], R["time"][k])
                n = len(av)
                if n == 0:
                    continue
                if n <= len(full) and np.array_equal(full[:n], av):
                    if self.req_row[k] < 0 and used < n - 1:
                        moment[n - 1] = R["time"][k]                     # the earliest request that reads it
                        used = n - 1
                    self.req_row[k], self.req_pos[k] = -2, n - 1         # the user's shared row, set below
                else:
                    av = av[-K:]
                    rows_x.append(src.renum[C["item"][av]])
                    rows_t.append(C["t"][av])
                    rows_m.append(np.r_[C["t"][av][1:], R["time"][k]])
                    self.req_row[k], self.req_pos[k] = len(rows_x) - 1, len(av) - 1
                    own += 1
            mine = ks[self.req_row[ks] == -2]
            if len(mine):
                e = int(self.req_pos[mine].max()) + 1
                rows_x.append(src.renum[C["item"][full[:e]]])
                rows_t.append(C["t"][full[:e]])
                rows_m.append(moment[:e])
                self.req_row[mine] = len(rows_x) - 1
        self.rows_x, self.rows_t, self.rows_m, self.pad = rows_x, rows_t, rows_m, src.model.pad
        self.R, self.iv, self.own = R, iv, own
        self.rows = np.flatnonzero(self.req_row >= 0)                     # requests with a history
        self.where = np.full(len(R["user"]), -1)
        self.where[self.rows] = np.arange(len(self.rows))
        r, c, y = heldout_rows(W, R, H, iv)
        keep = self.where[r] >= 0
        self.imp_req, self.imp_cand, self.y = r[keep], src.renum[c[keep]], y[keep]
        self.n_all, self.pos_all = len(y), int(y.sum())
        self.by_row = {}
        for k in self.rows:
            self.by_row.setdefault(int(self.req_row[k]), []).append(k)
        self.imps = {}
        for j, k in enumerate(self.imp_req):
            self.imps.setdefault(int(k), []).append(j)
        self.pools = None
        if pools is not None:                                     # each request's 50 candidates, as videos
            self.pools = src.renum[np.maximum(pools, 0)]
            self.pools[(pools < 0) | (self.pools < 0)] = -1

    def encode(self, seq, req):
        """The sequence model's output for each request in req, from the shared rows."""
        rows, inv = np.unique(self.req_row[req], return_inverse=True)
        L = max(len(self.rows_x[r]) for r in rows)
        x = np.full((len(rows), L), self.pad)
        t = np.zeros((len(rows), L))
        m = np.zeros((len(rows), L))
        for j, r in enumerate(rows):
            n = len(self.rows_x[r])
            x[j, :n], t[j, :n], m[j, :n] = self.rows_x[r], self.rows_t[r], self.rows_m[r]
            t[j, n:], m[j, n:] = t[j, n - 1], m[j, n - 1]
        h = seq.encode(torch.as_tensor(x), torch.as_tensor(t, dtype=torch.float32),
                       torch.as_tensor(m, dtype=torch.float32))
        return h[torch.as_tensor(inv.ravel()), torch.as_tensor(self.req_pos[req])]

    def batches(self, batch, gen=None, labelled=True):
        """Requests in batches of `batch` rows (shuffled with a generator); with labelled,
        only requests that have impressions."""
        rows = np.array(sorted(self.by_row))
        if gen is not None:
            rows = rows[torch.randperm(len(rows), generator=gen).numpy()]
        for a in range(0, len(rows), batch):
            req = np.concatenate([self.by_row[int(r)] for r in rows[a:a + batch]])
            if labelled:
                req = req[[int(k) in self.imps for k in req]]
            if len(req):
                yield np.sort(req)


def bce_epoch(model, data, opt, gen, teacher=None, kd="none", batch=64):
    """One epoch of the downstream loss: binary cross-entropy of satisfied plays on the
    impressions, plus, with a teacher, Listing 15.4's distillation term."""
    model.train()
    for req in data.batches(batch, gen):
        idx = np.concatenate([data.imps[int(k)] for k in req])
        pos_in_batch = np.repeat(np.arange(len(req)), [len(data.imps[int(k)]) for k in req])
        h = data.encode(model.seq, req)
        v = model.seq.vectors(torch.as_tensor(data.imp_cand[idx]))
        logit = model.pairs(h[pos_in_batch], v)
        L = F.binary_cross_entropy_with_logits(logit, torch.as_tensor(data.y[idx], dtype=torch.float32))
        if teacher is not None:
            if kd == "candidates":
                c = torch.as_tensor(np.maximum(data.pools[req], 0))                     # requests x 50
                valid = torch.as_tensor(data.pools[req] >= 0)
            else:
                c = torch.arange(model.seq.n_videos)[None].expand(len(req), -1)
                valid = torch.ones_like(c, dtype=torch.bool)
            has = valid.any(1)
            h_, c, valid, req = h[has], c[has], valid[has], req[has.numpy()]
            s = model.pairs(h_[:, None, :], model.seq.vectors(c))
            p = torch.as_tensor(teacher[data.where[req]], dtype=torch.float32)          # saved once
            if kd == "candidates":
                p = p.masked_fill(~valid, 0.0)
            L = (1 - KD_WEIGHT) * L + KD_WEIGHT * KD_TEMP ** 2 * distill_over(s, p, valid)
        opt.zero_grad()
        L.backward()
        opt.step()


def distill_over(student, teacher, valid, temp=KD_TEMP):
    """Listing 15.4. The KL divergence from the teacher's softened distribution over each
    request's candidates to the student's (Chapter 12's distill_loss, over a set of
    candidates instead of every video); invalid candidates are left out of both."""
    s = (student / temp).masked_fill(~valid, -math.inf)
    p = F.softmax((teacher / temp).masked_fill(~valid, -math.inf), dim=-1)
    logq = F.log_softmax(s, dim=-1).masked_fill(~valid, 0.0)
    return (p * (torch.log(p.clamp_min(1e-30)) - logq)).sum(-1).mean()


@torch.no_grad()
def teacher_table(teacher, data, kd, batch=64):
    """Listing 15.5. The teacher's score of every candidate of every request of data,
    computed once and kept in 16-bit floats (requests x 50 for kd "candidates", requests x
    every video for "all"), so that training the student never runs the teacher again."""
    kind, m = teacher
    m.eval()
    seq = m.seq if kind == "subgraph" else m
    width = K if kd == "candidates" else seq.n_videos
    out = np.zeros((len(data.rows), width), dtype=np.float16)
    for req in data.batches(batch, labelled=False):
        h = data.encode(seq, req)
        if kd == "candidates":
            c = torch.as_tensor(np.maximum(data.pools[req], 0))
        else:
            c = torch.arange(width)[None].expand(len(req), -1)
        for j in range(0, len(req), 16):                                # 16 requests at a time
            if kind == "subgraph":
                s = m.pairs(h[j:j + 16, None, :], m.seq.vectors(c[j:j + 16]))
            else:
                s = torch.gather(m.logits(h[j:j + 16]), 1, c[j:j + 16])
            out[data.where[req[j:j + 16]]] = s.numpy().astype(np.float16)
    return out


@torch.no_grad()
def bce_loss(model, data, batch=256):
    """Mean binary cross-entropy of the impressions of data."""
    model.eval()
    total = 0.0
    for req in data.batches(batch):
        idx = np.concatenate([data.imps[int(k)] for k in req])
        pos = np.repeat(np.arange(len(req)), [len(data.imps[int(k)]) for k in req])
        h = data.encode(model.seq, req)
        logit = model.pairs(h[pos], model.seq.vectors(torch.as_tensor(data.imp_cand[idx])))
        total += F.binary_cross_entropy_with_logits(logit, torch.as_tensor(data.y[idx], dtype=torch.float32),
                                                    reduction="sum").item()
    return total / len(data.y)


def fit_pairs(model, train, valid, epochs, lr_seq=1e-3, lr_head=1e-3, teacher=None, kd="none", seed=0, patience=2):
    """Adam on the downstream loss; with valid, the epoch with the lowest validation loss
    is kept and training stops after `patience` epochs without a new best."""
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam([{"params": model.seq.parameters(), "lr": lr_seq},
                            {"params": model.head.parameters(), "lr": lr_head}])
    curve, best, state = [], math.inf, None
    for e in range(1, epochs + 1):
        bce_epoch(model, train, opt, gen, teacher, kd, batch=8 if kd == "all" else 64)
        if valid is None:
            continue
        v = bce_loss(model, valid)
        curve.append(v)
        if v < best:
            best, state = v, copy.deepcopy(model.state_dict())
        elif e - (int(np.argmin(curve)) + 1) >= patience:
            break
    if state is not None:
        model.load_state_dict(state)
    model.eval()
    return model, curve


# ---------------------------------------------------------------- fine-tuning on satisfied plays
def satisfied_sequences(W, cutoff):
    """timed_sequences(W, cutoff), with a flag on each click that was satisfied (completed
    or liked) by an event the server had received by the cutoff."""
    ev, C = W["events"], W["clicks"]
    hit = np.isin(ev["event"], HITS) & (ev["item_id"] >= 0) & (ev["server_time"] < cutoff)
    keys = set(zip(ev["session_id"][hit].tolist(), ev["item_id"][hit].tolist()))
    starts = np.searchsorted(C["user"], np.arange(len(W["users"]["vec"]) + 1))
    seqs, times, sat = [], [], []
    for u in range(len(W["users"]["vec"])):
        idx = available_clicks(C, starts[u], starts[u + 1], cutoff)
        if len(idx) >= 2:
            seqs.append(C["item"][idx])
            times.append(C["t"][idx])
            sat.append(np.array([(s, i) in keys for s, i in zip(C["session"][idx].tolist(), C["item"][idx].tolist())]))
    return seqs, times, sat


def satisfied_windows(src, seqs, times, sat, lo=-math.inf, hi=math.inf):
    """Chapter 12's windows with only the satisfied clicks scored as targets."""
    starts = np.r_[0, np.cumsum([len(s) for s in seqs])]
    G = src.renum[np.concatenate(seqs)]
    ok = np.concatenate(sat)
    idx = [np.arange(a, b) for a, b in zip(starts[:-1], starts[1:])]
    xs, ts, ms, at, ys, sc = windows(idx, times, None, K, lo, hi)
    keep = [k for k, (y, s) in enumerate(zip(ys, sc)) if (s & ok[y]).any()]
    return ([G[xs[k]] for k in keep], [ts[k] for k in keep], [ms[k] for k in keep], [at[k] for k in keep],
            [G[ys[k]] for k in keep], [sc[k] & ok[ys[k]] for k in keep])


# ---------------------------------------------------------------- second scores
@torch.no_grad()
def pair_scores(model, src, H, T, t_req):
    """Requests x items: the downstream model's logit for every video (products -inf; 0
    for requests with no history, so the order is the heavy ranker's)."""
    out = np.full((len(H), len(src.renum)), -np.inf, dtype=np.float32)
    out[:, src.video] = 0.0
    model.eval()
    rows, x, t, tn, last = request_tensors(src, H, T, t_req)
    V = model.seq.vectors(torch.arange(len(src.video)))
    for a in range(0, len(rows), 256):
        h = model.user(x[a:a + 256], t[a:a + 256], tn[a:a + 256], last[a:a + 256])
        for j in range(0, len(h), 16):                                  # 16 requests x every video at a time
            out[np.ix_(rows[a + j:a + j + 16], src.video)] = model.pairs(h[j:j + 16, None, :], V[None]).numpy()
    return out


def day_requests(R):
    """The same requests, as seen at the start of their day (a daily batch)."""
    Rd = dict(R)
    Rd["time"] = np.floor(R["time"] / DAY) * DAY
    return Rd


def flops(fn):
    with FlopCounterMode(display=False) as fc:
        fn()
    return fc.get_total_flops()


# ---------------------------------------------------------------- main
def new_seq(src, d, layers):
    return SeqLM(src.model.n_videos, "HSTU", d, layers, max_len=K,
                 parts={k: src_items[k][src.video] for k in ("creator", "category", "language")})


def with_model(src, m):
    s = copy.copy(src)
    s.model = m
    return s


def serving_flops(label, m, src):
    """Forward FLOPs of the second score for one request with 50 earlier clicks: the
    sequence model's pass, and either the scores of every video (models that give a log
    probability) or the head over the request's 50 candidates (pair models). The daily
    embeddings arm reads a stored user vector and computes 50 dot products."""
    g = torch.Generator().manual_seed(0)
    x = torch.randint(0, len(src.video), (64, K), generator=g)
    t = torch.sort(torch.rand(64, K, generator=g) * 30 * DAY, dim=1).values
    tn = torch.cat([t[:, 1:], t[:, -1:] + 60.0], 1)
    if label == "+ Chapter 9's model":
        return flops(lambda: score(m, x.numpy())) / 64
    if label == "+ FM embeddings, daily":
        return 2 * src.model.d * K
    seq = m.seq if isinstance(m, PairScorer) else m
    seq.train()
    with torch.no_grad():
        if isinstance(m, PairScorer):
            c = torch.randint(0, len(src.video), (64, K), generator=g)
            f = flops(lambda: m.pairs(m.user(x, t, tn, torch.full((64,), K - 1))[:, None, :], m.seq.vectors(c)))
        else:
            f = flops(lambda: m.logits(m.encode(x, t, tn)[:, -1]))
    seq.eval()
    return f / 64


if __name__ == "__main__":
    t0 = time.time()
    print(environment())
    name = WORLDS[1]
    W = load(name)
    users, items = W["users"], W["items"]
    src_items = items
    n = len(items["vec"])
    video = ~items["is_product"]
    print(f"Chapter 9's {name}  [{time.time() - t0:.0f} s]")
    fm16, fm23, fm_ep, fm_curve = cached("pipe_fm", lambda: fit_hstu(W, *FM, True))
    print(f"Foundation model: HSTU, width {FM[0]}, {FM[1]} layers, {sum(p.numel() for p in fm23.model.parameters()):,} "
          f"weights; next-click validation loss " + " ".join(f"{v:.4f}" for v in fm_curve)
          + f"; chosen {fm_ep} epochs  [{time.time() - t0:.0f} s]", flush=True)
    kind, loss_name, ep9 = CHOICE[name]
    gru16, gru23 = cached("pipe_gru", lambda: (
        fit9(SeqModel(n, kind, items), training_sequences(W, 16 * DAY)[0], ep9, loss_name),
        fit9(SeqModel(n, kind, items), training_sequences(W, 23 * DAY)[0], ep9, loss_name)))

    # ---- the downstream data: impressions with satisfied plays, and each request's 50 candidates
    def snapshot_pools(sp, R):
        S = sp["S"]
        rs = RuleSets(rules(W["cfg"], users, items), [np.flatnonzero(S["used"][u]) for u in range(len(users["vec"]))],
                      n, "numpy")
        uu, inv = np.unique(R["user"], return_inverse=True)
        return ranked_pools(sp, W, rs, uu)[:, :K][inv]

    sp16, sp23 = split(W, "validation"), split(W, "test")
    tr16 = Requests(W, fm16, 0, 16, snapshot_pools(sp16, requests(W, 0, 16)))
    va16 = Requests(W, fm16, 16, 23)
    tr23 = Requests(W, fm23, 0, 23, snapshot_pools(sp23, requests(W, 0, 23)))
    n_clicks16 = sum(len(s) for s in timed_sequences(W, 16 * DAY)[0])
    print(f"Downstream data: impressions before day 16 {tr16.n_all:,} ({tr16.pos_all:,} satisfied); days 16 to 23 "
          f"{va16.n_all:,} ({va16.pos_all:,} satisfied); before day 23 {tr23.n_all:,} ({tr23.pos_all:,} satisfied); "
          f"clicks the foundation model was pre-trained on before day 16: {n_clicks16:,}; requests with a history "
          f"before day 16 {len(tr16.rows):,}, encoded in {len(tr16.rows_x):,} rows ({tr16.own:,} of their own)  "
          f"[{time.time() - t0:.0f} s]", flush=True)

    # ---- ways 2 and 3, each with a control that starts from random weights
    s16, s23 = satisfied_sequences(W, 16 * DAY), satisfied_sequences(W, 23 * DAY)
    ft_tr16 = satisfied_windows(fm16, *s16)
    ft_va16 = satisfied_windows(fm16, *s23, lo=16 * DAY, hi=23 * DAY)
    ft_tr23 = satisfied_windows(fm23, *s23)
    print(f"Satisfied clicks: {sum(int(s.sum()) for s in s16[2]):,} of {sum(len(s) for s in s16[0]):,} clicks before "
          f"day 16; {sum(int(s.sum()) for s in s23[2]):,} of {sum(len(s) for s in s23[0]):,} before day 23")
    models = {}

    def way2(pre):
        m16, c = fit_pairs(PairScorer(copy.deepcopy(fm16.model) if pre else new_seq(fm16, *FM)), tr16, va16, 20,
                           lr_seq=1e-4 if pre else 1e-3)
        e = int(np.argmin(c)) + 1
        m23 = fit_pairs(PairScorer(copy.deepcopy(fm23.model) if pre else new_seq(fm23, *FM)), tr23, None, e,
                        lr_seq=1e-4 if pre else 1e-3)[0]
        return m16, m23, c

    def way3(pre):
        m16, c = fit(copy.deepcopy(fm16.model) if pre else new_seq(fm16, *FM), ft_tr16, ft_va16, 30,
                     lr=3e-4 if pre else 1e-3, patience=2)
        e = int(np.argmin(c)) + 1
        m23 = fit(copy.deepcopy(fm23.model) if pre else new_seq(fm23, *FM), ft_tr23, None, e,
                  lr=3e-4 if pre else 1e-3)[0]
        return m16, m23, c

    for label, fn, what in (("+ FM subgraph", lambda: way2(True), "log loss of satisfied plays on impressions"),
                            ("control: subgraph, no pre-training", lambda: way2(False), "the same"),
                            ("+ FM fine-tuned", lambda: way3(True), "cross-entropy of the next satisfied play"),
                            ("control: fine-tuned, no pre-training", lambda: way3(False), "the same")):
        m16, m23, c = cached("pipe_" + label.replace(" ", "_").replace(":", "").replace(",", ""), fn)
        models[label] = (m16, m23)
        print(f"  {label}: validation {what} " + " ".join(f"{v:.4f}" for v in c)
              + f"; chosen {int(np.argmin(c)) + 1} epochs  [{time.time() - t0:.0f} s]", flush=True)
    models["+ Chapter 9's model"] = (gru16, gru23)
    models["+ FM, request time"] = (fm16.model, fm23.model)
    models["+ FM embeddings, daily"] = (fm16.model, fm23.model)

    def second(label, m, src, H, T, t_req, Hd, Td, td):
        if label == "+ Chapter 9's model":
            return log_softmax_rows(score(m, H), video)
        if label == "+ FM embeddings, daily":
            return with_model(src, m).log_probs(Hd, Td, td)
        if isinstance(m, PairScorer):
            return pair_scores(m, src, H, T, t_req)
        return with_model(src, m).log_probs(H, T, t_req)

    # ---- blends on the validation week; the teacher; then the students, distilled from it
    Rv = requests(W, 16, 23)
    Hv, ivv = histories(W, Rv)
    _, Tv = timed_histories(W, Rv)
    Rvd = day_requests(Rv)
    Hvd, Tvd = timed_histories(W, Rvd)
    rows, cand, y = heldout_rows(W, Rv, Hv, ivv)
    hp = score_pools(sp16["heavy"], sp16["S"], users, items, Rv["user"][rows], cand[:, None])[:, 0]
    wts, ll = {}, {}

    def fit_blend(label, m):
        lp = np.empty(len(rows))
        for a in range(0, len(Rv["user"]), 2_048):
            sel = (rows >= a) & (rows < a + 2_048)
            s = second(label, m, fm16, Hv[a:a + 2_048], Tv[a:a + 2_048], Rv["time"][a:a + 2_048],
                       Hvd[a:a + 2_048], Tvd[a:a + 2_048], Rvd["time"][a:a + 2_048])
            lp[sel] = s[rows[sel] - a, cand[sel]]
        X = features(hp, lp, ivv[rows])
        w = fit_logistic(X, y)
        p = np.clip(1 / (1 + np.exp(-(X @ w))), 1e-12, 1 - 1e-12)
        return w, float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())

    for label in list(models):
        wts[label], ll[label] = fit_blend(label, models[label][0])
    t_label = min(("+ FM subgraph", "+ FM fine-tuned"), key=lambda k: ll[k])
    t_kind = "subgraph" if t_label == "+ FM subgraph" else "fine-tuned"
    print(f"Blend log loss on the validation week's {len(y):,} impressions: "
          + "; ".join(f"{k} {v:.5f}" for k, v in ll.items()) + f"\nTeacher: {t_label}  [{time.time() - t0:.0f} s]",
          flush=True)

    def student(kd):
        tab16 = None if kd == "none" else teacher_table((t_kind, models[t_label][0]), tr16, kd)
        m16, c = fit_pairs(PairScorer(new_seq(fm16, *STUDENT)), tr16, va16, 20, teacher=tab16, kd=kd)
        del tab16
        e = int(np.argmin(c)) + 1
        tab23 = None if kd == "none" else teacher_table((t_kind, models[t_label][1]), tr23, kd)
        m23 = fit_pairs(PairScorer(new_seq(fm23, *STUDENT)), tr23, None, e, teacher=tab23, kd=kd)[0]
        return m16, m23, c

    for label, kd in (("+ student", "none"), ("+ student, distilled, candidates", "candidates"),
                      ("+ student, distilled, all videos", "all")):
        t1 = time.time()
        m16, m23, c = cached("pipe_" + label.replace(" ", "_").replace(",", "") + "_" + t_kind, lambda: student(kd))
        models[label] = (m16, m23)
        wts[label], ll[label] = fit_blend(label, m16)
        print(f"  {label}: validation log loss of satisfied plays " + " ".join(f"{v:.4f}" for v in c)
              + f"; chosen {int(np.argmin(c)) + 1} epochs; blend log loss {ll[label]:.5f}  [{time.time() - t0:.0f} s]",
              flush=True)
    order = ["+ Chapter 9's model", "+ FM, request time", "+ FM embeddings, daily", "+ FM subgraph",
             "+ FM fine-tuned", "control: subgraph, no pre-training", "control: fine-tuned, no pre-training",
             "+ student", "+ student, distilled, candidates", "+ student, distilled, all videos"]
    for label in order:
        w = wts[label]
        print(f"  blend with {label + ':':<40} {w[0]:+.3f} {w[1]:+.3f} x heavy logit {w[2]:+.3f} x second score "
              f"{w[3]:+.3f} x visit has a click {w[4]:+.3f} x second score once the visit has a click")
    cost = {label: serving_flops(label, models[label][1], fm23) for label in order}
    g = torch.Generator().manual_seed(0)
    x = torch.randint(0, len(fm23.video), (64, K), generator=g)
    t = torch.sort(torch.rand(64, K, generator=g) * 30 * DAY, dim=1).values
    tn = torch.cat([t[:, 1:], t[:, -1:] + 60.0], 1)
    for label in ("+ FM fine-tuned", "+ student"):
        seq = models[label][1].seq if isinstance(models[label][1], PairScorer) else models[label][1]
        seq.train()
        with torch.no_grad():
            f = flops(lambda: seq.encode(x, t, tn)) / 64
        seq.eval()
        print(f"  forward FLOPs of {'the foundation model' if label == '+ FM fine-tuned' else 'the student'}'s "
              f"layers over 50 clicks: {f:,.0f}")
    print(f"  teacher scores saved for the students before day 16: {len(tr16.rows):,} requests x {K} candidates = "
          f"{len(tr16.rows) * K:,}, or x {fm16.model.n_videos:,} videos = {len(tr16.rows) * fm16.model.n_videos:,}")

    # ---- the test week
    S = sp23["S"]
    R = requests(W, 23, 30)
    H, iv = histories(W, R)
    _, T = timed_histories(W, R)
    Rd = day_requests(R)
    Hd, Td = timed_histories(W, Rd)
    bad = exclude_mask(W, S, R, H)
    rl = rules(W["cfg"], users, items)
    rs = RuleSets(rl, [np.flatnonzero(S["used"][u]) for u in range(len(users["vec"]))], n, "numpy")
    uu, inv = np.unique(R["user"], return_inverse=True)
    kept = eligible_kept(ranked_pools(sp23, W, rs, uu), inv, bad)
    new = first_allowed(new_candidates(W, S, uu, rl)[inv], bad)
    ads = ad_setup(W, W["cfg"], rl)
    ad = best_ads(W, S, uu, ads)[0][:, 0][inv]
    arms = {"A: Chapter 8's row": []} | {label: [] for label in order}
    for a in range(0, len(R["user"]), 2_048):
        b = min(a + 2_048, len(R["user"]))
        k50 = kept[a:b]
        hpp = score_pools(sp23["heavy"], S, users, items, R["user"][a:b], np.maximum(k50, 0))
        hpp = np.where(k50 >= 0, hpp, np.nan)
        rel = np.where(k50 >= 0, np.nan_to_num(hpp, nan=-np.inf), -np.inf)
        arms["A: Chapter 8's row"].append(assemble_rows(k50, rel, new[a:b], ad[a:b], items))
        for label in order:
            sec = second(label, models[label][1], fm23, H[a:b], T[a:b], R["time"][a:b], Hd[a:b], Td[a:b],
                         Rd["time"][a:b])
            s2 = np.where(k50 >= 0, np.take_along_axis(sec, np.maximum(k50, 0), axis=1), 0.0)
            sc = features(np.nan_to_num(hpp, nan=0.5), s2, iv[a:b, None]) @ wts[label]
            rel = np.where(k50 >= 0, np.nan_to_num(sc, nan=-np.inf), -np.inf)
            arms[label].append(assemble_rows(k50, rel, new[a:b], ad[a:b], items))
    later = R["place"] >= 1
    tc = top_creators(W, S)
    print(f"Test week: {len(R['user']):,} requests ({later.mean():.1%} later in their visit)  [{time.time() - t0:.0f} s]")
    print(f"  {'':<38}{'NDCG@10':>9}{'coverage':>10}{'top-10% cr.':>12}{'online':>9}{'against A':>18}"
          f"{'95% interval':>20}{'revenue':>9}{'rule breaks':>13}{'not allowed':>13}{'FLOPs/request':>15}")
    base, res = None, {}
    for label, parts in arms.items():
        top = np.concatenate(parts)
        click, play = outcomes(W, R, top)
        on = play.sum(axis=1)
        base = on if base is None else base
        vids = np.where((top >= 0) & ~items["is_product"][np.maximum(top, 0)], top, -1)
        cov, cr = guardrails(vids, S["ok"], items["creator"], tc, n)
        i = np.maximum(top, 0)
        rev = np.where((top >= 0) & items["is_product"][i], click * ads["bid"][i], 0.0).sum(axis=1)
        breaks = check_rows(top, items, CREATOR_CAP, [(new, NEW_SLOT), (ad, AD_SLOT)])[0]
        shown_bad = forbidden_shown(top, bad, items)[0]
        if on is base:
            d = ci = ""
        else:
            m_, se, lo, hi = paired(on, base, R["user"])
            d, ci = f"{m_:+.4f} ({se:.4f})", f"[{lo:+.4f}, {hi:+.4f}]"
        print(f"  {label:<38}{ndcg(R, top):>9.4f}{cov:>10.1%}{cr:>12.1%}{on.mean():>9.4f}{d:>18}{ci:>20}"
              f"{rev.mean():>9.4f}{breaks:>13}{shown_bad:>13}{cost.get(label, 0):>15,.0f}")
        res[label] = on
    print("Direct comparisons (paired over the same requests, users resampled):")
    for a_, b_ in (("+ FM subgraph", "+ Chapter 9's model"), ("+ FM fine-tuned", "+ Chapter 9's model"),
                   ("+ FM subgraph", "control: subgraph, no pre-training"),
                   ("+ FM fine-tuned", "control: fine-tuned, no pre-training"),
                   ("+ FM embeddings, daily", "+ FM, request time"),
                   ("+ student, distilled, candidates", "+ student"),
                   ("+ student, distilled, all videos", "+ student"),
                   ("+ student, distilled, candidates", "+ student, distilled, all videos"),
                   (t_label, "+ student, distilled, candidates"), (t_label, "+ student, distilled, all videos")):
        print(f"  {a_} minus {b_}: " + "{:+.4f} ({:.4f}) [{:+.4f}, {:+.4f}]".format(*paired(res[a_], res[b_],
                                                                                            R["user"])))
    lang = users["language"][R["user"]]
    print("Online by app language (arms in the order above):")
    for g in range(len(LANGUAGES)):
        print(f"  {LANGUAGES[g]} ({(lang == g).sum():,}): " + " ".join(f"{res[k][lang == g].mean():.4f}" for k in res))
    print(f"(online: expected satisfied plays per row; against A: paired difference, standard error and 95% interval "
          f"from 1,000 bootstrap draws of whole users; coverage and top-10% cr.: Chapter 5's guardrails; FLOPs/request: "
          f"the second score's forward pass for one request with 50 earlier clicks at serving)  [{time.time() - t0:.0f} s]")
