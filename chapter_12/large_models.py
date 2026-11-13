"""Generative sequence models for Chapter 12: HSTU and SASRec at any size (sections 12.4 to 12.6).

Both read a user's clicked videos in order and predict the next click over every video
(a full softmax), the generative training of Chapter 9: every click with an earlier
click before its request is a training example. A video's vector (d numbers) is used at
the input and the output, as in Chapter 9, and a learned vector per position is added
at the input.

  HSTU     - Meta's Hierarchical Sequential Transduction Unit (Zhai and colleagues,
             2024; Listing 12.2): one projection with a SiLU gives U, V, Q and K;
             attention weights are SiLU(Q K^T + relative bias) divided by the length,
             not a softmax; the bias depends on the distance in positions and on the
             time between each earlier click and the moment of the prediction; the
             attended values are layer-normed, multiplied by U and projected back. One
             layer has 5 d^2 weights.
  SASRec   - a standard causal Transformer (Kang and McAuley, 2018), pre-norm, softmax
             attention and a feed-forward block 4 times as wide: 12 d^2 weights a layer.

  windows(seqs, times, moments, k, lo, hi) - sequences cut into windows of up to k
                                  clicks; every click is predicted at the moment of its
                                  request, from the 1 to k clicks of its window made
                                  before then; only targets whose time is in [lo, hi)
                                  are scored
  train_flops(model, k)         - training compute per predicted click (Listing 12.3)
  serving_cost(model, k, n)     - forward FLOPs per request, and the time of a batch of
                                  n requests (section 12.7)
  fit(model, train, valid, ...) - Adam on full-softmax cross-entropy; the epoch with the
                                  lowest validation loss is kept
  loss(model, data)             - mean cross-entropy (nats) of the next click
  distill_loss(...)             - a student's loss with a teacher (section 12.6)
"""
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.flop_counter import FlopCounterMode

torch.set_num_threads(2)
BUCKETS = 24                                   # time gaps: bucket = floor(log2(1 + seconds)), at most 24


class HSTULayer(nn.Module):
    """Listing 12.2. One HSTU layer (Zhai and colleagues, 2024, equations 1 to 3)."""
    def __init__(self, d, heads, max_len, dropout=0.1):
        super().__init__()
        self.d, self.h = d, heads
        self.uvqk = nn.Linear(d, 4 * d, bias=False)          # equation 1: U, V, Q, K in one projection
        self.out = nn.Linear(d, d)                            # equation 3: f2
        self.pos_bias = nn.Parameter(torch.zeros(max_len))    # by distance i - j in positions
        self.time_bias = nn.Parameter(torch.zeros(BUCKETS + 1))   # by time between the clicks
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask, gaps):
        B, n, d = x.shape
        u, v, q, k = F.silu(self.uvqk(F.layer_norm(x, (d,)))).split(d, dim=-1)
        split = lambda t: t.view(B, n, self.h, d // self.h).transpose(1, 2)           # batch x heads x n x d/h
        dist = torch.arange(n)[:, None] - torch.arange(n)[None, :]
        bias = self.pos_bias[dist.clamp(0, len(self.pos_bias) - 1)] + self.time_bias[gaps]
        w = F.silu(split(q) @ split(k).transpose(-1, -2) + bias[:, None]) / n          # equation 2, no softmax
        a = (w * mask[:, None]) @ split(v)
        a = a.transpose(1, 2).reshape(B, n, d)
        return x + self.out(self.drop(F.layer_norm(a, (d,)) * u))                      # equation 3


class SASRecLayer(nn.Module):
    """A pre-norm Transformer layer with causal softmax attention."""
    def __init__(self, d, heads, max_len, dropout=0.1):
        super().__init__()
        self.att = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.n1, self.n2, self.drop = nn.LayerNorm(d), nn.LayerNorm(d), nn.Dropout(dropout)

    def forward(self, x, mask, gaps):
        h = self.n1(x)
        blocked = (~mask).repeat_interleave(self.att.num_heads, 0)          # one mask per head
        a = self.att(h, h, h, attn_mask=blocked, need_weights=False)[0]
        x = x + self.drop(a)
        return x + self.drop(self.ff(self.n2(x)))


class SeqLM(nn.Module):
    """Videos in, the next video out, for HSTU or SASRec layers of width d. A video's
    vector is learned for its id; with parts (each video's creator, category and
    language), learned vectors for those are added, as in Chapter 9's models."""
    def __init__(self, n_videos, kind="HSTU", d=64, layers=2, heads=2, max_len=64, seed=0, parts=None):
        super().__init__()
        torch.manual_seed(seed)
        self.kind, self.d, self.n_layers, self.n_videos, self.max_len = kind, d, layers, n_videos, max_len
        self.pad = n_videos
        self.item = nn.Embedding(n_videos + 1, d, padding_idx=n_videos)
        nn.init.normal_(self.item.weight, std=d ** -0.5)
        self.pos = nn.Embedding(max_len, d)
        nn.init.normal_(self.pos.weight, std=0.02)
        Layer = HSTULayer if kind == "HSTU" else SASRecLayer
        self.layers = nn.ModuleList([Layer(d, heads, max_len) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.parts = None
        if parts is not None:
            self.parts = nn.ModuleDict({k: nn.Embedding(int(v.max()) + 2, d) for k, v in parts.items()})
            for k, v in parts.items():
                nn.init.normal_(self.parts[k].weight, std=d ** -0.5)
                self.register_buffer("part_" + k, torch.as_tensor(np.r_[v, int(v.max()) + 1]))

    def non_embedding_params(self):
        return sum(p.numel() for name, p in self.named_parameters()
                   if not name.startswith(("item", "pos", "parts")))

    def vectors(self, ids):
        v = self.item(ids)
        if self.parts is not None:
            v = v + sum(e(getattr(self, "part_" + k)[ids]) for k, e in self.parts.items())
        return v

    def encode(self, x, t, t_next):
        """x: videos (batch x n, right-padded); t: their times; t_next: at each position,
        the moment the next click is predicted at: the time of the request that led to
        it, which a server knows when the request arrives. Times in seconds. One output
        per position."""
        n = x.shape[1]
        real = x != self.pad
        mask = torch.tril(torch.ones(n, n, dtype=torch.bool))[None] & real[:, None, :] & real[:, :, None]
        mask = mask | torch.eye(n, dtype=torch.bool)[None]                    # padding sees itself only
        gaps = torch.log2(1 + (t_next[:, :, None] - t[:, None, :]).abs()).long().clamp(0, BUCKETS)
        h = self.vectors(x) * math.sqrt(self.d) + self.pos(torch.arange(n))[None]
        for layer in self.layers:
            h = layer(h, mask, gaps)
        return self.norm(h)

    def logits(self, h):
        return h @ self.vectors(torch.arange(self.n_videos)).T


def train_flops(model, k):
    """Listing 12.3. Training compute per predicted click, in floating-point operations:
    6 per weight touched (2 forward, 4 backward), for the layers and for the output
    softmax over every video, plus the attention over up to k earlier clicks."""
    d, L = model.d, model.n_layers
    weights = model.non_embedding_params() + d * model.n_videos      # layers + the output vectors
    attention = 2 * 2 * L * k * d                                    # Q K^T and the weighted sum, forward
    return 6 * weights + 3 * attention


def request_batch(n_videos, n=64, k=64, seed=0):
    """A made-up batch of n requests, each with k earlier clicks and their times."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, n_videos, (n, k), generator=g)
    t = torch.sort(torch.rand(n, k, generator=g) * 30 * 86_400, dim=1).values
    tn = torch.cat([t[:, 1:], t[:, -1:] + 60.0], 1)
    return x, t, tn


@torch.no_grad()
def serve(model, x, t, tn):
    """The scores of every video for each request: the last position's output."""
    return model.logits(model.encode(x, t, tn)[:, -1])


def serving_cost(model, k=64, n=64):
    """What serving one request costs: the forward FLOPs per request, and the time of a
    batch of n requests (the median of 5 timings) with that time divided by n. The time
    per request is a share of a batch's time, not the latency a request sees: each
    request waits for its whole batch, and for the queue before it. The FLOPs are
    counted in training mode, where dropout adds none: in evaluation mode PyTorch runs
    SASRec's attention as one fused operation that FlopCounterMode does not count. The
    timings are made in evaluation mode, as a server would run."""
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
    batch_ms = 1_000 * float(np.median(times))
    return fc.get_total_flops() / n, batch_ms / n, batch_ms


def windows(seqs, times, moments, k, lo=-math.inf, hi=math.inf):
    """Every sequence cut into windows of up to k clicks. Each click is predicted at a
    moment: the time of the request that led to it, which a server knows when the
    request arrives. It is predicted from the clicks made before that moment only, so
    its prediction is read at the last of them, and clicks from the same row share that
    history. With moments None, as in Chapter 9's logs, which keep no request times,
    each click is predicted at its own time from the clicks before it in the user's
    order. A click with no earlier click is not predicted. A window holds the clicks
    whose prediction is read in it, up to the last of those clicks; a target is scored
    only if its click time is in [lo, hi). Returns, per window, (videos, times, the
    moment each position predicts at, where each target is read, targets, scored),
    keeping only windows with a scored target."""
    out = [], [], [], [], [], []

    def keep(x, tt, mm, at, y, ok):
        if ok.any():
            for part, v in zip(out, (x, tt, mm, at, y, ok)):
                part.append(v)

    for u, (s, t) in enumerate(zip(seqs, times)):
        if moments is None:                                             # each click from the ones before it
            m, read = t, np.arange(len(s)) - 1
        else:
            m = moments[u]
            read = np.searchsorted(t, m, side="left") - 1              # the last click before each request
        ahead = np.r_[m[1:], t[-1:]]                                    # by default, the next click's moment
        ok = (t >= lo) & (t < hi)
        for a in range(0, len(s), k):
            b = min(a + k, len(s))
            j = np.flatnonzero((read >= a) & (read < b))               # the targets read in this window
            mm = ahead[a:b].copy()
            extra = []
            for p in np.unique(read[j]):
                at = j[read[j] == p]
                first = m[at].min()
                mm[p - a] = first                                       # a position predicts at its request
                for later in np.unique(m[at][m[at] > first]):           # another request with no click between:
                    i = at[m[at] == later]                              # its own copy of the history
                    mx = mm[:p - a + 1].copy()
                    mx[-1] = later
                    extra.append((s[a:p + 1], t[a:p + 1], mx, read[i] - a, s[i], ok[i]))
                j = np.setdiff1d(j, at[m[at] > first])
            if len(j):
                e = read[j].max() + 1                                   # up to the last click read
                keep(s[a:e], t[a:e], mm[:e - a], read[j] - a, s[j], ok[j])
            for e in extra:
                keep(*e)
    return out


def tensors(xs, ts, ms, at, ys, sc, pad):
    """Right-padded inputs, their times and moments, where each scored target is read
    (window, position) and the targets."""
    n = max(len(x) for x in xs)
    X = np.full((len(xs), n), pad)
    T = np.zeros((len(xs), n))
    M = np.zeros((len(xs), n))
    rows, pos, Y = [], [], []
    for r, (x, t, m, p, y, ok) in enumerate(zip(xs, ts, ms, at, ys, sc)):
        X[r, :len(x)] = x
        T[r, :len(t)], T[r, len(t):] = t, t[-1]
        M[r, :len(m)], M[r, len(m):] = m, m[-1]
        rows.append(np.full(int(ok.sum()), r))
        pos.append(p[ok])
        Y.append(y[ok])
    where = (torch.as_tensor(np.concatenate(rows)), torch.as_tensor(np.concatenate(pos)))
    return (torch.as_tensor(X), torch.as_tensor(T, dtype=torch.float32), torch.as_tensor(M, dtype=torch.float32),
            where, torch.as_tensor(np.concatenate(Y)))


def scored(data):
    """The number of scored targets in windows(...)."""
    return int(sum(ok.sum() for ok in data[5]))


def batches(data, pad, batch, gen=None):
    """(inputs, input times, the moment each position predicts at, where the targets are
    read, targets) in batches of windows of similar length; shuffled when a random
    generator is given."""
    length = torch.as_tensor([len(x) for x in data[0]], dtype=torch.float32)
    order = torch.argsort(length + (torch.rand(len(length), generator=gen) if gen is not None else 0))
    groups = list(torch.split(order, batch))
    if gen is not None:
        groups = [groups[g] for g in torch.randperm(len(groups), generator=gen).tolist()]
    for g in groups:
        g = g.tolist()
        yield tensors(*([part[i] for i in g] for part in data), pad)


def loss(model, data, batch=256):
    """Mean cross-entropy (nats) of every scored target in data."""
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for x, t, tn, where, y in batches(data, model.pad, batch):
            h = model.encode(x, t, tn)[where]
            total += F.cross_entropy(model.logits(h), y, reduction="sum").item()
            count += len(y)
    return total / max(count, 1)


def distill_loss(student, teacher, x, t, tn, where, y, weight=0.5, temp=2.0):
    """Section 12.6. The student's loss: (1 - weight) x its cross-entropy on the real next
    click, plus weight x temp^2 x the KL divergence from the teacher's softened
    distribution over every video to its own."""
    s = student.logits(student.encode(x, t, tn)[where])
    with torch.no_grad():
        p = F.softmax(teacher.logits(teacher.encode(x, t, tn)[where]) / temp, dim=-1)
    kl = F.kl_div(F.log_softmax(s / temp, dim=-1), p, reduction="batchmean")
    return (1 - weight) * F.cross_entropy(s, y) + weight * temp ** 2 * kl


def fit(model, train, valid, epochs, lr=1e-3, batch=128, seed=0, teacher=None, log=None, patience=None):
    """Adam on the full-softmax cross-entropy of every next click (or distill_loss with
    a teacher). After each epoch the validation loss is measured and the weights of the
    best epoch are kept; with patience p, training stops once p epochs in a row have not
    beaten the best. With valid None, the model after the last epoch is returned.
    Returns the model and the validation losses."""
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    curve, best, state = [], math.inf, None
    for e in range(1, epochs + 1):
        model.train()
        for x, t, tn, where, y in batches(train, model.pad, batch, gen):
            if teacher is None:
                L = F.cross_entropy(model.logits(model.encode(x, t, tn)[where]), y)
            else:
                L = distill_loss(model, teacher, x, t, tn, where, y)
            opt.zero_grad()
            L.backward()
            opt.step()
        if valid is None:
            continue
        v = loss(model, valid)
        curve.append(v)
        if log is not None:
            log(e, v)
        if v < best:
            best, state = v, {k: p.detach().clone() for k, p in model.state_dict().items()}
        elif patience is not None and e - (int(np.argmin(curve)) + 1) >= patience:
            break
    if state is not None:
        model.load_state_dict(state)
    model.eval()
    return model, curve
