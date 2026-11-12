"""Generative sequence models for Chapter 12: HSTU and SASRec at any size (sections 12.4 to 12.6).

Both read a user's clicked videos in order and predict the next click over every video
(a full softmax), the generative training of Chapter 9: every position of every
sequence is a training example. A video's vector (d numbers) is used at the input and
the output, as in Chapter 9, and a learned vector per position is added at the input.

  HSTU     - Meta's Hierarchical Sequential Transduction Unit (Zhai and colleagues,
             2024; Listing 12.2): one projection with a SiLU gives U, V, Q and K;
             attention weights are SiLU(Q K^T + relative bias) divided by the length,
             not a softmax; the bias depends on the distance in positions and on the
             time between each earlier click and the moment of the prediction; the
             attended values are layer-normed, multiplied by U and projected back. One
             layer has 5 d^2 weights.
  SASRec   - a standard causal Transformer (Kang and McAuley, 2018), pre-norm, softmax
             attention and a feed-forward block 4 times as wide: 12 d^2 weights a layer.

  windows(seqs, times, k, lo, hi) - sequences cut into windows of k + 1 clicks, so every
                                  click is predicted from at most k earlier ones; only
                                  targets whose time is in [lo, hi) are scored
  train_flops(model, k)         - training compute per predicted click (Listing 12.3)
  fit(model, train, valid, ...) - Adam on full-softmax cross-entropy; the epoch with the
                                  lowest validation loss is kept
  loss(model, data)             - mean cross-entropy (nats) of the next click
  distill_loss(...)             - a student's loss with a teacher (section 12.6)
"""
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

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
        the moment of the click to be predicted (the next click in training; the request
        at serving). Times in seconds. One output per position."""
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


def windows(seqs, times, k, lo=-math.inf, hi=math.inf):
    """Every sequence cut into consecutive windows of k + 1 clicks: the first k are inputs
    and each of the last k is predicted from the clicks before it in the window. A target
    is scored only if its time is in [lo, hi). Returns (videos, times, scored) per window,
    keeping only windows with a scored target."""
    xs, ts, sc = [], [], []
    for s, t in zip(seqs, times):
        for a in range(0, max(len(s) - 1, 1), k):
            x, tt = s[a:a + k + 1], t[a:a + k + 1]
            ok = (tt[1:] >= lo) & (tt[1:] < hi)
            if len(x) >= 2 and ok.any():
                xs.append(x)
                ts.append(tt)
                sc.append(ok)
    return xs, ts, sc


def tensors(xs, ts, sc, pad):
    n = max(len(x) for x in xs)
    X = np.full((len(xs), n), pad)
    T = np.zeros((len(xs), n))
    Y = np.full((len(xs), n - 1), pad)
    for r, (x, t, ok) in enumerate(zip(xs, ts, sc)):
        X[r, :len(x)] = x
        T[r, :len(t)] = t
        T[r, len(t):] = t[-1]
        Y[r, :len(x) - 1] = np.where(ok, x[1:], pad)          # targets not scored are padding
    T = torch.as_tensor(T, dtype=torch.float32)
    return torch.as_tensor(X[:, :-1]), T[:, :-1], T[:, 1:], torch.as_tensor(Y)


def batches(data, pad, batch, gen=None):
    """(inputs, input times, target times, targets) in batches of windows of similar
    length; shuffled when a random generator is given."""
    xs, ts, sc = data
    length = torch.as_tensor([len(x) for x in xs], dtype=torch.float32)
    order = torch.argsort(length + (torch.rand(len(xs), generator=gen) if gen is not None else 0))
    groups = list(torch.split(order, batch))
    if gen is not None:
        groups = [groups[g] for g in torch.randperm(len(groups), generator=gen).tolist()]
    for g in groups:
        g = g.tolist()
        yield tensors([xs[i] for i in g], [ts[i] for i in g], [sc[i] for i in g], pad)


def loss(model, data, batch=256):
    """Mean cross-entropy (nats) of every scored next click in data."""
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for x, t, tn, y in batches(data, model.pad, batch):
            h = model.encode(x, t, tn)
            keep = y != model.pad
            total += F.cross_entropy(model.logits(h[keep]), y[keep], reduction="sum").item()
            count += int(keep.sum())
    return total / max(count, 1)


def distill_loss(student, teacher, x, t, tn, y, weight=0.5, temp=2.0):
    """Section 12.6. The student's loss: (1 - weight) x its cross-entropy on the real next
    click, plus weight x temp^2 x the KL divergence from the teacher's softened
    distribution over every video to its own."""
    keep = y != student.pad
    s = student.logits(student.encode(x, t, tn)[keep])
    with torch.no_grad():
        p = F.softmax(teacher.logits(teacher.encode(x, t, tn)[keep]) / temp, dim=-1)
    kl = F.kl_div(F.log_softmax(s / temp, dim=-1), p, reduction="batchmean")
    return (1 - weight) * F.cross_entropy(s, y[keep]) + weight * temp ** 2 * kl


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
        for x, t, tn, y in batches(train, model.pad, batch, gen):
            if teacher is None:
                keep = y != model.pad
                L = F.cross_entropy(model.logits(model.encode(x, t, tn)[keep]), y[keep])
            else:
                L = distill_loss(model, teacher, x, t, tn, y)
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
