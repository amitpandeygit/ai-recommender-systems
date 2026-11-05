"""Listings 9.3 and 9.4 - GRU4Rec, SASRec and BERT4Rec on Tesselo's clicks (section 9.4).

All three read a user's last 50 clicked videos and score every video as the next
click. A video's vector (64 numbers) is the sum of learned vectors for its id,
creator, category and language, as in Chapter 8's cross-encoder, so a rarely
clicked video still shares what its creator and category have learned. The same
vectors are used at the input and the output: the score of a video is the dot
product of the sequence's output vector with that video's vector.

  GRU4Rec   - a recurrent network (Hidasi and colleagues, 2016), trained here with
              the full softmax over all videos at every step
  SASRec    - a Transformer encoder with causal attention: each position sees only
              the clicks before it (Kang and McAuley, 2018). Three losses:
                BCE, 1 negative         - the original: the next click against one
                                          random video, as two yes/no questions
                sampled softmax, 100    - the next click against 100 random videos
                full softmax            - the next click against every video
  BERT4Rec  - the same encoder without the causal mask, trained to fill in masked
              clicks (20% of positions) from both sides; to predict, a mask token is
              put after the last click (Sun and colleagues, 2019)

Sequences are right-padded: real clicks first, then padding, so every position that
is used has at least one real click to attend to.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

torch.set_num_threads(2)
D, LAYERS, HEADS, DROPOUT = 64, 2, 2, 0.2
MAX_LEN = 50
LOSSES = ("BCE, 1 negative", "sampled softmax, 100", "full softmax")


def right_pad(H, pad):
    """Left-padded histories (-1) -> right-padded ids and lengths."""
    n = (H >= 0).sum(axis=1)
    out = np.full(H.shape, pad)
    for k, m in enumerate(n):
        if m:
            out[k, :m] = H[k, -m:]
    return torch.as_tensor(out), torch.as_tensor(n)


class SeqModel(nn.Module):
    def __init__(self, n_items, kind, items, seed=0, max_len=MAX_LEN):
        super().__init__()
        torch.manual_seed(seed)
        self.n, self.kind, self.pad, self.mask_id = n_items, kind, n_items, n_items + 1
        self.item = nn.Embedding(n_items + 2, D, padding_idx=n_items)
        nn.init.normal_(self.item.weight, std=D ** -0.5)
        self.parts = nn.ModuleDict({k: nn.Embedding(int(items[k].max()) + 2, D)
                                    for k in ("creator", "category", "language")})
        for k, e in self.parts.items():
            nn.init.normal_(e.weight, std=D ** -0.5)
            self.register_buffer(k, torch.as_tensor(np.r_[items[k], int(items[k].max()) + 1, int(items[k].max()) + 1]))
        video = ~items["is_product"]
        self.drop = nn.Dropout(DROPOUT)
        if kind == "GRU4Rec":
            self.gru = nn.GRU(D, D, batch_first=True)
        else:
            self.pos = nn.Embedding(max_len + 1, D)
            layer = nn.TransformerEncoderLayer(D, HEADS, 2 * D, DROPOUT, batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, LAYERS, enable_nested_tensor=False)
            self.norm = nn.LayerNorm(D)
        self.register_buffer("video", torch.as_tensor(video))

    def vectors(self, ids):
        """A video's vector: learned vectors for its id, creator, category and language,
        added (the padding and mask tokens get their own)."""
        return self.item(ids) + sum(e(getattr(self, k)[ids]) for k, e in self.parts.items())

    def encode(self, x):
        """Listing 9.3 (first part). One output vector per position of x (batch x length)."""
        h = self.drop(self.vectors(x))
        if self.kind == "GRU4Rec":
            return self.gru(h)[0]
        L = x.shape[1]
        h = h * math.sqrt(D) + self.pos(torch.arange(L))[None]
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool), 1) if self.kind == "SASRec" else None
        out = self.enc(h, mask=causal, src_key_padding_mask=x == self.pad)
        return self.norm(out)

    def logits(self, h):
        """Scores of every video for each output vector (products can never be next)."""
        s = h @ self.vectors(torch.arange(self.n)).T
        return s.masked_fill(~self.video, -1e9)


def sasrec_loss(model, x, y, loss):
    """Listing 9.3 (second part). x: clicks, y: the click that came next at each
    position (pad where there is none). Three ways to turn scores into a loss."""
    h = model.encode(x)
    real = y != model.pad
    h, y = h[real], y[real]                                      # one row per real (position, next click)
    if loss == "full softmax":
        return F.cross_entropy(model.logits(h), y)
    vids = torch.nonzero(model.video).squeeze(1)
    k = 1 if loss == "BCE, 1 negative" else 100
    neg = vids[torch.randint(len(vids), (len(y), k))]            # random videos as negatives
    s_pos = (h * model.vectors(y)).sum(-1)
    s_neg = torch.einsum("nd,nkd->nk", h, model.vectors(neg)).masked_fill(neg == y[:, None], -1e9)
    if loss == "BCE, 1 negative":
        return (F.softplus(-s_pos) + F.softplus(s_neg[:, 0])).mean()   # -log s(pos) - log(1 - s(neg))
    return F.cross_entropy(torch.cat([s_pos[:, None], s_neg], 1), torch.zeros(len(y), dtype=torch.long))


def bert4rec_loss(model, x, p_mask=0.2):
    """Listing 9.4. Hide 20% of the clicks (at least one per sequence) behind the mask
    token and predict each hidden click from both sides, over every video."""
    real = x != model.pad
    hide = (torch.rand(x.shape) < p_mask) & real
    last = real.sum(1) - 1                                       # make sure each sequence hides one
    hide[torch.arange(len(x)), last] |= ~hide.any(1)
    h = model.encode(x.masked_fill(hide, model.mask_id))
    return F.cross_entropy(model.logits(h[hide]), x[hide])


def batches(seqs, pad, max_len=MAX_LEN, kind="SASRec"):
    """Training tensors: for causal models, inputs and next clicks; for BERT4Rec, the
    last max_len clicks."""
    n = max_len + 1 if kind != "BERT4Rec" else max_len
    X = np.full((len(seqs), n), pad)
    for k, s in enumerate(seqs):
        s = s[-n:]
        X[k, :len(s)] = s
    X = torch.as_tensor(X)
    if kind == "BERT4Rec":
        return X, None
    return X[:, :-1], X[:, 1:]


def fit(model, seqs, epochs, loss="full softmax", lr=1e-3, batch=128, seed=0, after=None, every=10):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    X, Y = batches(seqs, model.pad, kind=model.kind)
    length = (X != model.pad).sum(1)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for e in range(1, epochs + 1):
        model.train()
        noise = torch.rand(len(X), generator=gen)                 # sequences of similar length together,
        order = torch.argsort(length + noise)                    # so batches carry little padding
        groups = list(torch.split(order, batch))
        for g in torch.randperm(len(groups), generator=gen).tolist():
            b = groups[g]
            L_b = int(length[b].max())
            if model.kind == "BERT4Rec":
                L = bert4rec_loss(model, X[b, :L_b])
            else:
                L = sasrec_loss(model, X[b, :L_b], Y[b, :L_b], loss if model.kind == "SASRec" else "full softmax")
            opt.zero_grad()
            L.backward()
            opt.step()
        if after is not None and e % every == 0:
            model.eval()
            after(e, model)
    model.eval()
    return model


@torch.no_grad()
def score(model, H):
    """Scores of every video as the next click, for each history in H (left-padded)."""
    if model.kind == "BERT4Rec":
        H = H[:, -(MAX_LEN - 1):]                                # room for the mask token
    x, n = right_pad(H, model.pad)
    out = np.zeros((len(H), model.n), dtype=np.float32)
    for a in range(0, len(x), 1_024):
        xb, nb = x[a:a + 1_024], n[a:a + 1_024]
        xb = xb[:, :max(int(nb.max()), 1)]                       # drop columns that are padding everywhere
        if model.kind == "BERT4Rec":                             # a mask token after the last click
            xb = torch.cat([xb, torch.full((len(xb), 1), model.pad)], 1)
            xb[torch.arange(len(xb)), nb] = model.mask_id
            h = model.encode(xb)
            at = nb
        else:
            h = model.encode(xb)
            at = torch.clamp(nb - 1, min=0)
        s = model.logits(h[torch.arange(len(xb)), at])
        out[a:a + 1_024] = np.where((nb > 0)[:, None].numpy(), s.numpy(), 0.0)
    return out
