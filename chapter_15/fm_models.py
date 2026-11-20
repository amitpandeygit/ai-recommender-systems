"""Shared code for Chapter 15: a recommendation foundation model and the ways to transfer
what it knows (sections 15.3 and 15.5 to 15.7).

The foundation model is Chapter 12's HSTU (chapter_12/large_models.py), pre-trained on
every consenting user's clicks to predict the next click. This module adds what the
chapter needs on top of it:

  day_moments(r, every)          - the moment an embedding computed in batch was made: the
                                   start of the period (a day, by default) that contains
                                   each request, so a click is predicted from what was
                                   known then
  future_data(seqs, times, moments, k, J, half_life, lo, hi)
                                 - Chapter 12's windows, plus, for every target, the next
                                   J - 1 clicks of the same user and a weight for each
                                   that halves every half_life seconds after the moment
                                   of the prediction
  multi_token_loss(logits, future, weights)
                                 - Listing 15.1: the weighted cross-entropy of the next J
                                   clicks from one output vector
  fit_future(model, train, valid, epochs, ...)
                                 - Adam on multi_token_loss, with Chapter 12's validation
                                   loop (the epoch with the lowest validation loss kept)
  GatedItemLM                    - Listing 15.2: Chapter 13's item model with a video's
                                   vector made of a content map plus its ID vector, mixed
                                   by a learned gate on how many clicks the ID vector was
                                   trained on, with the ID vector masked at random in
                                   training and replaced by one learned vector for
                                   unknown videos
  stabilise(T, W, T0)            - Listing 15.7: Netflix's orthogonal low-rank
                                   stabilisation (Zielnicki and Hsiao, 2025): rotate a
                                   retrained model's user and item vectors onto a
                                   reference run's, without changing any dot product
"""
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

HERE = Path(__file__).resolve().parent
for _ch in ("chapter_13", "chapter_12"):
    sys.path.insert(0, str(HERE.parent / _ch))
from large_models import loss, tensors, windows  # noqa: E402
from id_models import ItemLM  # noqa: E402

torch.set_num_threads(2)
DAY = 86_400.0


def day_moments(r, every=DAY):
    """The start of the period of length `every` seconds that contains each moment r."""
    return [np.floor(np.asarray(x, dtype=np.float64) / every) * every for x in r]


def future_data(seqs, times, moments, k, J, half_life, lo=-math.inf, hi=math.inf):
    """Chapter 12's windows (each click predicted at its moment, from the clicks made
    before it), and for every target its next J - 1 clicks in the same user's sequence,
    if their time is in [lo, hi). Each of the J targets gets the weight
    0.5 ** ((its time - the moment of the prediction) / half_life); the weights of one
    target sum to 1. Returns the windows' six parts, then the targets (n x J, -1 where
    there is none) and their weights, one array of each per window."""
    starts = np.r_[0, np.cumsum([len(s) for s in seqs])]
    G, T = np.concatenate(seqs), np.concatenate(times)
    end = np.repeat(starts[1:], np.diff(starts))                          # where each click's user ends
    ok_time = (T >= lo) & (T < hi)
    idx = [np.arange(a, b) for a, b in zip(starts[:-1], starts[1:])]      # global positions, not videos
    xs, ts, ms, at, ys, sc = windows(idx, times, moments, k, lo, hi)
    fut, wts = [], []
    for x, m, p, y in zip(xs, ms, at, ys):
        j = y[:, None] + np.arange(J)[None, :]                            # the target and the clicks after it
        real = (j < end[y][:, None]) & ok_time[np.minimum(j, len(G) - 1)]
        real[:, 0] = True
        jj = np.minimum(j, len(G) - 1)
        w = np.where(real, 0.5 ** (np.maximum(T[jj] - m[p][:, None], 0.0) / half_life), 0.0)
        fut.append(np.where(real, G[jj], -1))
        wts.append(w / w.sum(1, keepdims=True))
    xs = [G[x] for x in xs]
    ys = [G[y] for y in ys]
    return xs, ts, ms, at, ys, sc, fut, wts


def future_batches(data, pad, batch, gen=None):
    """Chapter 12's batches of windows of similar length, with each scored target's
    future targets and weights in the same order."""
    length = torch.as_tensor([len(x) for x in data[0]], dtype=torch.float32)
    order = torch.argsort(length + (torch.rand(len(length), generator=gen) if gen is not None else 0))
    groups = list(torch.split(order, batch))
    if gen is not None:
        groups = [groups[g] for g in torch.randperm(len(groups), generator=gen).tolist()]
    for g in groups:
        g = g.tolist()
        x, t, tn, where, y = tensors(*([data[part][i] for i in g] for part in range(6)), pad)
        fut = np.concatenate([data[6][i][data[5][i]] for i in g])
        wts = np.concatenate([data[7][i][data[5][i]] for i in g])
        yield x, t, tn, where, y, torch.as_tensor(fut), torch.as_tensor(wts, dtype=torch.float32)


def multi_token_loss(logits, future, weights):
    """Listing 15.1. One output vector predicts the next J clicks: the cross-entropy of
    each over every video, weighted (each row's weights sum to 1). With one target of
    weight 1 it is the ordinary next-click loss."""
    logp = F.log_softmax(logits, dim=-1)
    nll = -logp.gather(1, future.clamp_min(0))                           # n x J
    return (weights * nll).sum(1).mean()


def fit_future(model, train, valid, epochs, lr=1e-3, batch=128, seed=0, patience=None, log=None):
    """Adam on multi_token_loss. After each epoch the validation loss (Chapter 12's
    next-click cross-entropy on valid's windows) is measured, and the weights of the best
    epoch are kept; with patience p, training stops once p epochs in a row have not beaten
    the best. Returns the model and the validation losses."""
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    curve, best, state = [], math.inf, None
    for e in range(1, epochs + 1):
        model.train()
        for x, t, tn, where, y, fut, wts in future_batches(train, model.pad, batch, gen):
            L = multi_token_loss(model.logits(model.encode(x, t, tn)[where]), fut, wts)
            opt.zero_grad()
            L.backward()
            opt.step()
        v = loss(model, valid)
        curve.append(v)
        if log is not None:
            log(e, v)
        if v < best:
            best, state = v, {k: p.detach().clone() for k, p in model.state_dict().items()}
        elif patience is not None and e - (int(np.argmin(curve)) + 1) >= patience:
            break
    model.load_state_dict(state)
    model.eval()
    return model, curve


class GatedItemLM(ItemLM):
    """Listing 15.2. Chapter 13's HSTU with each video's vector made of a learned map of
    its content plus its ID vector, the ID part multiplied by a gate,
    sigmoid(a x log(1 + clicks) + b), learned from how many training clicks the video's
    ID vector has seen. In training, each use of an ID vector (in a history or in the
    softmax) is replaced with probability `mask` by one learned vector for unknown videos,
    and its click count by 0, so the model must also score a video from its content
    alone. New videos use the unknown vector, with 0 clicks, always."""
    def __init__(self, n_videos, content, clicks, unseen, mask=0.0, gate=True, **kw):
        super().__init__(n_videos, use_id=True, content=content, unseen=unseen, **kw)
        self.mask, self.gate = mask, gate
        self.register_buffer("log_clicks", torch.as_tensor(np.r_[np.log1p(clicks), 0.0], dtype=torch.float32))
        self.unknown = nn.Parameter(torch.zeros(self.d))
        self.a, self.b = nn.Parameter(torch.tensor(1.0)), nn.Parameter(torch.tensor(0.0))

    def vectors(self, ids):
        idv = self.item(ids)
        n = self.log_clicks[ids]
        new = self.unseen[ids.clamp_max(self.n_videos - 1)] & (ids < self.n_videos)
        if self.training and self.mask > 0:
            new = new | (torch.rand(ids.shape) < self.mask)
        idv = torch.where(new[..., None], self.unknown.expand_as(idv), idv)
        n = torch.where(new, torch.zeros_like(n), n)
        g = torch.sigmoid(self.a * n + self.b)[..., None] if self.gate else 1.0
        return self.proj(self.content[ids]) + g * idv


def stabilise(T, W, T0=None):
    """Listing 15.7. Orthogonal low-rank stabilisation (Zielnicki and Hsiao, 2025), in the
    paper's notation: T holds a model's item vectors (items x e) and W its user vectors
    (users x e), and the scores are W @ T.T. Step 1 (low-rank SVD): from the QR
    factorisations T = Q_T R_T and W = Q_W R_W, the SVD of the small e x e matrix
    R_T R_W^T = U S V^T gives items Q_T U S^(1/2) and users Q_W V S^(1/2), so each coordinate
    is one principal direction of the scores, largest first. Step 2 (orthogonal Procrustes,
    only if T0 is given): the rotation that best maps these item vectors onto a reference
    run's stabilised item vectors T0, the same items in the same order. Neither step changes
    any user-item dot product. Returns the new item and user vectors and the maps M_T, M_W
    (T @ M_T and W @ M_W) to apply to later vectors of the same run."""
    Q_T, R_T = np.linalg.qr(T)
    Q_W, R_W = np.linalg.qr(W)
    U, S, Vt = np.linalg.svd(R_T @ R_W.T)
    M_T = R_W.T @ Vt.T / np.sqrt(S)                                  # T M_T = Q_T U S^(1/2)
    M_W = R_T.T @ U / np.sqrt(S)                                     # W M_W = Q_W V S^(1/2)
    if T0 is not None:
        a, _, b = np.linalg.svd((T @ M_T).T @ T0)                    # Procrustes on the items
        M_T, M_W = M_T @ (a @ b), M_W @ (a @ b)
    return T @ M_T, W @ M_W, M_T, M_W
