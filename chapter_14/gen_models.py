"""Generative retrieval for Chapter 14: a model that writes the next video code by code
(sections 14.2 to 14.7).

The encoder is Chapter 13's: Chapter 12's HSTU, width 64 with 2 layers, reading each
video of the history through its semantic ID (chapter_13/id_models.ItemLM, Listing
13.5, one learned row per prefix of the ID). Only the output changes. Where Chapter
13's model scores every video with a full softmax, the generator writes the next
video's semantic ID one code at a time, as TIGER does (Rajput and colleagues, 2023):

    P(video | history) = P(c1 | h) x P(c2 | h, c1) x P(c3 | h, c1, c2) x P(c4 | h, c1, c2, c3)

  Generator           - Listing 14.1: the HSTU encoder and a small code decoder
  Trie                - Listing 14.2: which codes may follow each prefix, so that only
                        IDs of real videos can be written
  beam_search(...)    - Listing 14.3: the B best IDs, level by level, with or without
                        the trie; the beam can differ at each level
  all_logprob(...)    - the generator's log probability of every video at once (every
                        prefix is shared), with the trie's renormalisation: the exact
                        ranking that beam search approximates
  ListGenerator       - Listing 14.5: writes a whole row, one video after another, each
                        conditioned on the videos already chosen
  fit_generator(...)  - Adam on the codes' cross-entropy (teacher forcing), with an
                        optional weight per target (section 14.7), keeping the epoch
                        with the lowest validation loss, as Chapter 12's fit does
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
from id_models import D_MODEL, LAYERS, ItemLM, prefixes  # noqa: E402
from large_models import tensors  # noqa: E402

torch.set_num_threads(2)


class Generator(ItemLM):
    """Listing 14.1. Chapter 13's HSTU reads the history through semantic IDs; a small
    decoder then writes the next video's codes. Level j sees the user's state and the
    codes chosen at the levels before it."""
    def __init__(self, n_videos, codes, unseen, d=D_MODEL, layers=LAYERS, seed=0):
        super().__init__(n_videos, use_id=False, codes=codes, unseen=unseen, d=d, layers=layers, seed=seed)
        self.sizes = [int(codes[:, j].max()) + 1 for j in range(codes.shape[1])]
        self.register_buffer("codes", torch.as_tensor(codes, dtype=torch.long))
        self.code_emb = nn.ModuleList([nn.Embedding(k, d) for k in self.sizes[:-1]])   # a code, at its level
        self.step = nn.ModuleList([nn.Sequential(nn.Linear(2 * d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
                                   for _ in self.sizes[:-1]])
        self.head = nn.ModuleList([nn.Linear(d, k) for k in self.sizes])               # scores of level j's codes

    def advance(self, z, j, c):
        """The decoder's state after code c is chosen at level j."""
        return F.layer_norm(z + self.step[j](torch.cat([z, self.code_emb[j](c)], -1)), (self.d,))

    def code_loss(self, h, y, weight=None):
        """Teacher forcing: the cross-entropy of each level's true code, given the true
        codes before it, summed over the levels (= -log P of the video, with no trie)."""
        c, z, total = self.codes[y], h, 0.0
        for j in range(len(self.sizes)):
            total = total + F.cross_entropy(self.head[j](z), c[:, j], reduction="none")
            if j < len(self.sizes) - 1:
                z = self.advance(z, j, c[:, j])
        return total.mean() if weight is None else (total * weight).sum() / weight.sum()


class Trie:
    """Listing 14.2. The valid IDs as a tree: node n at depth j is one distinct prefix
    of j codes (Chapter 13's prefix rows), and child[j][n, c] is the node reached by
    adding code c, or -1 if no video has that prefix. `live` marks the videos that may
    be generated, so a video that is not uploaded yet, or not allowed, has no path."""
    def __init__(self, codes, live, late=None):
        groups, sizes = prefixes(codes, late)
        self.groups, self.sizes = torch.as_tensor(groups), sizes
        V, L = codes.shape
        k = [int(codes[:, j].max()) + 1 for j in range(L)]
        self.child = [np.full((1, k[0]), -1)] + [np.full((sizes[j], k[j + 1]), -1) for j in range(L - 1)]
        self.leaf = np.full(sizes[-1], -1)                       # the video at each full ID
        for v in np.flatnonzero(live):
            parent = 0
            for j in range(L):
                self.child[j][parent, codes[v, j]] = groups[v, j]
                parent = groups[v, j]
            self.leaf[parent] = v
        self.child = [torch.as_tensor(c) for c in self.child]
        self.leaf = torch.as_tensor(self.leaf)


@torch.no_grad()
def beam_search(model, h, trie=None, beams=(16, 16, 16, 16), banned=None):
    """Listing 14.3. Keeps the beams[j] best prefixes at level j, scoring each by the sum
    of its codes' log probabilities. With a trie, only codes that lead to a live video
    are allowed, and each level's probabilities are renormalised over them. banned
    (n x V, True = not allowed for this request) removes videos at the last level.
    Returns the videos (n x beams[-1]; -1 where an ID names no video, or where fewer
    allowed videos exist than were asked for) and their scores."""
    n = len(h)
    z = h[:, None, :]                                          # n x 1 x d: one beam, the empty prefix
    node = torch.zeros(n, 1, dtype=torch.long)
    score = torch.zeros(n, 1)
    codes = torch.zeros(n, 1, 0, dtype=torch.long)
    for j, B in enumerate(beams):
        logit = model.head[j](z)                               # n x beams x K
        if trie is not None:
            nxt = trie.child[j][node.clamp(min=0)]             # n x beams x K: child node, -1 if none
            ok = (nxt >= 0) & (node >= 0)[:, :, None]          # a dead prefix has no children
            if j == len(beams) - 1 and banned is not None:
                vid = trie.leaf[nxt.clamp(min=0)].clamp(min=0)
                ok &= ~torch.gather(banned, 1, vid.flatten(1)).view(vid.shape)
            logit = logit.masked_fill(~ok, -math.inf)
        lp = torch.log_softmax(logit, -1)
        lp = torch.nan_to_num(lp, nan=-math.inf, neginf=-math.inf)   # a beam with no valid child
        total = (score[:, :, None] + lp).flatten(1)             # n x (beams x K)
        top = total.topk(min(B, total.shape[1]), dim=1)
        k = logit.shape[-1]
        parent, c = top.indices // k, top.indices % k
        score = top.values
        codes = torch.cat([torch.gather(codes, 1, parent[:, :, None].expand(-1, -1, codes.shape[2])),
                           c[:, :, None]], 2)
        if trie is not None:
            node = torch.gather(nxt.flatten(1), 1, top.indices)
        z = torch.gather(z, 1, parent[:, :, None].expand(-1, -1, z.shape[2]))
        if j < len(beams) - 1:
            z = model.advance(z, j, c)
    if trie is not None:
        video = torch.where(node >= 0, trie.leaf[node.clamp(min=0)], -1)
    else:
        video = lookup(model.codes, codes)
    video = torch.where(torch.isfinite(score), video, -1)
    return video, score


def lookup(all_codes, codes):
    """The video with each sequence of codes, or -1 if no video has it."""
    key = {tuple(r): v for v, r in enumerate(all_codes.tolist())}
    flat = codes.reshape(-1, codes.shape[-1]).tolist()
    return torch.as_tensor([key.get(tuple(r), -1) for r in flat]).view(codes.shape[:-1])


@torch.no_grad()
def all_logprob(model, h, trie, constrained=True):
    """The generator's log probability of every video (n x V), computed once per prefix
    (the trie's nodes). Constrained: each level is renormalised over the codes that lead
    to live videos, which is the distribution constrained beam search explores; videos
    that are not live get -inf. Otherwise the plain product of the levels' softmaxes."""
    codes = model.codes
    V, L = codes.shape
    groups, sizes = trie.groups, trie.sizes
    out = torch.zeros(len(h), V)
    z = h[:, None, :]                                          # states of the nodes at depth j
    parent = torch.zeros(V, dtype=torch.long)                  # each video's node at depth j
    for j in range(L):
        logit = model.head[j](z)                               # n x nodes x K
        if constrained:
            logit = logit.masked_fill(trie.child[j][None] < 0, -math.inf)
        lp = torch.log_softmax(logit, -1)
        out += lp[:, parent, codes[:, j]]
        if j < L - 1:
            first = torch.zeros(sizes[j], dtype=torch.long)
            first[groups[:, j]] = torch.arange(V)               # a video in each node, to find its code
            nz = model.advance(z[:, parent[first]], j, codes[first, j].expand(len(h), -1))
            z, parent = nz, groups[:, j]
    out = torch.nan_to_num(out, nan=-math.inf, neginf=-math.inf)
    return out.masked_fill((trie.leaf[groups[:, -1]] < 0)[None], -math.inf) if constrained else out


class ListGenerator(Generator):
    """Listing 14.5 (first part). Writes a row video by video. Before each video, a GRU
    cell adds the vector of the video chosen just before to the state, so each choice is
    conditioned on the user's history and on the videos already in the row."""
    def __init__(self, n_videos, codes, unseen, d=D_MODEL, layers=LAYERS, seed=0):
        super().__init__(n_videos, codes, unseen, d=d, layers=layers, seed=seed)
        self.row = nn.GRUCell(d, d)

    def list_loss(self, h, lists):
        """lists: n x m videos (-1 = no more). The loss of each video given the history
        and the true videos before it in the list (teacher forcing)."""
        g, total, count = h, 0.0, 0
        for k in range(lists.shape[1]):
            y = lists[:, k]
            ok = y >= 0
            if not ok.any():
                break
            total = total + self.code_loss(g[ok], y[ok]) * ok.sum()
            count += int(ok.sum())
            prev = self.vectors(y.clamp(min=0)) * math.sqrt(self.d)
            g = torch.where(ok[:, None], self.row(prev, g), g)
        return total / max(count, 1)


@torch.no_grad()
def generate_row(model, h, trie, banned, creator, beam=16, k=10, cap=2):
    """Listing 14.5 (second part). A row of k videos written one after another: each
    video is the best of a constrained beam search from the state after the videos
    already chosen. A video already in the row, or whose creator already has `cap`
    videos in it, is banned inside the search, not afterwards."""
    n = len(h)
    banned = banned.clone()
    row = torch.full((n, k), -1, dtype=torch.long)
    count = torch.zeros(n, int(creator.max()) + 1, dtype=torch.long)
    g = h
    for slot in range(k):
        v, _ = beam_search(model, g, trie, (beam,) * len(model.sizes), banned)
        best = v[:, 0]
        row[:, slot] = best
        ok = best >= 0
        idx = torch.nonzero(ok).flatten()
        banned[idx, best[ok]] = True
        cr = creator[best.clamp(min=0)]
        count[idx, cr[ok]] += 1
        full = count[idx, cr[ok]] >= cap
        if full.any():
            same = creator[None, :] == cr[ok][full][:, None]
            banned[idx[full]] |= same
        prev = model.vectors(best.clamp(min=0)) * math.sqrt(model.d)
        g = torch.where(ok[:, None], model.row(prev, g), g)
    return row


def weighted_batches(data, weights, pad, batch, gen=None):
    """Chapter 12's batches (large_models.batches), with one weight per scored target."""
    length = torch.as_tensor([len(x) for x in data[0]], dtype=torch.float32)
    order = torch.argsort(length + (torch.rand(len(length), generator=gen) if gen is not None else 0))
    groups = list(torch.split(order, batch))
    if gen is not None:
        groups = [groups[g] for g in torch.randperm(len(groups), generator=gen).tolist()]
    for g in groups:
        g = g.tolist()
        part = [[p[i] for i in g] for p in data]
        w = None if weights is None else torch.as_tensor(
            np.concatenate([weights[i][data[5][i]] for i in g]), dtype=torch.float32)
        yield (*tensors(*part, pad), w)


def code_eval(model, data, batch=256):
    """Mean teacher-forced code loss (nats per video, no trie) of every scored target."""
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for x, t, tn, where, y, _ in weighted_batches(data, None, model.pad, batch):
            h = model.encode(x, t, tn)[where]
            total += model.code_loss(h, y).item() * len(y)
            count += len(y)
    return total / max(count, 1)


def fit_generator(model, train, valid, epochs=30, lr=1e-3, batch=128, seed=0, patience=2, weights=None,
                  log=None):
    """Adam on the code loss of every scored next click (weighted, if weights are given),
    keeping the epoch with the lowest validation loss and stopping after `patience`
    epochs without a new best, as Chapter 12's fit does."""
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    curve, best, state = [], math.inf, None
    for e in range(1, epochs + 1):
        model.train()
        for x, t, tn, where, y, w in weighted_batches(train, weights, model.pad, batch, gen):
            loss = model.code_loss(model.encode(x, t, tn)[where], y, w)
            opt.zero_grad()
            loss.backward()
            opt.step()
        v = code_eval(model, valid)
        curve.append(v)
        if log is not None:
            log(e, v)
        if v < best:
            best, state = v, {k: p.detach().clone() for k, p in model.state_dict().items()}
        elif e - (int(np.argmin(curve)) + 1) >= patience:
            break
    model.load_state_dict(state)
    model.eval()
    return model, curve
