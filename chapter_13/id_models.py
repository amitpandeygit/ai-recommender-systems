"""How a sequence model reads an item: a random ID, content, or a semantic ID (sections 13.2 and 13.4).

Chapter 12's HSTU (chapter_12/large_models.py), width 64 with 2 layers, reads each
video as a learned vector and scores every video with the same vectors. Only that
vector changes between the arms here:

  random ID   - one learned vector per video (Chapter 12)
  content     - a learned linear map of the video's content vector (sid_data.content)
  semantic ID - Listing 13.5: one learned vector per prefix of the video's codes, added
                up: the first code, the first two codes, and so on (the "prefix-ngram"
                method of Zheng and colleagues at Meta, 2025)
  semantic ID + random ID - both added
  semantic ID + content   - both added
  drop                    - in training, with this probability a video keeps only its
                            first 1 to L-1 rows, as a new video does

New videos are uploaded on day 23, so their scores are left out of the softmax in
training and in validation, which both end before day 23: a model cannot learn that a
video exists before it is uploaded. In the test, from day 23, every video is scored.
The rows that only new videos use (their random-ID rows, and the semantic-ID rows of
prefixes no trained video shares) therefore get no gradient; they start at zero, so a
new video's vector is what it shares with trained videos and no random noise.

  ItemLM(n, ...)              - the model; a subclass of Chapter 12's SeqLM
  prefixes(codes, late)       - each video's row at each prefix depth, numbered in order
                                of first appearance (videos marked late last), and the
                                number of rows at each depth
  evaluate(model, test)       - each test click's cross-entropy, and how many videos
                                score above it and tie with it
  hit_rate(r, k)              - recall@k with ties broken at random
  table_rows(model)           - how many item vectors the model learns
"""
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "chapter_12"))
from large_models import SeqLM, tensors  # noqa: E402

D_MODEL, LAYERS = 64, 2


def prefixes(codes, late=None):
    """For each depth j = 1..L, each video's row among the distinct first-j codes, rows
    numbered in order of first appearance, with videos marked late (uploaded later) after
    all others: adding late videos appends rows and never renumbers earlier ones.
    Returns a (V x L) array of row numbers and the number of rows at each depth."""
    late = np.zeros(len(codes), dtype=bool) if late is None else np.asarray(late)
    order = np.lexsort((np.arange(len(codes)), late))              # earlier videos first
    groups, sizes = [], []
    for j in range(1, codes.shape[1] + 1):
        _, first, g = np.unique(codes[order, :j], axis=0, return_index=True, return_inverse=True)
        rank = np.empty(len(first), dtype=np.int64)
        rank[np.argsort(first, kind="stable")] = np.arange(len(first))
        row = np.empty(len(codes), dtype=np.int64)
        row[order] = rank[g.ravel()]
        groups.append(row)
        sizes.append(len(first))
    return np.stack(groups, 1), sizes


class ItemLM(SeqLM):
    """Listing 13.5. Chapter 12's HSTU with a video's vector built from a random ID, a
    semantic ID, content, or a sum of these. `unseen` marks the videos uploaded on day 23,
    left out of the softmax until `uploaded` is set (the test)."""
    def __init__(self, n_videos, use_id=True, codes=None, content=None, unseen=None, drop=0.0, d=D_MODEL,
                 layers=LAYERS, max_len=64, seed=0):
        super().__init__(n_videos, "HSTU", d, layers, max_len=max_len, seed=seed)
        self.use_id, self.drop = use_id, drop
        if not use_id:
            nn.init.zeros_(self.item.weight)
            self.item.weight.requires_grad_(False)
        self.sid = None
        if codes is not None:
            groups, sizes = prefixes(codes, late=unseen)
            self.register_buffer("groups", torch.as_tensor(np.vstack([groups, np.array(sizes)[None]])))  # pad row
            self.sid = nn.ModuleList([nn.Embedding(s + 1, d) for s in sizes])
            for e in self.sid:
                nn.init.normal_(e.weight, std=d ** -0.5)
        self.proj = None
        if content is not None:
            c = np.vstack([content, np.zeros((1, content.shape[1]), dtype=np.float32)])
            self.register_buffer("content", torch.as_tensor(c, dtype=torch.float32))
            self.proj = nn.Linear(content.shape[1], d, bias=False)
        mask = np.zeros(n_videos, dtype=bool) if unseen is None else unseen
        self.register_buffer("unseen", torch.as_tensor(mask))
        self.uploaded = False                         # True once the new videos exist (the test)
        with torch.no_grad():                         # rows only new videos use are never trained: start at 0
            self.item.weight[:n_videos][self.unseen] = 0.0
            for j, e in enumerate(self.sid or []):
                trained = np.zeros(e.num_embeddings, dtype=bool)
                trained[groups[~mask, j]] = True
                e.weight[torch.as_tensor(~trained)] = 0.0

    def vectors(self, ids):
        v = self.item(ids) if self.use_id else 0.0
        if self.sid is not None:
            g = self.groups[ids]
            rows = torch.full(ids.shape, len(self.sid))                  # how many rows each video keeps
            if self.training and self.drop > 0:                           # sometimes keep only the first 1 to L-1
                cut = torch.rand(ids.shape) < self.drop
                rows = torch.where(cut, torch.randint(1, len(self.sid), ids.shape), rows)
            v = v + sum(e(g[..., j]) * (j < rows)[..., None] for j, e in enumerate(self.sid))
        if self.proj is not None:
            v = v + self.proj(self.content[ids])
        return v

    def logits(self, h):
        out = h @ self.vectors(torch.arange(self.n_videos)).T
        if not self.uploaded:
            out = out.masked_fill(self.unseen, -math.inf)                 # not uploaded yet
        return out


def table_rows(model):
    """The number of item vectors a model learns: one per video for a random ID, one per
    group at each depth for a semantic ID (the content map has no rows)."""
    n = model.n_videos if model.use_id else 0
    if model.sid is not None:
        n += sum(e.num_embeddings - 1 for e in model.sid)
    return n


@torch.no_grad()
def evaluate(model, test, batch=256):
    """Every scored test click's video, cross-entropy (nats), the number of videos that
    score above it ("above") and that tie with it ("ties"), and the window it came from,
    in the order of the windows. New videos are scored: the test starts on day 23."""
    model.eval()
    model.uploaded = True
    out = {"video": [], "ce": [], "above": [], "ties": [], "window": []}
    n = len(test[0])
    for a in range(0, n, batch):
        part = [p[a:a + batch] for p in test]
        x, t, tn, where, y = tensors(*part, model.pad)
        lg = model.logits(model.encode(x, t, tn)[where])
        target = lg.gather(1, y[:, None]).squeeze(1)
        out["video"].append(y.numpy())
        out["ce"].append((torch.logsumexp(lg, 1) - target).numpy())
        out["above"].append((lg > target[:, None]).sum(1).numpy())
        out["ties"].append((lg == target[:, None]).sum(1).numpy() - 1)
        out["window"].append(where[0].numpy() + a)
    model.uploaded = False
    return {k: np.concatenate(v) for k, v in out.items()}


def hit_rate(r, k):
    """Recall@k for each click, with ties broken at random: the chance that the clicked
    video is in the top k when `above` videos score higher and `ties` others score the
    same. It shares ties + 1 places, of which k - above (if any) are inside the top k."""
    return np.clip((k - r["above"]) / (r["ties"] + 1), 0.0, 1.0)
