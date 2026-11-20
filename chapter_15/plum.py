"""YouTube's PLUM at Tesselo (section 15.4): a language model taught Chapter 13's semantic
IDs by continued pre-training, then fine-tuned to write the next video's ID.

PLUM (He and colleagues, 2025) has three stages: start from a pre-trained language model;
continue its pre-training (CPT) on a mix of users' watch histories, written as semantic
IDs, and a corpus that ties each video's ID to its metadata, half of the examples each;
then fine-tune it for generative retrieval, writing the ID of the next video. This
script runs the same three stages at Tesselo's size, in Chapter 12's scale world with
Chapter 13's 348 new videos, and tests PLUM's 2 x 2 ablation:

  start      random       - random weights
             language     - a small language model pre-trained on Tesselo's titles in
                            its six languages (Chapter 11's words), each title written
                            in every language and as pairs of translations
  CPT        no / yes     - CPT: half the examples users' click histories (each video
                            its 4 codes, every next token predicted), half metadata
                            ("[SID] codes [TXT] title" and the reverse) for the videos
                            that exist before day 23

The model is a decoder-only Transformer (width 64, 2 layers, 4 heads, the size of
Chapter 13's and 14's models), its vocabulary the title words, the codes of each level
and a few markers. Fine-tuning (Listing 15.3) appends to each window of clicks one slot
per predicted click: a [NEXT] marker and the click's codes. A slot sees only the clicks
before its request (Chapter 12's contract) and its own earlier codes, and is trained to
write the click's 4 codes. Every stage keeps the epoch with the lowest validation loss
and stops after 2 epochs without a new best.

The test is exact, with no beam search (Chapter 14 covers decoding): for each test click
of a random quarter of the held-out users, every video's log probability, level by level
over the trie of real IDs (chapter_14/gen_models.Trie), each level renormalised over the
codes that lead to a video; the cross-entropy of the clicked video, on common, rare and
new videos, and recall@20. Chapter 14's generator (HSTU encoder and code decoder) is
retrained here as the reference and tested on the same clicks.

Needs chapter_13/sids.npz (python chapter_13/build_ids.py) and PyTorch. Results are kept
in chapter_15/cache. Run from the repository root (about 6 hours on 2 CPUs):
    python chapter_15/plum.py
"""
import os
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "2"
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
for _ch in ("chapter_14", "chapter_13", "chapter_12", "chapter_11", "chapter_09"):
    sys.path.insert(0, str(ROOT / _ch))
from fm_pipeline import CACHE, cached  # noqa: E402
from sid_data import SLICE_NAMES, data, new_videos, slices, world  # noqa: E402
from title_words import WORDS, render  # noqa: E402
from gen_models import Generator, Trie, all_logprob, fit_generator  # noqa: E402
from large_models import tensors  # noqa: E402
from seq_data import environment  # noqa: E402

torch.set_num_threads(2)
D_MODEL, LAYERS, HEADS = 64, 2, 4
PAD, BOS, EOS, SEP, SID, TXT, NEXT = range(7)
MARKERS = 7
TEST_SHARE = 0.25


# ---------------------------------------------------------------- vocabulary and corpora
class Vocab:
    """Token numbers: 7 markers, then every word of Chapter 11's titles in the six
    languages, then the codes of each ID level (a code is a different token at each level)."""
    def __init__(self, codes):
        words = sorted({w for forms in WORDS.values() for f in forms for w in f.split()})
        self.word = {w: MARKERS + i for i, w in enumerate(words)}
        self.k = [int(codes[:, j].max()) + 1 for j in range(codes.shape[1])]
        self.off = np.cumsum([MARKERS + len(words)] + self.k[:-1])
        self.size = int(self.off[-1] + self.k[-1])
        self.sid = torch.as_tensor(codes + self.off[None, :], dtype=torch.long)    # each video's 4 tokens

    def text(self, title, language):
        return [self.word[w] for w in render(title, language).split()]


def text_corpus(V, titles, n_lang=6, pairs=5, seed=0):
    """The language model's corpus: every title in every language, and `pairs` random
    translation pairs per title ("[TXT] title in a [SEP] title in b")."""
    rng = np.random.default_rng(seed)
    out = []
    for t in titles:
        for a in range(n_lang):
            out.append([TXT] + V.text(t, a) + [EOS])
        for _ in range(pairs):
            a, b = rng.choice(n_lang, 2, replace=False)
            out.append([TXT] + V.text(t, a) + [SEP] + V.text(t, b) + [EOS])
    return out


def metadata_corpus(V, titles, languages, videos):
    """CPT's metadata half: each video's ID and its title (in its own language), both ways."""
    out = []
    for v in videos:
        sid, words = V.sid[v].tolist(), V.text(titles[v], languages[v])
        out.append([SID] + sid + [TXT] + words + [EOS])
        out.append([TXT] + words + [SID] + sid + [EOS])
    return out


def history_corpus(V, windows):
    """CPT's behaviour half: each window's clicks as one token stream, [BOS] then each
    video's 4 codes; every next token is predicted."""
    return [[BOS] + V.sid[torch.as_tensor(x)].reshape(-1).tolist() for x in windows[0]]


# ---------------------------------------------------------------- the model
class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.h = heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.out = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def split(self, z):
        return z.unflatten(-1, (self.h, -1)).transpose(-3, -2)                 # ... x heads x T x dh

    def forward(self, x, allow, keep=None):
        q, k, v = self.qkv(self.ln1(x)).chunk(3, -1)
        q, k, v = self.split(q), self.split(k), self.split(v)
        if keep is not None:
            keep.append((k, v))
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=allow[:, None])
        x = x + self.out(a.transpose(-3, -2).flatten(-2))
        return x + self.mlp(self.ln2(x))

    def step(self, x, hk, hv, ak, av):
        """New tokens x (n x d), each attending to the shared history (keys and values hk,
        hv: heads x T x dh), to its own ancestors (ak, av: n x heads x a x dh) and to itself."""
        q, k, v = self.qkv(self.ln1(x)).chunk(3, -1)
        q, k, v = (z.unflatten(-1, (self.h, -1)) for z in (q, k, v))           # n x heads x dh
        scale = q.shape[-1] ** -0.5
        s = torch.cat([torch.einsum("nhd,htd->nht", q, hk), torch.einsum("nhd,nhad->nha", q, ak),
                       (q * k).sum(-1, keepdim=True)], -1) * scale
        w = torch.softmax(s, -1)
        T, a = hk.shape[1], ak.shape[2]
        o = (torch.einsum("nht,htd->nhd", w[..., :T], hv) + torch.einsum("nha,nhad->nhd", w[..., T:T + a], av)
             + w[..., -1:] * v)
        x = x + self.out(o.flatten(-2))
        return x + self.mlp(self.ln2(x)), k[:, :, None], v[:, :, None]


class SidLM(nn.Module):
    """A decoder-only Transformer over words, ID codes and markers, its output layer the
    token embeddings (tied)."""
    def __init__(self, vocab, d=D_MODEL, layers=LAYERS, heads=HEADS, max_pos=320, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.tok, self.pos = nn.Embedding(vocab, d), nn.Embedding(max_pos, d)
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.ln = nn.LayerNorm(d)

    def forward(self, tok, pos, allow, keep=None):
        x = self.tok(tok) + self.pos(pos)
        for b in self.blocks:
            x = b(x, allow, keep)
        return self.ln(x)

    def logits(self, h):
        return h @ self.tok.weight.T


def causal_batch(seqs):
    """Right-padded token streams with their causal masks (a padding token sees itself)."""
    n = max(len(s) for s in seqs)
    tok = torch.full((len(seqs), n), PAD)
    for r, s in enumerate(seqs):
        tok[r, :len(s)] = torch.as_tensor(s)
    real = tok != PAD
    allow = torch.tril(torch.ones(n, n, dtype=torch.bool))[None] & real[:, None, :]
    allow |= torch.eye(n, dtype=torch.bool)[None]
    return tok, torch.arange(n).expand(len(seqs), -1), allow


def lm_loss(model, seqs):
    """Mean cross-entropy of every next token of the streams."""
    tok, pos, allow = causal_batch(seqs)
    h = model(tok, pos, allow)
    return F.cross_entropy(model.logits(h[:, :-1]).flatten(0, 1), tok[:, 1:].flatten(), ignore_index=PAD)


def slot_batch(V, part):
    """Listing 15.3. Chapter 12's windows as one token stream each: [BOS], each history
    video's 4 codes, then one slot per scored target: [NEXT] and the target's first 3
    codes, placed (by position) right after the last click before its request. A slot sees
    the history up to that click and its own earlier tokens, never another slot, so each
    target is written from what the server had at its request. Returns tokens, positions,
    the mask, and for each slot token its row, column and the code it must predict."""
    xs, at, ys, ok = part[0], part[3], part[4], part[5]
    seqs, pos, slots = [], [], []
    for x, p, y, s in zip(xs, at, ys, ok):
        hist = [BOS] + V.sid[torch.as_tensor(x)].reshape(-1).tolist()
        p, y = p[s], y[s]
        tail = V.sid[torch.as_tensor(y)]                                        # m x 4
        seqs.append(hist + torch.cat([torch.full((len(y), 1), NEXT), tail[:, :-1]], 1).reshape(-1).tolist())
        start = 1 + 4 * (p + 1)                                                 # history tokens a slot may see
        pos.append(np.r_[np.arange(len(hist)), (start[:, None] + np.arange(4)).ravel()])
        slots.append((len(hist), start, tail))
    n = max(len(s) for s in seqs)
    tok = torch.full((len(seqs), n), PAD)
    P = torch.zeros((len(seqs), n), dtype=torch.long)
    allow = torch.zeros((len(seqs), n, n), dtype=torch.bool)
    rows, cols, target = [], [], []
    for r, (s, ps, (h, start, tail)) in enumerate(zip(seqs, pos, slots)):
        tok[r, :len(s)], P[r, :len(s)] = torch.as_tensor(s), torch.as_tensor(ps)
        allow[r, :h, :h] = torch.tril(torch.ones(h, h, dtype=torch.bool))
        for i, b in enumerate(start):
            c = h + 4 * i
            allow[r, c:c + 4, :b] = True                                       # the history before the request
            allow[r, c:c + 4, c:c + 4] = torch.tril(torch.ones(4, 4, dtype=torch.bool))
        rows.append(np.full(4 * len(start), r))
        cols.append(h + np.arange(4 * len(start)))
        target.append(tail.reshape(-1))
    allow |= torch.eye(n, dtype=torch.bool)[None]
    return tok, P, allow, (torch.as_tensor(np.concatenate(rows)), torch.as_tensor(np.concatenate(cols))), \
        torch.cat(target)


def slot_loss(model, V, part, reduction="mean"):
    tok, pos, allow, where, target = slot_batch(V, part)
    h = model(tok, pos, allow)[where]
    ce = F.cross_entropy(model.logits(h), target, reduction="none")
    return ce.mean() if reduction == "mean" else ce.sum()


# ---------------------------------------------------------------- training
def batches(n, size, gen=None, lengths=None):
    order = np.argsort(lengths, kind="stable") if lengths is not None else np.arange(n)
    groups = [order[a:a + size] for a in range(0, n, size)]
    if gen is not None:
        groups = [groups[g] for g in gen.permutation(len(groups))]
    return groups


def take(part, idx):
    return [[p[i] for i in idx] for p in part]


def fit_stage(model, train_step, valid_loss, epochs=30, lr=1e-3, patience=2, ckpt=None):
    """Adam; after each epoch the validation loss, keeping the best epoch's weights and
    stopping after `patience` epochs without a new best. With ckpt, the run is saved after
    every epoch and resumes from there if interrupted (the batches of each epoch depend
    only on its number, so a resumed run trains exactly as an uninterrupted one)."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    curve, best, state, done = [], math.inf, None, 0
    path = None if ckpt is None else CACHE / f"ckpt_{ckpt}.pt"
    if path is not None and path.exists():
        c = torch.load(path, weights_only=False)
        model.load_state_dict(c["model"])
        opt.load_state_dict(c["opt"])
        curve, best, state, done = c["curve"], c["best"], c["state"], c["epoch"]
    for e in range(done + 1, epochs + 1):
        if curve and e - 1 - (int(np.argmin(curve)) + 1) >= patience:
            break
        model.train()
        for loss in train_step(e):
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            curve.append(valid_loss())
        if curve[-1] < best:
            best, state = curve[-1], {k: p.detach().clone() for k, p in model.state_dict().items()}
        if path is not None:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "curve": curve, "best": best,
                        "state": state, "epoch": e}, path.with_suffix(".tmp"))
            path.with_suffix(".tmp").rename(path)
        if e - (int(np.argmin(curve)) + 1) >= patience:
            break
    model.load_state_dict(state)
    model.eval()
    return model, curve


def lm_stage(model, train, valid, batch=64, seed=0, **kw):
    """Next-token training on lists of token streams."""
    lt = np.array([len(s) for s in train])

    def step(e):
        gen = np.random.default_rng(seed + e)
        for g in batches(len(train), batch, gen, lt + gen.random(len(lt))):
            yield lm_loss(model, [train[i] for i in g])

    def valid_loss():
        tot = n = 0
        for g in batches(len(valid), 256, lengths=np.array([len(s) for s in valid])):
            seqs = [valid[i] for i in g]
            k = sum(len(s) - 1 for s in seqs)
            tot, n = tot + lm_loss(model, seqs).item() * k, n + k
        return tot / n
    return fit_stage(model, step, valid_loss, **kw)


def cpt_stage(model, hist_train, meta, hist_valid, batch=64, seed=0, **kw):
    """CPT: each epoch, every history stream and as many metadata examples (drawn again
    each epoch), in batches of one kind; validated on held-out users' history streams."""
    lt = np.array([len(s) for s in hist_train])

    def step(e):
        gen = np.random.default_rng(seed + e)
        hb = [("h", g) for g in batches(len(hist_train), batch, gen, lt + gen.random(len(lt)))]
        mi = gen.choice(len(meta), len(hist_train), replace=True)
        mb = [("m", mi[a:a + batch]) for a in range(0, len(mi), batch)]
        for kind, g in [(hb + mb)[i] for i in gen.permutation(len(hb) + len(mb))]:
            yield lm_loss(model, [(hist_train if kind == "h" else meta)[i] for i in g])

    def valid_loss():
        tot = n = 0
        for g in batches(len(hist_valid), 128, lengths=np.array([len(s) for s in hist_valid])):
            seqs = [hist_valid[i] for i in g]
            k = sum(len(s) - 1 for s in seqs)
            tot, n = tot + lm_loss(model, seqs).item() * k, n + k
        return tot / n
    return fit_stage(model, step, valid_loss, **kw)


def finetune_stage(model, V, train, valid, batch=32, seed=0, **kw):
    """Fine-tuning for generative retrieval: the slots' code loss (Listing 15.3), in nats
    per video (the sum over its 4 codes) on the validation windows."""
    lt = np.array([len(x) for x in train[0]])

    def step(e):
        gen = np.random.default_rng(seed + e)
        for g in batches(len(lt), batch, gen, lt + gen.random(len(lt))):
            yield slot_loss(model, V, take(train, g))

    def valid_loss():
        tot = n = 0
        lv = np.array([len(x) for x in valid[0]])
        for g in batches(len(lv), 64, lengths=lv):
            part = take(valid, g)
            tot += slot_loss(model, V, part, "sum").item()
            n += int(sum(s.sum() for s in part[5]))
        return tot / n
    return fit_stage(model, step, valid_loss, **kw)


# ---------------------------------------------------------------- exact test
def trie_nodes(trie, codes):
    """For each depth 1..L-1, each node's last code and its parent node."""
    out = []
    for j in range(codes.shape[1] - 1):
        rep = np.zeros(trie.sizes[j], dtype=np.int64)
        rep[trie.groups[:, j].numpy()] = np.arange(len(codes))
        parent = np.zeros(trie.sizes[j], dtype=np.int64) if j == 0 else trie.groups[rep, j - 1].numpy()
        out.append((torch.as_tensor(codes[rep, j]), torch.as_tensor(parent)))
    return out


@torch.no_grad()
def exact_lm(model, V, trie, nodes, x, p):
    """Every video's log probability for a target read after history position p of the
    window x: the history is encoded once, then the [NEXT] slot and every trie node are
    fed one token each, each attending to the history and its own ancestors; each level
    is renormalised over the codes that lead to a video."""
    L = V.sid.shape[1]
    hist = [BOS] + V.sid[torch.as_tensor(x[:p + 1])].reshape(-1).tolist()
    n = len(hist)
    keep = []
    model(torch.as_tensor([hist]), torch.arange(n)[None], torch.tril(torch.ones(1, n, n, dtype=torch.bool)), keep)
    hk = [k[0] for k, _ in keep]                                               # heads x n x dh, per layer
    hv = [v[0] for _, v in keep]
    anc_k = [[] for _ in model.blocks]
    anc_v = [[] for _ in model.blocks]
    out = torch.zeros(len(V.sid))
    parent = torch.zeros(len(V.sid), dtype=torch.long)                    # each video's node at depth j
    chains = torch.zeros(1, 0, dtype=torch.long)                            # each node's ancestors
    for j in range(L):
        if j == 0:
            tok = torch.as_tensor([NEXT])
        else:
            code, par = nodes[j - 1]
            tok = code + int(V.off[j - 1])
            chains = torch.cat([chains[par], par[:, None]], 1)
        m = len(tok)
        z = model.tok(tok) + model.pos(torch.full((m,), n + j))
        for i, b in enumerate(model.blocks):
            ak = torch.cat([anc_k[i][d][chains[:, d]] for d in range(j)], 2) if j else z.new_zeros(m, b.h, 0, hk[i].shape[-1])
            av = torch.cat([anc_v[i][d][chains[:, d]] for d in range(j)], 2) if j else ak
            z, k_, v_ = b.step(z, hk[i], hv[i], ak, av)
            anc_k[i].append(k_)
            anc_v[i].append(v_)
        a_, b_ = int(V.off[j]), int(V.off[j]) + V.k[j]
        logit = (model.ln(z) @ model.tok.weight[a_:b_].T).masked_fill(trie.child[j] < 0, -math.inf)
        lp = torch.log_softmax(logit, -1)
        out += lp[parent, V.sid[:, j] - a_]
        if j < L - 1:
            parent = trie.groups[:, j]
    out = torch.nan_to_num(out, nan=-math.inf, neginf=-math.inf)
    return out.masked_fill(trie.leaf[trie.groups[:, -1]] < 0, -math.inf)


@torch.no_grad()
def brute_force(model, V, trie, x, p, v):
    """The same log probability of video v, from one ordinary forward pass over the
    history, [NEXT] and v's first 3 codes, renormalised level by level over the trie."""
    toks = [BOS] + V.sid[torch.as_tensor(x[:p + 1])].reshape(-1).tolist() + [NEXT] + V.sid[v, :3].tolist()
    n, T = len(toks) - 4, len(toks)
    h = model(torch.as_tensor([toks]), torch.arange(T)[None], torch.tril(torch.ones(1, T, T, dtype=torch.bool)))[0]
    total, node = 0.0, 0
    for j in range(V.sid.shape[1]):
        a_, b_ = int(V.off[j]), int(V.off[j]) + V.k[j]
        logit = (h[n + j] @ model.tok.weight[a_:b_].T).masked_fill(trie.child[j][node] < 0, -math.inf)
        total += torch.log_softmax(logit, -1)[V.sid[v, j] - a_].item()
        node = int(trie.groups[v, j]) if j < V.sid.shape[1] - 1 else node
    return total


def check_exact(model, V, trie, nodes, D, targets, n=20):
    """For the first n test clicks: the largest difference between the exact scorer and
    brute force (the clicked video and 4 others), and how far the probabilities of all
    videos are from summing to 1."""
    diff = gap = 0.0
    for w, p, y in targets[:n]:
        x = D["test"][0][w]
        lp = exact_lm(model, V, trie, nodes, x, p)
        gap = max(gap, abs(float(torch.exp(lp).sum()) - 1))
        for v in (y, 0, 100, 1000, 3000):
            diff = max(diff, abs(float(lp[v]) - brute_force(model, V, trie, x, p, v)))
    return diff, gap


def rank_stats(lp, y, rng):
    """Cross-entropy of the clicked video and whether it is in the top 20, ties broken at random."""
    above = int((lp > lp[y]).sum())
    ties = int((lp == lp[y]).sum()) - 1
    return float(-lp[y]), above + rng.integers(0, ties + 1) < 20


def test_targets(D, share=TEST_SHARE, seed=15):
    """The scored test targets of a random share of the held-out users: (window, position, video)."""
    users = np.unique(D["test_user"])
    pick = set(np.random.default_rng(seed).choice(users, int(round(share * len(users))), replace=False).tolist())
    out = []
    for w, (p, y, ok) in enumerate(zip(D["test"][3], D["test"][4], D["test"][5])):
        if D["test_user"][w] in pick:
            out += [(w, int(a), int(b)) for a, b, s in zip(p, y, ok) if s]
    return out, len(pick)


def evaluate_lm(model, V, trie, nodes, D, targets):
    rng = np.random.default_rng(0)
    ce, hit = [], []
    for w, p, y in targets:
        c, h = rank_stats(exact_lm(model, V, trie, nodes, D["test"][0][w], p), y, rng)
        ce.append(c)
        hit.append(h)
    return np.array(ce), np.array(hit)


@torch.no_grad()
def evaluate_generator(model, trie, D, targets, batch=64):
    rng = np.random.default_rng(0)
    ce, hit = [], []
    by_w = {}
    for w, p, y in targets:
        by_w.setdefault(w, []).append((p, y))
    ws = sorted(by_w)
    for a in range(0, len(ws), batch):
        g = ws[a:a + batch]
        part = [[D["test"][0][w] for w in g], [D["test"][1][w] for w in g], [D["test"][2][w] for w in g],
                [np.array([p for p, _ in by_w[w]]) for w in g], [np.array([y for _, y in by_w[w]]) for w in g],
                [np.ones(len(by_w[w]), dtype=bool) for w in g]]
        x, t, tn, where, Y = tensors(*part, model.pad)
        lp = all_logprob(model, model.encode(x, t, tn)[where], trie)
        for r in range(len(Y)):
            c, h = rank_stats(lp[r], int(Y[r]), rng)
            ce.append(c)
            hit.append(h)
    return np.array(ce), np.array(hit)


def boot(a, b, user, mask, n=1000, seed=0):
    """Mean of a - b over the clicks in mask, with a 95% interval from draws of whole users."""
    u, inv = np.unique(user[mask], return_inverse=True)
    d = (a - b)[mask]
    s, c = np.bincount(inv, d, len(u)), np.bincount(inv, minlength=len(u))
    rng = np.random.default_rng(seed)
    draws = [(s[i].sum() / c[i].sum()) for i in (rng.integers(0, len(u), len(u)) for _ in range(n))]
    return d.mean(), *np.percentile(draws, [2.5, 97.5])


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    t0 = time.time()
    print(environment())
    W = world()
    D = data(W)
    nv = D["n_videos"]
    new = np.zeros(nv, dtype=bool)
    new[new_videos(W)] = True
    sl = slices(D["clicks"], new)
    codes = np.load(ROOT / "chapter_13" / "sids.npz")["rq_vae_co_clicks"]
    V = Vocab(codes)
    items = W["items"]
    titles = [items["title"][i] for i in W["video"]]
    languages = items["language"][W["video"]]
    old = np.flatnonzero(~new)
    print(f"Vocabulary: {len(V.word):,} title words in 6 languages, ID levels of {'/'.join(map(str, V.k))} codes, "
          f"{MARKERS} markers: {V.size:,} tokens; {nv:,} videos, {new.sum():,} new on day 23  [{time.time() - t0:.0f} s]",
          flush=True)

    # the language model: every catalogue title that exists before day 23 (products too), held out 5% of titles
    shop = [items["title"][i] for i in np.flatnonzero(items["is_product"])]
    corpus_titles = [titles[i] for i in old] + shop
    rng = np.random.default_rng(1)
    hold = rng.random(len(corpus_titles)) < 0.05
    text_train = text_corpus(V, [t for t, h in zip(corpus_titles, hold) if not h])
    text_valid = text_corpus(V, [t for t, h in zip(corpus_titles, hold) if h], seed=2)
    words = sum(len(s) for s in text_train)
    print(f"Language model's corpus: {len(corpus_titles):,} titles (videos before day 23 and products), each in 6 "
          f"languages and 5 translation pairs: {len(text_train):,} training texts, {words:,} tokens; "
          f"{len(text_valid):,} held-out texts", flush=True)

    def make():
        return SidLM(V.size)

    def train_lm():
        t1 = time.time()
        m, curve = lm_stage(make(), text_train, text_valid, ckpt="plum_lm")
        return {"state": m.state_dict(), "curve": curve, "sec": time.time() - t1}
    lm = cached("plum_language_model", train_lm)
    print("  language model: validation loss " + " ".join(f"{v:.4f}" for v in lm["curve"])
          + f"; chosen {int(np.argmin(lm['curve'])) + 1} epochs  [{lm['sec']:.0f} s]", flush=True)

    hist_train, hist_valid = history_corpus(V, D["train"]), history_corpus(V, D["valid"])
    meta = metadata_corpus(V, titles, languages, old)
    print(f"CPT: {len(hist_train):,} history streams ({sum(len(s) for s in hist_train):,} tokens) and "
          f"{len(meta):,} metadata examples of {len(old):,} videos, drawn to match the streams' number each epoch; "
          f"fine-tuning: {sum(int(s.sum()) for s in D['train'][5]):,} targets in {len(D['train'][0]):,} windows",
          flush=True)

    def start(kind):
        m = make()
        if kind == "language":
            m.load_state_dict(lm["state"])
        return m

    results = {}
    for kind in ("random", "language"):
        for cpt in (False, True):
            label = f"{kind} start, {'CPT' if cpt else 'no CPT'}"

            def run(kind=kind, cpt=cpt):
                m, cc = start(kind), None
                if cpt:
                    m, cc = cpt_stage(m, hist_train, meta, hist_valid, ckpt=f"plum_cpt_{kind}")
                m, fc = finetune_stage(m, V, D["train"], D["valid"], ckpt=f"plum_fine_{kind}_{int(cpt)}")
                return {"state": m.state_dict(), "cpt": cc, "fine": fc}
            r = cached("plum_" + label.replace(" ", "_").replace(",", ""), run)
            results[label] = r
            line = f"  {label}: "
            if r["cpt"]:
                line += ("CPT validation " + " ".join(f"{v:.4f}" for v in r["cpt"]) + f", chosen "
                         f"{int(np.argmin(r['cpt'])) + 1} epochs; ")
            line += ("fine-tuning validation (nats per video) " + " ".join(f"{v:.4f}" for v in r["fine"])
                     + f", chosen {int(np.argmin(r['fine'])) + 1} epochs")
            print(line, flush=True)

    def gen_run():
        m, curve = fit_generator(Generator(nv, codes, new), D["train"], D["valid"])
        return {"state": m.state_dict(), "curve": curve}
    g = cached("plum_reference_generator", gen_run)
    gen = Generator(nv, codes, new)
    gen.load_state_dict(g["state"])
    gen.eval()
    print("  Chapter 14's generator: validation (nats per video) " + " ".join(f"{v:.4f}" for v in g["curve"])
          + f", chosen {int(np.argmin(g['curve'])) + 1} epochs", flush=True)

    # ---- the exact test
    trie = Trie(codes, np.ones(nv, dtype=bool), late=new)
    nodes = trie_nodes(trie, codes)
    targets, n_users = test_targets(D)
    y = np.array([t[2] for t in targets])
    user = D["test_user"][np.array([t[0] for t in targets])]
    s = sl[y]
    print(f"Test: {len(targets):,} clicks of {n_users:,} held-out users (a quarter, seed 15): "
          + ", ".join(f"{(s == j).sum():,} {n}" for j, n in enumerate(SLICE_NAMES))
          + f"; the trie has {' / '.join(f'{k:,}' for k in trie.sizes)} nodes  [{time.time() - t0:.0f} s]", flush=True)
    ev = {}
    ev["Chapter 14's generator"] = cached("plum_test_generator", lambda: evaluate_generator(gen, trie, D, targets))
    for label, r in results.items():
        m = make()
        m.load_state_dict(r["state"])
        m.eval()
        ev[label] = cached("plum_test_" + label.replace(" ", "_").replace(",", ""),
                           lambda: evaluate_lm(m, V, trie, nodes, D, targets))
    print(f"  {'':<28}" + "".join(f"{n + ' CE':>11}{'R@20':>8}" for n in SLICE_NAMES) + f"{'all CE':>9}{'R@20':>8}")
    for label, (ce, hit) in ev.items():
        cells = "".join(f"{ce[s == j].mean():>11.4f}{hit[s == j].mean():>8.4f}" for j in range(3))
        print(f"  {label:<28}{cells}{ce.mean():>9.4f}{hit.mean():>8.4f}")
    m = make()
    m.load_state_dict(results["language start, CPT"]["state"])
    m.eval()
    diff, gap = check_exact(m, V, trie, nodes, D, targets)
    print(f"  check (language start, CPT; first 20 test clicks): exact scorer against one full forward pass per video, "
          f"largest difference {diff:.1e} nats; probabilities of all videos sum to 1 within {gap:.1e}")
    print(f"  (CE: cross-entropy of the clicked video, nats, every video scored exactly over the trie of real IDs; "
          f"R@20: recall@20, ties broken at random)  [{time.time() - t0:.0f} s]")
    print("Differences in test cross-entropy (95% interval, 1,000 draws of whole users):")
    pairs = [("random start, CPT", "random start, no CPT"), ("language start, no CPT", "random start, no CPT"),
             ("language start, CPT", "random start, CPT"), ("language start, CPT", "language start, no CPT"),
             ("language start, CPT", "Chapter 14's generator")]
    for a_, b_ in pairs:
        for j, name in list(enumerate(SLICE_NAMES)) + [(None, "all")]:
            mask = np.ones(len(y), dtype=bool) if j is None else s == j
            m_, lo, hi = boot(ev[a_][0], ev[b_][0], user, mask)
            print(f"  {name}: {a_} minus {b_}: {m_:+.4f} ({lo:+.4f} to {hi:+.4f})")
    print(f"[{time.time() - t0:.0f} s]")
