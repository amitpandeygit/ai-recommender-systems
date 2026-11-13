"""Residual quantisation for Chapter 13: semantic IDs from item vectors (sections 13.3 to 13.5).

A semantic ID writes an item as a short sequence of codes. The first code picks the
nearest of K vectors in a first codebook; the second picks the nearest of K vectors in a
second codebook to what is left over, the residual; and so on, for L levels. Items with
similar vectors share their first codes, so the ID is a coarse-to-fine address.

  residual_quantise(x, codebooks) - Listing 13.1: the codes of each row, and its
                                    reconstruction (the sum of its chosen code vectors)
  kmeans(x, K, seed)              - plain k-means with k-means++ starts
  rq_kmeans(x, K, L, seed)        - RQ-KMeans: one k-means per level, on the residuals
  RQVAE                           - Listing 13.2: an encoder, L codebooks on its output
                                    and a decoder, trained to rebuild x (TIGER's RQ-VAE)
  co_click_loss(z, pairs, ...)    - Listing 13.4: a contrastive loss that pulls together
                                    the vectors of videos clicked one after the other
  train_rqvae(x, ...)             - trains an RQVAE, with the co-click loss if pairs are given;
                                    returns the model, its codes and its error
  disambiguate(codes, late)       - Listing 13.3: one more code that numbers the items
                                    sharing the same codes, so every ID is unique
  report(codes, K, ...)           - how evenly codes are used, collisions, and how well
                                    the codes group items with similar hidden taste
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

torch.set_num_threads(2)


def residual_quantise(x, codebooks):
    """Listing 13.1. For each level, the nearest code vector to what is left, which is
    then subtracted. Returns the codes (n x L) and the reconstructions (n x d)."""
    r = np.array(x, dtype=np.float64)
    codes = []
    for C in codebooks:
        d = (r ** 2).sum(1)[:, None] - 2 * r @ C.T + (C ** 2).sum(1)[None, :]   # squared distances
        c = d.argmin(1)
        codes.append(c)
        r = r - C[c]                                                          # the residual
    return np.stack(codes, 1), x - r


def kmeans(x, K, seed=0, iters=50):
    """k-means with k-means++ starts. Returns the K centres."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.float64)
    C = [x[rng.integers(len(x))]]
    d = ((x - C[0]) ** 2).sum(1)
    for _ in range(K - 1):                                       # k-means++: far points are likelier
        C.append(x[rng.choice(len(x), p=d / d.sum() if d.sum() > 0 else None)])   # all covered: any point
        d = np.minimum(d, ((x - C[-1]) ** 2).sum(1))
    C = np.array(C)
    for _ in range(iters):
        a = ((x ** 2).sum(1)[:, None] - 2 * x @ C.T + (C ** 2).sum(1)[None, :]).argmin(1)
        for k in range(K):
            if (a == k).any():
                C[k] = x[a == k].mean(0)
    return C


def rq_kmeans(x, K, L, seed=0, fit=None):
    """RQ-KMeans: k-means at each level on the residuals left by the levels before. The
    codebooks are learned on the rows marked by fit (all rows if None); every row is then
    coded with them."""
    x = np.asarray(x, dtype=np.float64)
    books, r = [], x if fit is None else x[fit]
    for level in range(L):
        C = kmeans(r, K, seed + level)
        books.append(C)
        c = ((r ** 2).sum(1)[:, None] - 2 * r @ C.T + (C ** 2).sum(1)[None, :]).argmin(1)
        r = r - C[c]
    codes, recon = residual_quantise(x, books)
    return codes, recon, books


def mlp(dims):
    """Linear layers of the given sizes, with a ReLU between each two."""
    layers = []
    for a, b in zip(dims[:-1], dims[1:]):
        layers += [nn.Linear(a, b), nn.ReLU()]
    return nn.Sequential(*layers[:-1])


class RQVAE(nn.Module):
    """Listing 13.2. TIGER's RQ-VAE: an encoder maps x to a latent z, L codebooks quantise
    z residual by residual, and a decoder rebuilds x from the sum of the chosen code
    vectors. The straight-through trick lets the decoder's gradient reach the encoder."""
    def __init__(self, d_in, K, L, latent=32, hidden=(256, 128), seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.enc = mlp((d_in, *hidden, latent))
        self.dec = mlp((latent, *hidden[::-1], d_in))
        self.books = nn.Parameter(torch.randn(L, K, latent) * 0.1)     # L codebooks of K vectors

    def quantise(self, z):
        r, q, codes, commit = z, torch.zeros_like(z), [], 0.0
        for C in self.books:
            c = torch.cdist(r, C).argmin(1)
            e = C[c]
            commit = commit + F.mse_loss(e, r.detach()) + 0.25 * F.mse_loss(r, e.detach())
            codes.append(c)
            q = q + e
            r = r - e.detach()
        return torch.stack(codes, 1), z + (q - z).detach(), commit       # straight-through

    def forward(self, x):
        z = self.enc(x)
        codes, q, commit = self.quantise(z)
        return codes, self.dec(q), commit


def co_click_loss(z, pairs, gen, batch=1024, temp=0.1):
    """Listing 13.4. A contrastive loss on pairs of videos clicked one after the other:
    each video's latent should be closer to its partner's than to the other partners in
    the batch (as in PLUM's co-occurrence regularisation). A column whose video is the
    right answer's video, drawn again, is not counted as a wrong answer."""
    i = torch.randint(len(pairs), (batch,), generator=gen)
    va, vb = pairs[i, 0], pairs[i, 1]
    a = F.normalize(z[va], dim=1)
    b = F.normalize(z[vb], dim=1)
    s = a @ b.T / temp
    other = ~torch.eye(batch, dtype=torch.bool)
    target = torch.arange(batch)
    s_ab = s.masked_fill((vb[:, None] == vb[None, :]) & other, -float("inf"))      # the same partner again
    s_ba = s.T.masked_fill((va[:, None] == va[None, :]) & other, -float("inf"))
    return 0.5 * (F.cross_entropy(s_ab, target) + F.cross_entropy(s_ba, target))


def train_rqvae(x, K, L, steps=3000, lr=1e-3, kmeans_init=True, reset_dead=True, seed=0, latent=32,
                pairs=None, w_con=1.0, fit=None):
    """Trains an RQ-VAE (full batch) on the rows of x marked by fit (all rows if None);
    every row is then coded with the trained model. With kmeans_init the codebooks start
    from k-means on the first latents, as TIGER does; with reset_dead, every 100 steps a
    code no item uses is moved onto a randomly chosen residual. With pairs (rows of
    co-clicked videos, all of them among the fitted rows), the co-click loss is added with
    weight w_con. Returns the model, the codes of all rows and the relative reconstruction
    error over all rows."""
    X = torch.as_tensor(np.asarray(x, dtype=np.float32))
    rows = torch.arange(len(X)) if fit is None else torch.as_tensor(np.flatnonzero(fit))
    m = RQVAE(X.shape[1], K, L, latent=latent, seed=seed)
    gen = torch.Generator().manual_seed(seed)
    P = None if pairs is None else torch.as_tensor(pairs)
    if P is not None:
        assert fit is None or np.asarray(fit)[np.asarray(pairs)].all(), "co-click pairs must be fitted rows"
        opt = torch.optim.Adam(m.enc.parameters(), lr=lr)              # learn the co-clicks first, then quantise
        for step in range(steps // 3):
            loss = co_click_loss(m.enc(X), P, gen)
            opt.zero_grad()
            loss.backward()
            opt.step()
    if kmeans_init:
        with torch.no_grad():
            r = m.enc(X[rows])
            for level in range(L):
                C = torch.as_tensor(kmeans(r.numpy(), K, seed + level), dtype=torch.float32)
                m.books[level] = C
                r = r - C[torch.cdist(r, C).argmin(1)]
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for step in range(1, steps + 1):
        z = m.enc(X)
        codes, q, commit = m.quantise(z[rows])
        loss = F.mse_loss(m.dec(q), X[rows]) + commit
        if P is not None:
            loss = loss + w_con * co_click_loss(z, P, gen)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if reset_dead and step % 100 == 0 and step < steps:
            with torch.no_grad():
                r = m.enc(X[rows])
                for level in range(L):
                    c = torch.cdist(r, m.books[level]).argmin(1)
                    dead = torch.nonzero(torch.bincount(c, minlength=K) == 0).flatten()
                    if len(dead):
                        pick = torch.randint(len(r), (len(dead),), generator=gen)
                        m.books[level][dead] = r[pick]
                    r = r - m.books[level][torch.cdist(r, m.books[level]).argmin(1)]
    with torch.no_grad():
        codes, rec, _ = m(X)
        err = float(((rec - X) ** 2).sum() / (X ** 2).sum())
    return m, codes.numpy(), err


def disambiguate(codes, late=None):
    """Listing 13.3. Appends one code: 0 for the first item with a given sequence of codes,
    1 for the second, and so on, so that no two items share an ID (TIGER). Items marked
    late (uploaded later) take their numbers after the others, so earlier IDs never change."""
    late = np.zeros(len(codes), dtype=bool) if late is None else np.asarray(late)
    _, group = np.unique(codes, axis=0, return_inverse=True)
    group = group.ravel()
    order = np.lexsort((np.arange(len(codes)), late, group))      # by group, then earlier items first
    rank = np.empty(len(codes), dtype=np.int64)
    first = np.r_[0, np.flatnonzero(np.diff(group[order])) + 1]
    sizes = np.diff(np.r_[first, len(order)])
    rank[order] = np.arange(len(order)) - np.repeat(first, sizes)
    return np.hstack([codes, rank[:, None]])


def perplexity(c, K):
    """exp of the entropy of how often each code is used: K when all are used equally."""
    p = np.bincount(c, minlength=K) / len(c)
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def report(codes, K, taste=None, category=None):
    """Code usage per level, collisions, and two checks against what the simulator knows:
    the share of the variance of the hidden tastes explained by the group means of each
    prefix of codes (the first code, the first two, ...), and the share of videos in the
    most common category of their first code's group."""
    out = {"used": [int(len(np.unique(codes[:, j]))) for j in range(codes.shape[1])],
           "perplexity": [perplexity(codes[:, j], K) for j in range(codes.shape[1])]}
    _, inv, cnt = np.unique(codes, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    out["unique_ids"] = int(len(cnt))
    out["colliding_items"] = int((cnt[inv] > 1).sum())
    out["largest_group"] = int(cnt.max())
    if taste is not None:
        total = ((taste - taste.mean(0)) ** 2).sum()
        out["taste_explained"] = []
        for j in range(1, codes.shape[1] + 1):
            _, g = np.unique(codes[:, :j], axis=0, return_inverse=True)
            g = g.ravel()
            means = np.stack([np.bincount(g, weights=taste[:, k]) for k in range(taste.shape[1])], 1)
            means /= np.bincount(g)[:, None]
            out["taste_explained"].append(float(1 - ((taste - means[g]) ** 2).sum() / total))
    if category is not None:
        top = sum(np.bincount(category[codes[:, 0] == c]).max() for c in np.unique(codes[:, 0]))
        out["first_code_category"] = float(top / len(codes))
    return out
