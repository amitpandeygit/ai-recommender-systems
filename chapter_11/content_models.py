"""Content-based scoring for Chapter 11: TF-IDF, history profiles and fused content
vectors (sections 11.3 and 11.4).

  tfidf(docs)              - unit-length TF-IDF vectors of documents given as token lists
  profiles(F, H)           - each request's profile: the mean content vector of the
                             videos in its history (Listing 11.2)
  scorer(P, F)             - f(a, b): the scores of requests a to b for every item
  fuse(parts, weights)     - one content vector per item from several modalities, each
                             part unit length and weighted, renormalised over the parts
                             the item has (Listing 11.3)
"""
import numpy as np
from scipy import sparse


def tfidf(docs):
    """Unit-length TF-IDF rows (sparse), one per document; docs are lists of tokens."""
    vocab = {}
    rows, cols = [], []
    for r, d in enumerate(docs):
        for t in d:
            rows.append(r)
            cols.append(vocab.setdefault(t, len(vocab)))
    X = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(docs), len(vocab)))
    X.sum_duplicates()
    df = np.bincount(X.indices, minlength=len(vocab))
    X = X @ sparse.diags(np.log(len(docs) / df))
    norm = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
    return sparse.diags(1.0 / np.maximum(norm, 1e-12)) @ X


def profiles(F, H):
    """Listing 11.2. Each request's profile: the mean of the content vectors F of the
    videos in its history H (one row of item ids per request, -1 for padding). A request
    with no history gets a zero profile, so every item scores 0."""
    rows = np.repeat(np.arange(len(H)), H.shape[1])
    ids = H.ravel()
    keep = ids >= 0
    n = np.maximum((H >= 0).sum(axis=1), 1)
    M = sparse.csr_matrix((1.0 / n[rows[keep]], (rows[keep], ids[keep])), shape=(len(H), F.shape[0]))
    P = M @ F
    return P.toarray() if sparse.issparse(P) else P


def scorer(P, F):
    """f(a, b): the dot products of profiles a to b with every item's content vector."""
    Ft = F.T.tocsr() if sparse.issparse(F) else np.ascontiguousarray(F.T)
    return lambda a, b: np.asarray(P[a:b] @ Ft, dtype=np.float32)


def fuse(parts, weights):
    """Listing 11.3. One content vector per item: the vector of each modality times its
    weight, side by side, scaled by the weights of the modalities the item has. When every
    block it has is unit length, the whole has length 1. A missing modality is a zero
    block, so an item without sound is described by what it has. (Section 11.4's filled
    sound is a deliberate exception: a shorter block, counted as present.)"""
    blocks = [w * x for (x, _), w in zip(parts, weights)]
    has = np.stack([h for _, h in parts], axis=1)
    w2 = np.asarray(weights, dtype=np.float64) ** 2
    length = np.sqrt(has @ w2)
    return (np.hstack(blocks) / np.maximum(length, 1e-12)[:, None]).astype(np.float32)
