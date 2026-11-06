"""Listing 9.2 - Simple sequence baselines (section 9.3).

Four ways to score the next video from what a user just did, none of them neural:

  popularity      - clicks of the last 7 days before the cutoff
  co-occurrence   - item-to-item: videos clicked by the same users as the history's
                    videos (cosine over users), summed over the history
  session kNN     - the 100 past visits most similar to the current one (cosine over
                    their sets of clicked videos); each scores its videos by its
                    similarity (Ludewig and Jannach, 2018). Before the first click of a
                    visit, the user's last 5 clicks stand in for the visit. The code
                    compares the visit with every past visit; a production system
                    would first look up the past visits that share a video with it.
  Markov chain    - first order: how often each video followed the history's last
                    video, in the same user's clicks, mixed with a little popularity

All of them are fitted on the clicks that had arrived by the cutoff.
"""
import numpy as np
from scipy import sparse

from seq_data import known

SHRINK = 0.05                            # weight of popularity behind the Markov chain


def popularity(W, cutoff, days=7):
    C = W["clicks"]
    ok = known(C, cutoff) & (C["t"] >= cutoff - days * 86_400)
    return np.bincount(C["item"][ok], minlength=len(W["items"]["vec"])).astype(float)


def cooccurrence(seqs, n_items):
    """Cosine similarity between videos over the users who clicked them."""
    rows = np.concatenate([np.full(len(s), u) for u, s in enumerate(seqs)])
    M = sparse.csr_matrix((np.ones(len(rows)), (rows, np.concatenate(seqs))), shape=(len(seqs), n_items))
    M.data[:] = 1.0
    M.sum_duplicates()
    M.data[:] = 1.0
    G = (M.T @ M).tocsr()
    norm = np.sqrt(np.maximum(G.diagonal(), 1.0))
    G = sparse.diags(1 / norm) @ G @ sparse.diags(1 / norm)
    G.setdiag(0)
    return G.tocsr()


def cooc_scores(G, H):
    rows = np.repeat(np.arange(len(H)), H.shape[1])
    flat = H.ravel()
    Hm = sparse.csr_matrix((np.ones((flat >= 0).sum()), (rows[flat >= 0], flat[flat >= 0])),
                           shape=(len(H), G.shape[0]))
    return (Hm @ G).toarray().astype(np.float32)


def markov(seqs, n_items):
    """Listing 9.2 (first part). Counts of 'video j was clicked right after video i'."""
    a = np.concatenate([s[:-1] for s in seqs])
    b = np.concatenate([s[1:] for s in seqs])
    T = sparse.csr_matrix((np.ones(len(a)), (a, b)), shape=(n_items, n_items))
    T.sum_duplicates()
    return T


def markov_scores(T, H, pop):
    """P(next | last clicked video): 95% from the counts and 5% from popularity, or
    popularity alone when the last video was never followed by anything (or there
    is no last video). Each row sums to 1."""
    last = H[:, -1]
    counts = T[np.maximum(last, 0)].toarray().astype(np.float32)
    counts[last < 0] = 0
    total = counts.sum(axis=1, keepdims=True)
    share = (pop / pop.sum()).astype(np.float32)[None, :]
    mixed = (1 - np.float32(SHRINK)) * counts / np.maximum(total, 1) + np.float32(SHRINK) * share
    return np.where(total > 0, mixed, share)


def session_matrix(W, cutoff):
    """Past visits as rows of a binary visit x video matrix (visits with 2+ clicks)."""
    C = W["clicks"]
    ok = known(C, cutoff)
    ses, item = C["session"][ok], C["item"][ok]
    ids, row = np.unique(ses, return_inverse=True)
    M = sparse.csr_matrix((np.ones(len(row)), (row, item)), shape=(len(ids), len(W["items"]["vec"])))
    M.data[:] = 1.0
    M.sum_duplicates()
    M.data[:] = 1.0
    keep = np.asarray(M.sum(axis=1)).ravel() >= 2
    return M[keep].tocsr()


def session_knn(M, context, k=100):
    """Listing 9.2 (second part). Score videos by the k past visits most like the
    current one: cosine between binary sets, then the sum of their similarities.
    With fewer than k past visits, all of them vote; with none, every score is 0
    and the caller's fallback (popularity) decides."""
    n_visits = M.shape[0]
    if n_visits == 0:
        return np.zeros((context.shape[0], M.shape[1]), dtype=np.float32)
    k = min(k, n_visits)
    size = np.sqrt(np.asarray(M.sum(axis=1)).ravel())
    sims = (context @ M.T).toarray().astype(np.float32) / np.maximum(size[None, :], 1e-12)
    sims /= np.maximum(np.sqrt(np.asarray(context.sum(axis=1))), 1e-12)
    top = np.argsort(-sims, axis=1, kind="stable")[:, :k]          # the k most similar; ties: the earlier visit
    S = sparse.csr_matrix((np.take_along_axis(sims, top, axis=1).ravel(),
                           (np.repeat(np.arange(len(top)), k), top.ravel())), shape=sims.shape)
    return (S @ M).toarray().astype(np.float32)


def contexts(H, in_visit, n_items, fallback=5):
    """The current visit's clicks; before the first click, the last `fallback` clicks."""
    n = np.where(in_visit > 0, in_visit, fallback)
    rows, cols = [], []
    for k, (h, m) in enumerate(zip(H, n)):
        c = h[-m:] if m > 0 else h[:0]
        c = c[c >= 0]
        rows += [k] * len(c)
        cols += c.tolist()
    X = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(H), n_items))
    X.data[:] = 1.0
    return X
