"""What a graph model costs at a larger scale (section 10.7).

Tesselo's test graph has a few thousand users and videos. This script builds a
synthetic user-item graph closer to production size, 1,000,000 users, 1,000,000
items and about 20,000,000 edges, with skewed activity and popularity, and measures
three ways of computing node vectors with two LightGCN layers (32 numbers per node,
float32) on this workspace's two CPU cores:

  full propagation    - every node, every layer: A_hat times the vectors, twice.
                        This is what LightGCN does at every training step, and what
                        an offline job does to precompute every vector
  neighbour sampling  - GraphSAGE and PinSage: a batch of 1,024 target nodes, each
                        with 10 sampled neighbours, each of those with 10 of its own;
                        only those nodes are read
  request time        - Listing 10.4: with every node's layer vectors precomputed,
                        a user's vector from its 50 most recent items

It reports time, memory and how many nodes each method reads. The graph is random,
not a model of Tesselo: only the costs mean something. Timings depend on the
machine; compare them with each other.

Needs Python 3.12 or later. Run from the repository root (about a minute, about 2 GB of memory):
    python chapter_10/scaling.py
"""
import os
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "2"
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "chapter_09"))
from seq_data import environment  # noqa: E402

N_USERS, N_ITEMS, N_EDGES, DIM, LAYERS = 1_000_000, 1_000_000, 20_000_000, 32, 2
BATCH, FANOUT, HISTORY = 1_024, 10, 50


def random_graph(rng):
    """Edges with skewed activity (users) and Zipf-like popularity (items)."""
    activity = rng.lognormal(0.0, 1.0, N_USERS)
    pop = 1.0 / np.arange(1, N_ITEMS + 1) ** 0.8
    u = rng.choice(N_USERS, N_EDGES, p=activity / activity.sum()).astype(np.int32)
    i = rng.choice(N_ITEMS, N_EDGES, p=pop / pop.sum()).astype(np.int32)
    n = N_USERS + N_ITEMS
    A = sparse.coo_matrix((np.ones(2 * N_EDGES, dtype=np.float32),
                           (np.r_[u, i + N_USERS], np.r_[i + N_USERS, u])), shape=(n, n)).tocsr()
    A.sum_duplicates()
    A.data[:] = 1.0
    deg = np.diff(A.indptr).astype(np.float32)
    inv = 1.0 / np.sqrt(np.maximum(deg, 1))
    A.data *= inv[np.repeat(np.arange(n), np.diff(A.indptr))] * inv[A.indices]   # D^-1/2 A D^-1/2
    return A, deg


def full_propagation(A, E0):
    """Every layer for every node; the final vectors are their mean."""
    out, e = E0.copy(), E0
    for _ in range(LAYERS):
        e = A @ e
        out += e
    return out / (LAYERS + 1)


def sampled_batch(A, E0, targets, rng):
    """Two-hop neighbour sampling for a batch of targets (a mean of FANOUT sampled
    neighbours per hop, as a GraphSAGE layer would aggregate them). Returns the
    vectors and the number of distinct nodes read."""
    def sample(nodes):
        start, size = A.indptr[nodes], np.diff(A.indptr)[nodes]
        pick = start[:, None] + (rng.random((len(nodes), FANOUT)) * np.maximum(size, 1)[:, None]).astype(np.int64)
        return A.indices[np.minimum(pick, len(A.indices) - 1)]
    hop1 = sample(targets)                                       # targets x FANOUT
    hop2 = sample(hop1.ravel()).reshape(len(targets), FANOUT, FANOUT)
    h1 = (E0[hop1] + E0[hop2].mean(axis=2)) / 2                  # each neighbour from its own neighbours
    vec = (E0[targets] + h1.mean(axis=1)) / 2
    read = len(np.unique(np.r_[targets, hop1.ravel(), hop2.ravel()]))
    return vec, read


def request_time(L, deg, histories):
    """Listing 10.4 at scale: each user's vector from its recent items and their
    precomputed layer vectors."""
    rows = np.repeat(np.arange(len(histories)), HISTORY)
    cols = histories.ravel() + N_USERS
    w = 1.0 / np.sqrt(HISTORY * np.maximum(deg[cols], 1))
    M = sparse.csr_matrix((w.astype(np.float32), (rows, cols)), shape=(len(histories), L[0].shape[0]))
    return (M @ L[0] + M @ L[1]) / (LAYERS + 1)                  # layer 0 of a new user is zero


if __name__ == "__main__":
    t0 = time.time()
    print(environment())
    rng = np.random.default_rng(10)
    A, deg = random_graph(rng)
    n = N_USERS + N_ITEMS
    print(f"Graph: {N_USERS:,} users, {N_ITEMS:,} items, {A.nnz // 2:,} distinct user-item edges; "
          f"degree: users mean {deg[:N_USERS].mean():.1f}, max {deg[:N_USERS].max():,.0f}; items median "
          f"{np.median(deg[N_USERS:]):.0f}, max {deg[N_USERS:].max():,.0f}  [{time.time() - t0:.0f} s]")
    print(f"Memory: normalised adjacency {(A.data.nbytes + A.indices.nbytes + A.indptr.nbytes) / 1e9:.2f} GB; "
          f"one layer of vectors {n * DIM * 4 / 1e9:.2f} GB")
    E0 = rng.normal(0, 0.1, (n, DIM)).astype(np.float32)

    t = time.perf_counter()
    final = full_propagation(A, E0)
    t_full = time.perf_counter() - t
    print(f"Full propagation, {LAYERS} layers: {t_full:.1f} s for all {n:,} nodes "
          f"({t_full / n * 1e6:.2f} microseconds per node); reads every edge {LAYERS} times")

    t = time.perf_counter()
    reads = []
    for _ in range(20):
        targets = rng.integers(0, n, BATCH)
        _, read = sampled_batch(A, E0, targets, rng)
        reads.append(read)
    t_batch = (time.perf_counter() - t) / 20
    print(f"Neighbour sampling, fan-out {FANOUT} x {FANOUT}: {t_batch * 1e3:.1f} ms per batch of {BATCH:,} targets "
          f"({t_batch / BATCH * 1e6:.1f} microseconds per target); {np.mean(reads):,.0f} distinct nodes read per "
          f"batch, {np.mean(reads) / BATCH:.0f} per target; all nodes this way: {t_batch / BATCH * n:.0f} s")

    e = E0
    L = [E0]
    for _ in range(LAYERS - 1):
        e = A @ e
        L.append(e)
    hist = rng.integers(0, N_ITEMS, (10_000, HISTORY))
    t = time.perf_counter()
    U = request_time(L, deg, hist)
    t_req = (time.perf_counter() - t) / len(hist)
    item_final = final[N_USERS:]
    t = time.perf_counter()
    for k in range(100):
        s = item_final @ U[k]
        np.argpartition(-s, 100)[:100]
    t_score = (time.perf_counter() - t) / 100
    print(f"Request time: {t_req * 1e6:.1f} microseconds per user vector from {HISTORY} items (10,000 users); "
          f"scoring all {N_ITEMS:,} items exactly and keeping 100: {t_score * 1e3:.1f} ms per request "
          f"(Chapter 6's approximate search makes this step cheap)  [{time.time() - t0:.0f} s]")
