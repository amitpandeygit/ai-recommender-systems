"""What a graph model costs at a larger scale (section 10.7).

Tesselo's test graph has a few thousand users and videos. This script builds a
synthetic user-item graph closer to production size, 1,000,000 users, 1,000,000
items and about 20,000,000 edges, with skewed activity and popularity, and measures
three ways of computing node vectors with two LightGCN layers (32 numbers per node,
float32) on this workspace's two CPU cores:

  full propagation    - every node, every layer: A_hat times the vectors, twice.
                        This is what LightGCN does at every training step, and what
                        an offline job does to precompute every vector
  neighbour sampling  - the same two LightGCN layers, estimated as GraphSAGE and
                        PinSage do it: a batch of 1,024 target nodes, each with 10
                        sampled neighbours, each of those with 10 of its own; only
                        those nodes are read. Each sum over neighbours is estimated
                        from the sample (scaled by degree / 10, so that on average it
                        equals the full sum), and the result is compared with the
                        exact vectors
  request time        - Listing 10.4: with every node's layer vectors precomputed,
                        a user's vector from its 50 most recent items

It reports time, memory, how many nodes each method reads, and how close the sampled
vectors come to the exact ones. The graph is random,
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


def sampled_batch(A, deg, E0, targets, rng):
    """Two LightGCN layers for a batch of targets, estimated from FANOUT neighbours
    sampled (with replacement) per node and hop. A node's layer k + 1 is the sum over
    its d neighbours n of layer k of n / sqrt(d x d_n); the estimate is that sum over
    the sample, times d / FANOUT. A node with no neighbours is never sampled from and
    gets zero, at either hop, even in a graph with no edges. deg: the degrees, computed
    once. Returns the final vectors and the number of distinct nodes read."""
    def sample(nodes, valid):
        """FANOUT neighbours of each valid node with neighbours (-1 elsewhere, weight 0).
        Nodes that are not valid or have no neighbours are never looked up."""
        nb = np.full((len(nodes), FANOUT), -1, dtype=np.int64)
        w = np.zeros((len(nodes), FANOUT), dtype=np.float32)
        live = np.flatnonzero(valid)
        live = live[deg[nodes[live]] > 0]
        n, d = nodes[live], deg[nodes[live]]
        s = A.indices[A.indptr[n][:, None] + (rng.random((len(n), FANOUT)) * d[:, None]).astype(np.int64)]
        nb[live] = s
        w[live] = np.sqrt(d)[:, None] / FANOUT / np.sqrt(np.maximum(deg[s], 1))   # the estimate's weight
        return nb, w
    nb1, w1 = sample(targets, np.ones(len(targets), dtype=bool))                  # targets x FANOUT
    nb2, w2 = sample(nb1.ravel(), nb1.ravel() >= 0)              # the second hop only from real neighbours
    nb2, w2 = nb2.reshape(len(targets), FANOUT, FANOUT), w2.reshape(len(targets), FANOUT, FANOUT)
    E0n1 = E0[np.maximum(nb1, 0)]
    layer1 = (w1[:, :, None] * E0n1).sum(axis=1)                 # layer 1 of each target
    layer1_n = (w2[:, :, :, None] * E0[np.maximum(nb2, 0)]).sum(axis=2)   # layer 1 of each sampled neighbour
    layer2 = (w1[:, :, None] * layer1_n).sum(axis=1)             # layer 2 of each target
    read = len(np.unique(np.r_[targets, nb1[nb1 >= 0], nb2[nb2 >= 0]]))
    return (E0[targets] + layer1 + layer2) / (LAYERS + 1), read


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
          f"one layer of vectors {n * DIM * 4 / 1e9:.2f} GB, so {LAYERS + 1} layers "
          f"{(LAYERS + 1) * n * DIM * 4 / 1e9:.2f} GB")
    E0 = rng.normal(0, 0.1, (n, DIM)).astype(np.float32)

    t = time.perf_counter()
    final = full_propagation(A, E0)
    t_full = time.perf_counter() - t
    print(f"Full propagation, {LAYERS} layers: {t_full:.1f} s for all {n:,} nodes "
          f"({t_full / n * 1e6:.2f} microseconds per node); reads every edge {LAYERS} times")

    t = time.perf_counter()
    reads, cos, tgts = [], [], []
    for _ in range(20):
        targets = rng.integers(0, n, BATCH)
        vec, read = sampled_batch(A, deg, E0, targets, rng)
        reads.append(read)
        prop = (LAYERS + 1) * vec - E0[targets]                  # layers 1 and 2 only: the sampled part
        ex = (LAYERS + 1) * final[targets] - E0[targets]
        cos.append((prop * ex).sum(axis=1) / np.maximum(np.linalg.norm(prop, axis=1) * np.linalg.norm(ex, axis=1),
                                                        1e-12))
        tgts.append(targets)
    t_batch = (time.perf_counter() - t) / 20                     # (the comparison with the exact vectors is cheap)
    cos, tgts = np.concatenate(cos), np.concatenate(tgts)
    iso = deg[tgts] == 0
    print(f"Neighbour sampling, fan-out {FANOUT} x {FANOUT}: {t_batch * 1e3:.1f} ms per batch of {BATCH:,} targets "
          f"({t_batch / BATCH * 1e6:.1f} microseconds per target); {np.mean(reads):,.0f} distinct nodes read per "
          f"batch, {np.mean(reads) / BATCH:.0f} per target; all nodes this way: {t_batch / BATCH * n:.0f} s")
    print(f"  Sampled against exact, layers 1 and 2 of the {(~iso).sum():,} targets with neighbours: cosine "
          f"similarity median {np.median(cos[~iso]):.3f}, 10th percentile {np.percentile(cos[~iso], 10):.3f}; the "
          f"{iso.sum():,} targets with no neighbours get no layers 1 and 2, as in the exact vectors. Isolated "
          f"nodes in the graph: {(deg == 0).sum():,}")

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
