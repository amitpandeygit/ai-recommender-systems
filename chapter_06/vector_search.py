"""Listings 6.4 to 6.8 - Vector search over one million items (section 6.7).

  1. From MIPS to nearest-neighbour search: add one coordinate to every item
     vector so that the largest dot product becomes the smallest distance.
     Checked on Chapter 4's iALS vectors, whose lengths differ, and on the
     two-tower vectors.
  2. Teaching versions, in NumPy, of the three ideas inside every vector index:
     an inverted file (IVF), product quantisation (PQ) and a navigable graph (HNSW).
  3. The production versions, from Faiss, on Tesselo's one-million-item catalogue
     (tesselo_catalogue.py): exact search, IVF, IVF with PQ, and HNSW.
  4. Filtered search: a hard rule that few items pass.

The main quality measure is recall@10 against exact search, on a fixed sample of
1,000 query users. A returned item counts as correct if its exact score is at
least the exact 10th-best score (so ties with the 10th item count). Next to it
are counts that do not depend on the machine: distances computed per query and
bytes stored per item. Times are measured on one machine, one thread, and are
only an illustration.

Needs Python 3.12 or later. Run from the repository root (about 15 minutes on a two-core machine):
    python chapter_06/vector_search.py
"""
import sys
import time
from pathlib import Path

import faiss
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from tesselo_catalogue import build_catalogue  # noqa: E402
from two_tower import MAIN, embed_all, make_data, train_two_tower  # noqa: E402
from tesselo_shop import DAY, Config  # noqa: E402
from similarity_recs import item_features, user_item_scores  # noqa: E402
from implicit_mf import ials  # noqa: E402

faiss.omp_set_num_threads(1)               # one thread: deterministic builds, comparable timings
K = 10


# ---------------------------------------------------------------- 1. MIPS as nearest-neighbour search
def mips_to_nn(items, queries):
    """Bachrach and colleagues (2014): with M the largest item length, append
    sqrt(M^2 - |x|^2) to every item and 0 to every query. Then
    |q - x|^2 = |q|^2 + M^2 - 2 q.x, so the smallest distance is the largest dot product."""
    norms2 = (items ** 2).sum(axis=1)
    M2 = norms2.max()
    extra = np.sqrt(np.maximum(M2 - norms2, 0.0))
    return (np.hstack([items, extra[:, None]]).astype(np.float32),
            np.hstack([queries, np.zeros((len(queries), 1))]).astype(np.float32))


def exact_top(queries, items, k=K, chunk=20_000, largest=True):
    """Exact top-k by dot product (largest=True) or by squared distance (False)."""
    best_s = np.full((len(queries), k), -np.inf)
    best_i = np.zeros((len(queries), k), dtype=np.int64)
    qn = (queries ** 2).sum(axis=1)[:, None]
    for s in range(0, len(items), chunk):
        x = items[s:s + chunk]
        sc = queries @ x.T
        if not largest:
            sc = -(qn - 2 * sc + (x ** 2).sum(axis=1)[None, :])
        both_s = np.hstack([best_s, sc])
        both_i = np.hstack([best_i, np.arange(s, s + len(x))[None, :].repeat(len(queries), axis=0)])
        o = np.argsort(-both_s, axis=1, kind="stable")[:, :k]
        best_s, best_i = np.take_along_axis(both_s, o, axis=1), np.take_along_axis(both_i, o, axis=1)
    return best_i, best_s


def recall_vs_exact(found, queries, items, exact_scores, tol=1e-5):
    """Share of the k returned items whose exact dot product is at least the exact
    k-th best score (ties with the k-th item count as correct)."""
    hits = 0
    for q, f, ex in zip(queries, found, exact_scores):
        f = f[f >= 0]
        hits += int(((items[f] @ q) >= ex[-1] - tol).sum())
    return hits / (len(queries) * exact_scores.shape[1])


# ---------------------------------------------------------------- 2. teaching versions
def kmeans(x, k, iters=10, seed=0):
    """Lloyd's algorithm with random starting centres, by matrix products."""
    rng = np.random.default_rng(seed)
    c = x[rng.choice(len(x), k, replace=False)].copy()
    for _ in range(iters):
        a = assign(x, c)
        for j in range(k):
            m = a == j
            if m.any():
                c[j] = x[m].mean(axis=0)
    return c


def assign(x, c, chunk=100_000):
    out = np.empty(len(x), dtype=np.int64)
    cn = (c ** 2).sum(axis=1)
    for s in range(0, len(x), chunk):
        out[s:s + chunk] = np.argmin(cn[None, :] - 2 * x[s:s + chunk] @ c.T, axis=1)
    return out


class IVF:
    """Inverted file: cluster the items, keep one list of item ids per cluster, and at
    query time scan only the nprobe lists whose centres are nearest the query."""

    def __init__(self, items, n_lists, seed=0):
        sample = items[np.random.default_rng(seed).choice(len(items), min(len(items), 50 * n_lists),
                                                          replace=False)]
        self.items, self.centres = items, kmeans(sample, n_lists, seed=seed)
        a = assign(items, self.centres)
        order = np.argsort(a, kind="stable")
        self.lists = np.split(order, np.cumsum(np.bincount(a, minlength=n_lists))[:-1])

    def search(self, q, k=K, nprobe=8):
        d_centre = ((self.centres - q) ** 2).sum(axis=1)               # n_lists distances
        cand = np.concatenate([self.lists[j] for j in np.argsort(d_centre)[:nprobe]])
        d = ((self.items[cand] - q) ** 2).sum(axis=1)                  # one distance per candidate
        top = cand[np.argsort(d, kind="stable")[:k]]
        return np.pad(top, (0, k - len(top)), constant_values=-1), len(self.centres) + len(cand)


class PQ:
    """Product quantisation: cut each vector into m pieces and replace each piece by the
    nearest of 256 centres learned for that piece, so an item is stored in m bytes."""

    def __init__(self, items, m, seed=0):
        self.m, self.sub = m, items.shape[1] // m
        sample = items[np.random.default_rng(seed).choice(len(items), min(len(items), 50_000), replace=False)]
        self.books = [kmeans(sample[:, j * self.sub:(j + 1) * self.sub], 256, seed=seed) for j in range(m)]
        self.codes = np.stack([assign(items[:, j * self.sub:(j + 1) * self.sub], self.books[j])
                               for j in range(m)], axis=1).astype(np.uint8)

    def search(self, q, k=K):
        """Asymmetric distance: the query stays exact; one small table per piece holds the
        distance from the query's piece to all 256 centres, and an item's distance is the
        sum of m table look-ups."""
        table = np.stack([((self.books[j] - q[j * self.sub:(j + 1) * self.sub]) ** 2).sum(axis=1)
                          for j in range(self.m)])                     # m x 256
        d = np.zeros(len(self.codes))
        for j in range(self.m):
            d += table[j][self.codes[:, j]]                            # m look-ups per item
        return np.argsort(d, kind="stable")[:k].copy()                 # copy: do not keep the full order


class HNSW:
    """Hierarchical navigable small world graph (Malkov and Yashunin), kept small and
    plain: each item gets a top level drawn from an exponential distribution, links to
    its M nearest found items on every level it lives on, and a search walks greedily
    down from the top level, keeping the ef best items found on the bottom level."""

    def __init__(self, items, M=16, ef_construction=64, seed=0):
        self.x, self.M, self.ef_c = items, M, ef_construction
        rng = np.random.default_rng(seed)
        self.level = np.floor(-np.log(rng.random(len(items))) / np.log(M)).astype(int)
        self.links = [dict() for _ in range(self.level.max() + 1)]      # level -> item -> neighbours
        self.entry, self.count = None, 0
        for i in range(len(items)):
            self.insert(i)

    def dist(self, q, ids):
        self.count += len(ids)
        return ((self.x[ids] - q) ** 2).sum(axis=1)

    def search_level(self, q, entries, ef, lev):
        """Best-first search on one level: the ef nearest items reachable from entries."""
        seen = set(entries)
        d = self.dist(q, list(entries))
        cand = sorted(zip(d.tolist(), entries))                         # to expand, nearest first
        best = list(cand)                                               # found so far, nearest first
        while cand:
            dc, c = cand.pop(0)
            if dc > best[-1][0] and len(best) >= ef:
                break                                                   # nothing nearer can follow
            new = [n for n in self.links[lev].get(c, []) if n not in seen]
            seen.update(new)
            if new:
                for dn, n in zip(self.dist(q, new).tolist(), new):
                    if len(best) < ef or dn < best[-1][0]:
                        cand.append((dn, n))
                        best.append((dn, n))
                cand.sort()
                best = sorted(best)[:ef]
        return best

    def insert(self, i):
        q, top = self.x[i], self.level[i]
        if self.entry is None:
            self.entry = i
            for lev in range(top + 1):
                self.links[lev][i] = []
            return
        ep = [self.entry]
        for lev in range(self.level[self.entry], top, -1):              # greedy above the item's levels
            ep = [self.search_level(q, ep, 1, lev)[0][1]]
        for lev in range(min(top, self.level[self.entry]), -1, -1):
            found = self.search_level(q, ep, self.ef_c, lev)
            nbrs = [n for _, n in found[:self.M]]
            self.links[lev][i] = nbrs
            for n in nbrs:                                              # links go both ways
                self.links[lev][n].append(i)
                if len(self.links[lev][n]) > 2 * self.M:                # prune to the nearest 2M
                    ids = self.links[lev][n]
                    keep = np.argsort(((self.x[ids] - self.x[n]) ** 2).sum(axis=1), kind="stable")
                    self.links[lev][n] = [ids[j] for j in keep[:2 * self.M]]
            ep = [n for _, n in found]
        for lev in range(self.level[self.entry] + 1, top + 1):
            self.links[lev][i] = []
        if top > self.level[self.entry]:
            self.entry = i

    def search(self, q, k=K, ef=64):
        self.count = 0
        ep = [self.entry]
        for lev in range(self.level[self.entry], 0, -1):
            ep = [self.search_level(q, ep, 1, lev)[0][1]]
        found = self.search_level(q, ep, max(ef, k), 0)
        return np.array([n for _, n in found[:k]]), self.count


# ---------------------------------------------------------------- 3. Faiss
def faiss_run(index, queries, k=K, kind=None):
    """Search all queries one by one (as a server would), timing them and counting the
    distances Faiss computed."""
    stats = faiss.cvar.hnsw_stats if kind == "hnsw" else faiss.cvar.indexIVF_stats
    stats.reset()
    out = np.zeros((len(queries), k), dtype=np.int64)
    t = time.perf_counter()
    for r in range(len(queries)):
        out[r] = index.search(queries[r:r + 1], k)[1][0]
    ms = (time.perf_counter() - t) * 1000 / len(queries)
    return out, ms, stats.ndis / len(queries)


def index_bytes(index):
    return len(faiss.serialize_index(index))


if __name__ == "__main__":
    t0 = time.time()
    cfg = Config()
    users, items, events, cat, _ = build_catalogue(cfg)
    n_users, n_items = cfg.num_users, len(items["vec"])
    print(f"Catalogue: {len(cat['vec']):,} items, {cat['n_original']:,} with history  [{time.time() - t0:.0f} s]")

    # ---- the query users and the two-tower model of section 6.6 (MAIN settings, trained to day 23)
    weight, used = user_item_scores(events, users, 23 * DAY, n_users, n_items)
    feats = item_features(items, n_items)
    data = make_data(weight, items, users, feats, int(items["creator"].max()) + 1)
    pu, pi = np.nonzero(weight > 0)
    model = train_two_tower(data, pu, pi, **MAIN)
    U, V5 = embed_all(model, data)
    hist = np.flatnonzero((weight > 0).any(axis=1))
    qusers = np.sort(np.random.default_rng(cfg.seed + 610).choice(hist, 1_000, replace=False))
    Q = U[qusers]
    with torch.no_grad():                                             # new items: content only
        V = np.zeros((len(cat["vec"]), V5.shape[1]))
        V[:n_items] = V5
        f = torch.as_tensor(cat["feats"])
        c = torch.as_tensor(cat["creator"])
        for s in range(n_items, len(V), 100_000):
            e = min(s + 100_000, len(V))
            V[s:e] = model.item_from_inputs(torch.zeros(e - s, model.item_id.embedding_dim),
                                            f[s:e], c[s:e]).numpy()
    del f, c, cat["feats"], events, data, model, feats, U, V5       # free memory for the indexes
    print(f"Item vectors: {V.shape[0]:,} x {V.shape[1]}; query users: {len(Q):,}  [{time.time() - t0:.0f} s]")

    # ---- 1. MIPS as nearest-neighbour search, on Chapter 4's iALS vectors (5,000 items)
    X4, Y4 = ials(weight, 16, 1.0, 10.0)
    q4 = X4[qusers]
    _, ex4 = exact_top(q4, Y4)
    norms = np.linalg.norm(Y4, axis=1)
    print(f"\n1. MIPS on Chapter 4's iALS vectors: item lengths from {norms[norms > 0].min():.3f} to "
          f"{norms.max():.3f} (median {np.median(norms[norms > 0]):.3f})")
    Yt, qt = mips_to_nn(Y4, q4)
    for label, found in (("cosine (directions only)", exact_top(q4, Y4 / np.maximum(norms, 1e-12)[:, None])[0]),
                         ("nearest by distance, as they are", exact_top(q4, Y4, largest=False)[0]),
                         ("nearest by distance, one coordinate added", exact_top(qt, Yt, largest=False)[0])):
        print(f"  top 10 by {label:<42} recall@10 against the largest dot products: "
              f"{recall_vs_exact(found, q4, Y4, ex4):.3f}")

    # ---- where the exact top 10 comes from in the full catalogue
    full_i, full_s = exact_top(Q, V)
    lens = np.linalg.norm(V, axis=1)
    print(f"\nTwo-tower vectors: {V.shape[1]} numbers (a unit vector and the item bias), lengths from "
          f"{lens.min():.3f} to {lens.max():.3f}")
    print(f"  share of the exact top 10 over all {len(V):,} items that are the {n_items:,} items with history: "
          f"{np.mean(full_i < n_items):.1%}")
    print(f"  so the {n_items:,} items with history are searched exactly, and the index holds the other "
          f"{len(V) - n_items:,}")

    # ---- the items without history, made searchable by distance
    warm, cold = V[:n_items], V[n_items:]
    Vt, Qt = mips_to_nn(cold, Q)
    pad = (-Vt.shape[1]) % 4                                       # PQ cuts vectors into pieces of 4
    Vt = np.hstack([Vt, np.zeros((len(Vt), pad), np.float32)])
    Qt = np.hstack([Qt, np.zeros((len(Qt), pad), np.float32)])
    t = time.perf_counter()
    exact_i, exact_s = exact_top(Q, cold)
    t_exact = (time.perf_counter() - t) * 1000 / len(Q)
    same = np.mean([set(a) == set(b) for a, b in zip(exact_i, exact_top(Qt, Vt, largest=False)[0])])
    print(f"  after the added coordinate and padding, {Vt.shape[1]} numbers per item")
    print(f"  exact top 10 by dot product equals exact top 10 by distance after the change: {same:.1%} of queries")
    print(f"  exact search in NumPy, all {len(cold):,} items, batched: {t_exact:.1f} ms per query")
    rec = lambda found: recall_vs_exact(found, Q, cold, exact_s)               # noqa: E731

    # ---- 2. teaching versions
    print(f"\n2. Teaching versions (NumPy), {len(cold):,} items, {len(Q):,} queries")
    t = time.perf_counter()
    ivf = IVF(Vt, 1024)
    print(f"  IVF with 1,024 lists built in {time.perf_counter() - t:.0f} s; list sizes "
          f"{min(len(x) for x in ivf.lists):,} to {max(len(x) for x in ivf.lists):,}")
    for nprobe in (1, 8, 32):
        res = [ivf.search(q, nprobe=nprobe) for q in Qt]
        print(f"    nprobe {nprobe:>2}: recall@10 {rec(np.array([r[0] for r in res])):.3f}, "
              f"distances per query {np.mean([r[1] for r in res]):,.0f}")
    t = time.perf_counter()
    pq = PQ(Vt, Vt.shape[1] // 4)
    print(f"  PQ with {pq.m} pieces of 4 numbers ({pq.m} bytes per item) built in {time.perf_counter() - t:.0f} s")
    found_pq = np.array([pq.search(q, k=100) for q in Qt])
    print(f"    PQ alone, top 10: recall@10 {rec(found_pq[:, :10]):.3f}")
    rerank = np.array([f[np.argsort(-(cold[f] @ q), kind='stable')[:K]] for f, q in zip(found_pq, Q)])
    print(f"    PQ top 100, then the 100 rescored exactly: recall@10 {rec(rerank):.3f}")
    n_small = 10_000
    t = time.perf_counter()
    hn = HNSW(Vt[:n_small], M=16, ef_construction=64)
    build_s = time.perf_counter() - t
    _, small_s = exact_top(Q, cold[:n_small])
    print(f"  HNSW on the first {n_small:,} items (M 16, ef_construction 64) built in {build_s:.0f} s; "
          f"{hn.level.max() + 1} levels")
    for ef in (16, 64):
        res = [hn.search(q, ef=ef) for q in Qt]
        print(f"    ef {ef:>3}: recall@10 {recall_vs_exact(np.array([r[0] for r in res]), Q, cold[:n_small], small_s):.3f}, "
              f"distances per query {np.mean([r[1] for r in res]):,.0f} (exact search: {n_small:,})")
    fs = faiss.IndexHNSWFlat(Vt.shape[1], 16)
    fs.hnsw.efConstruction = 64
    fs.add(Vt[:n_small])
    for ef in (16, 64):
        fs.hnsw.efSearch = ef
        found, _, nd = faiss_run(fs, Qt, kind="hnsw")
        print(f"    Faiss HNSW, same items and settings, ef {ef:>3}: recall@10 "
              f"{recall_vs_exact(found, Q, cold[:n_small], small_s):.3f}, distances per query {nd:,.0f}")
    print(f"  [{time.time() - t0:.0f} s]")

    # ---- 3. Faiss on one million items
    d = Vt.shape[1]
    sample = Vt[np.random.default_rng(cfg.seed + 611).choice(len(Vt), 100_000, replace=False)]
    print(f"\n3. Faiss, {len(Vt):,} items without history, {len(Q):,} queries, one thread")
    print(f"  {'index':<40}{'recall@10':>10}{'distances':>12}{'ms/query':>10}{'bytes/item':>12}")
    flat = faiss.IndexFlatL2(d)
    flat.add(Vt)
    found, ms, _ = faiss_run(flat, Qt)
    print(f"  {'exact (flat)':<40}{rec(found):>10.3f}{len(Vt):>12,}{ms:>10.2f}{index_bytes(flat) / len(Vt):>12.0f}")
    t = time.perf_counter()
    ivf_f = faiss.IndexIVFFlat(faiss.IndexFlatL2(d), d, 1024)
    ivf_f.train(sample)
    ivf_f.add(Vt)
    ivf_build = time.perf_counter() - t
    for nprobe in (1, 4, 16, 64):
        ivf_f.nprobe = nprobe
        found, ms, nd = faiss_run(ivf_f, Qt)
        print(f"  {f'IVF, 1,024 lists, nprobe {nprobe}':<40}{rec(found):>10.3f}{nd + 1024:>12,.0f}{ms:>10.2f}"
              f"{index_bytes(ivf_f) / len(Vt):>12.0f}")
    t = time.perf_counter()
    ivfpq = faiss.IndexIVFPQ(faiss.IndexFlatL2(d), d, 1024, d // 4, 8)
    ivfpq.train(sample)
    ivfpq.add(Vt)
    pq_build = time.perf_counter() - t
    for nprobe in (16, 64):
        ivfpq.nprobe = nprobe
        found, ms, nd = faiss_run(ivfpq, Qt)
        print(f"  {f'IVF-PQ, {d // 4} bytes, nprobe {nprobe}':<40}{rec(found):>10.3f}{nd + 1024:>12,.0f}{ms:>10.2f}"
              f"{index_bytes(ivfpq) / len(Vt):>12.0f}")
    base = faiss.IndexIVFPQ(faiss.IndexFlatL2(d), d, 1024, d // 4, 8)   # the same index, with an exact copy
    base.train(sample)
    refine = faiss.IndexRefineFlat(base)                           # keeps the full vectors for rescoring
    refine.add(Vt)
    refine.k_factor = 10
    base.nprobe = 16
    found, ms, nd = faiss_run(refine, Qt)
    print(f"  {'IVF-PQ nprobe 16, top 100 rescored':<40}{rec(found):>10.3f}{nd + 1024 + 100:>12,.0f}{ms:>10.2f}"
          f"{index_bytes(refine) / len(Vt):>12.0f}")
    t = time.perf_counter()
    hnsw = faiss.IndexHNSWFlat(d, 32)
    hnsw.hnsw.efConstruction = 64
    hnsw.add(Vt)
    hnsw_build = time.perf_counter() - t
    for ef in (16, 32, 64, 128, 256):
        hnsw.hnsw.efSearch = ef
        found, ms, nd = faiss_run(hnsw, Qt, kind="hnsw")
        print(f"  {f'HNSW, M 32, efSearch {ef}':<40}{rec(found):>10.3f}{nd:>12,.0f}{ms:>10.2f}"
              f"{index_bytes(hnsw) / len(Vt):>12.0f}")
    print(f"  build times, one thread: IVF {ivf_build:.0f} s, IVF-PQ {pq_build:.0f} s, HNSW {hnsw_build:.0f} s  "
          f"[{time.time() - t0:.0f} s]")
    hnsw.hnsw.efSearch = 64                                        # the full catalogue: exact warm + HNSW cold
    found, _, _ = faiss_run(hnsw, Qt, kind="hnsw")
    both = np.hstack([exact_top(Q, warm)[0], np.where(found >= 0, found + n_items, 0)])
    sc = np.where(np.hstack([np.ones_like(found), found]) >= 0, np.einsum("qd,qkd->qk", Q, V[both]), -np.inf)
    merged = np.take_along_axis(both, np.argsort(-sc, axis=1, kind="stable")[:, :K], axis=1)
    print(f"  all {len(V):,} items: exact search of the {n_items:,} with history plus HNSW (efSearch 64) on the rest, "
          f"merged by score: recall@10 {recall_vs_exact(merged, Q, V, full_s):.3f}")

    # ---- 4. filtered search: a "New this month" row allows only items uploaded in the last 30 days
    allowed = cat["upload_day"][n_items:] >= 335
    ids = np.flatnonzero(allowed)
    _, fexact_s = exact_top(Q, cold[ids])
    frec = lambda found: recall_vs_exact(found, Q, cold, fexact_s)            # noqa: E731
    print(f"\n4. Filtered search: {allowed.mean():.1%} of items pass the rule ({len(ids):,} items)")
    hnsw.hnsw.efSearch = 64
    for k_ask in (10, 100, 1000):
        found, ms, nd = faiss_run(hnsw, Qt, k=k_ask, kind="hnsw")
        kept = [f[allowed[f]][:K] for f in found]
        filled = np.mean([len(x) for x in kept]) / K
        padded = np.array([np.pad(x, (0, K - len(x)), constant_values=-1) for x in kept])
        print(f"  HNSW, ask for {k_ask:>4}, then drop what fails the rule: slots filled {filled:.1%}, "
              f"recall@10 {frec(padded):.3f}, {ms:.2f} ms/query")
    params = faiss.SearchParametersHNSW(sel=faiss.IDSelectorBatch(ids), efSearch=64)
    t = time.perf_counter()
    found = np.vstack([hnsw.search(Qt[r:r + 1], K, params=params)[1] for r in range(len(Qt))])
    ms = (time.perf_counter() - t) * 1000 / len(Qt)
    print(f"  HNSW with the rule inside the search (IDSelector): slots filled {np.mean(found >= 0):.1%}, "
          f"recall@10 {frec(found):.3f}, {ms:.2f} ms/query")
    print(f"[{time.time() - t0:.0f} s]")
