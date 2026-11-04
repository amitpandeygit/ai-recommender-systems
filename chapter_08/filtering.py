"""Listing 8.1 - The filtering layer: hard rules as set operations (section 8.2).

Before anything is ranked, Tesselo removes what must never be shown to this user:
  used          - videos the user has already used (Chapter 3)
  blocked       - every video of a creator the user has blocked
  age rating    - videos rated 18+, for users with a teen account
  licence       - videos not licensed in the user's region
  removed       - videos taken down by moderation
  out of stock  - products that cannot be bought (for the shop and ad slots)
The rule data come from slate_data.rules() (synthetic, its own seed).

Each rule is a set of item ids, and a user's banned set is their union. The same
filter is written three ways and timed: Python sets, NumPy boolean masks, and
Roaring Bitmaps (the pyroaring package). Two jobs are timed, per request:
  candidates - filter a list of 500 candidates (after retrieval)
  catalogue  - the user's whole eligible set (for filtered retrieval, Chapter 6)
at Tesselo's 5,000 items and at Chapter 6's catalogue of 1,000,000 items. All
three ways must give the same answer; the script checks this.

Needs Python 3.12 or later, PyTorch and pyroaring. Run from the repository root (about two minutes):
    python chapter_08/filtering.py
"""
import os
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "2"              # NumPy's linear algebra: the same thread count on every run
import sys
import time
from pathlib import Path

import numpy as np
from pyroaring import BitMap

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from slate_data import POOL, world, rules  # noqa: E402
from ranking_data import candidate_pools  # noqa: E402
from tesselo_catalogue import build_catalogue  # noqa: E402

REQUESTS = 1_000


class RuleSets:
    """The rules as item-id sets, kept in one of three forms: 'set' (Python sets),
    'numpy' (boolean masks over the catalogue, built per request from stored
    index arrays) or 'roaring' (Roaring Bitmaps)."""
    def __init__(self, R, used_ids, n_items, kind):
        self.kind, self.n = kind, n_items
        make = {"set": lambda ids: set(ids.tolist()), "numpy": lambda ids: np.asarray(ids, dtype=np.int32),
                "roaring": lambda ids: BitMap(ids.tolist())}[kind]
        creators = R["creators"]
        order = np.argsort(creators, kind="stable")
        bounds = np.searchsorted(creators[order], np.arange(int(creators.max()) + 2))
        self.by_creator = [make(order[bounds[c]:bounds[c + 1]]) for c in range(len(bounds) - 1)]
        self.used = [make(ids) for ids in used_ids]
        self.adult = make(np.flatnonzero(R["adult"]))
        self.unlicensed = [make(ids) for ids in R["unlicensed"]]
        self.always = make(np.flatnonzero(R["removed"] | R["out_of_stock"]))     # banned for everyone
        self.blocked, self.teen, self.region = R["blocked_creators"], R["teen"], R["region"]
        self.everything = make(np.arange(n_items))

    def banned(self, u):
        """The union of every rule that applies to user u."""
        parts = [self.used[u], self.always, self.unlicensed[self.region[u]]]
        parts += [self.by_creator[c] for c in self.blocked[u]]
        if self.teen[u]:
            parts.append(self.adult)
        if self.kind == "set":
            return set().union(*parts)
        if self.kind == "roaring":
            return BitMap.union(*parts)
        mask = np.zeros(self.n, dtype=bool)
        for p in parts:
            mask[p] = True
        return mask

    def filter(self, u, candidates):
        """The candidates that pass every rule, in their original order."""
        ban = self.banned(u)
        if self.kind == "numpy":
            return candidates[~ban[candidates]]
        return np.array([i for i in candidates.tolist() if i not in ban], dtype=np.int64)

    def eligible(self, u):
        """The user's whole eligible set, everything minus the banned set (its size is returned)."""
        ban = self.banned(u)
        if self.kind == "numpy":
            return int((~ban).sum())
        return len(self.everything - ban)


def nbytes(x):
    """Memory of one rule set: the mask or array buffer, the serialised bitmap, or the
    set object (its hash table only; the integers it points to are extra)."""
    if isinstance(x, np.ndarray):
        return x.nbytes
    if isinstance(x, BitMap):
        return len(x.serialize())
    return sys.getsizeof(x)


def median_ms(fn, users, args):
    times = []
    for u, a in zip(users, args):
        t = time.perf_counter()
        fn(u, a)
        times.append(time.perf_counter() - t)
    return 1000 * float(np.median(times))


def benchmark(label, R, used_ids, n_items, users, candidates):
    print(f"\n{label}: {n_items:,} items; median milliseconds per request over {len(users):,} requests")
    print(f"  {'':<10}{'candidates (500)':>18}{'whole catalogue':>17}   memory of one region's licence rule")
    results = {}
    for kind in ("set", "numpy", "roaring"):
        rs = RuleSets(R, used_ids, n_items, kind)
        c_ms = median_ms(rs.filter, users, candidates)
        e_ms = median_ms(lambda u, _: rs.eligible(u), users, candidates)
        results[kind] = ([rs.filter(u, c) for u, c in zip(users[:200], candidates[:200])],
                         [rs.eligible(u) for u in users[:200]])
        print(f"  {kind:<10}{c_ms:>18.3f}{e_ms:>17.3f}   {nbytes(rs.unlicensed[0]):>10,} bytes")
    same = all(all(np.array_equal(a, b) for a, b in zip(results["set"][0], results[k][0]))
               and results["set"][1] == results[k][1] for k in ("numpy", "roaring"))
    mask_bytes = n_items                                         # one boolean per item
    print(f"  the three give the same filtered lists and eligible counts: {same}; a NumPy boolean mask of the "
          f"catalogue would need {mask_bytes:,} bytes per rule")


if __name__ == "__main__":
    t0 = time.time()
    W = world()
    cfg, users, items = W["cfg"], W["users"], W["items"]
    S = W["snaps"][23]
    n_items = len(items["vec"])
    R = rules(cfg, users, items)
    aud = W["consenting"]

    # ---- what the rules remove at Tesselo's scale, from each consenting user's pool of 500
    print("Section 8.2: the rule data (synthetic, seed cfg.seed + 800)")
    print(f"  users who block at least one creator: {np.mean([len(b) > 0 for b in R['blocked_creators']]):.1%}; "
          f"teen accounts: {R['teen'].mean():.1%}; regions: {R['n_regions']}")
    print(f"  videos rated 18+: {R['adult'].sum():,}; removed by moderation: {R['removed'].sum():,}; "
          f"not licensed in a region: {np.mean([len(x) for x in R['unlicensed']]):.0f} per region on average; "
          f"products out of stock: {R['out_of_stock'].sum():,} of {items['is_product'].sum():,}")
    used_ids = [np.flatnonzero(S["used"][u]) for u in range(len(users["vec"]))]
    pools = candidate_pools(S, users, items, aud, POOL)
    rs = RuleSets(R, used_ids, n_items, "set")
    removed = {"blocked": 0, "age rating": 0, "licence": 0, "removed": 0}
    any_removed = 0
    for u, p in zip(aud, pools):
        p = set(p[p >= 0].tolist())
        hits = {"blocked": set().union(*[rs.by_creator[c] for c in rs.blocked[u]]) & p,
                "age rating": (rs.adult & p) if rs.teen[u] else set(),
                "licence": rs.unlicensed[rs.region[u]] & p,
                "removed": rs.always & p}
        for k, v in hits.items():
            removed[k] += len(v)
        any_removed += len(set().union(*hits.values()))
    print(f"\nTest week, pools of {POOL} for {len(aud):,} consenting users (used videos are already left out of "
          f"Chapter 6's pool): candidates removed per user")
    for k, v in removed.items():
        print(f"  {k:<12}{v / len(aud):>8.2f}")
    print(f"  {'any rule':<12}{any_removed / len(aud):>8.2f}   [{time.time() - t0:.0f} s]")

    # ---- timing: Tesselo's catalogue, then Chapter 6's million items
    rng = np.random.default_rng(cfg.seed + 801)
    req = rng.choice(aud, REQUESTS)
    cand = [p[p >= 0] for p in candidate_pools(S, users, items, req, POOL)]
    benchmark("Tesselo's catalogue", R, used_ids, n_items, req, cand)

    _, _, _, cat, _ = build_catalogue(cfg)
    n_big = len(cat["creator"])
    R_big = rules(cfg, users, items, catalogue=cat, seed_offset=802)
    cand_big = [np.sort(rng.choice(n_big, POOL, replace=False)) for _ in req]   # 500 candidates from retrieval
    benchmark("Chapter 6's catalogue", R_big, used_ids, n_big, req, cand_big)
    print(f"[{time.time() - t0:.0f} s]")
