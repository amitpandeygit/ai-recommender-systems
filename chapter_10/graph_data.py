"""Shared data for Chapter 10: Chapter 2's world, its graph and the test contract (section 10.2).

Every test in this chapter runs in Chapter 2's world, with Chapter 9's test
contract: models are fitted on what had arrived by day 23 (day 16 on the
validation week), and each request of days 23 to 30 is answered with the clicks the
server had received, and could credit to the user, before it was served. Follows
obey the same rule: a follow counts only once it has been made.

  load()                - Chapter 9's version of Chapter 2's world (logs, Chapter 8's
                          snapshots, the click log, the hidden truth for the online
                          test), plus tesselo_graph's follows, brands, sellers and
                          knowledge graph
  phase(W, start, S)    - a week's requests, their histories and what each may not show
  click_graph(W, cutoff) - users x items: 1 where a click had arrived, and could be
                          credited to its consenting user, by the cutoff
  social_edges(W, t)    - follows usable in a shared graph at time t: made by then,
                          between consenting users, and neither side private
  own_follows(W, R)     - for each request, the user's own follows made before it
                          (a private user's own follows are used for that user only)
  row_line(...)         - one line of a results table, paired against a reference

Needs Python 3.12 or later and PyTorch. Imported by the other Chapter 10 scripts.
"""
import os
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "2"              # NumPy's linear algebra: the same thread count on every run
import sys
from pathlib import Path

import numpy as np
from scipy import sparse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "chapter_09"))
from seq_data import (DAY, WORLDS, chapter8_rows, environment, exclude_mask, histories, known,  # noqa: E402,F401
                      ndcg, online, outcomes, paired, requests, top10)
import seq_data  # noqa: E402
from tesselo_graph import add_graph  # noqa: E402


def load():
    """Chapter 2's world with everything Chapter 10 needs."""
    W = seq_data.load(WORLDS[0])
    W["graph"] = add_graph(W["cfg"], W["users"], W["items"])
    return W


def phase(W, start, S):
    """Requests of [start, start + 7) days, their histories and what may not be shown."""
    R = requests(W, start, start + 7)
    H, in_visit = histories(W, R)
    return R, H, in_visit, exclude_mask(W, S, R, H)


def click_graph(W, cutoff):
    """Users x items, 1 where a consenting user's click had arrived, and could be
    credited to them, by the cutoff (the user-video edges of the graph)."""
    C = W["clicks"]
    ok = known(C, cutoff)
    n_u, n_i = len(W["users"]["vec"]), len(W["items"]["vec"])
    A = sparse.csr_matrix((np.ones(ok.sum()), (C["user"][ok], C["item"][ok])), shape=(n_u, n_i))
    A.data[:] = 1.0                                              # one edge however often clicked
    return A


def usable(W):
    """Accounts whose edges may appear in a graph shared by everyone: consenting now
    and not private."""
    users = W["users"]
    return users["current_consent"] & ~W["graph"]["private"]


def social_edges(W, t):
    """Follows usable in a shared graph at time t: made before t, between consenting
    users, and neither side private. Returns (user->user, user->creator) pairs."""
    g, ok = W["graph"], usable(W)
    uf, cf = g["user_follows"], g["creator_follows"]
    m = (uf["t"] < t) & ok[uf["src"]] & ok[uf["dst"]]
    mc = (cf["t"] < t) & ok[cf["src"]]
    return (uf["src"][m], uf["dst"][m]), (cf["src"][mc], cf["dst"][mc])


def own_follows(W, R):
    """For each request: the users and creators the user had followed before it was
    served. The user's own list is theirs to use, even if private; a followed account
    counts only if it is consenting and not private, since its clicks would otherwise
    reach someone else."""
    g, ok = W["graph"], usable(W)
    out = []
    for kind in ("user_follows", "creator_follows"):
        e = g[kind]
        order = np.argsort(e["src"], kind="stable")
        src, dst, t = e["src"][order], e["dst"][order], e["t"][order]
        start = np.searchsorted(src, np.arange(len(W["users"]["vec"]) + 1))
        lists = []
        for u, tt in zip(R["user"], R["time"]):
            a, b = start[u], start[u + 1]
            d = dst[a:b][t[a:b] < tt]
            lists.append(d[ok[d]] if kind == "user_follows" else d)
        out.append(lists)
    return out[0], out[1]


def own_followers(W, R):
    """For each request: the accounts that had followed the user before it was served,
    when that follow edge may be in the shared graph (both accounts consenting and not
    private). With them, a user's links at the request are the same kinds of links the
    user had in the trained graph."""
    g, ok = W["graph"], usable(W)
    e = g["user_follows"]
    order = np.argsort(e["dst"], kind="stable")
    src, dst, t = e["src"][order], e["dst"][order], e["t"][order]
    start = np.searchsorted(dst, np.arange(len(W["users"]["vec"]) + 1))
    out = []
    for u, tt in zip(R["user"], R["time"]):
        if not ok[u]:
            out.append(np.zeros(0, dtype=src.dtype))
            continue
        a, b = start[u], start[u + 1]
        s = src[a:b][t[a:b] < tt]
        out.append(s[ok[s]])
    return out


def check_shared_graph(W, edges):
    """The privacy rules, checked on the edges of a graph that every user's recommendations
    are computed from: every follow edge between two consenting, non-private accounts,
    every creator follow from one, every click from a consenting user. Returns the click
    edges and accounts that come from private (consenting) users, whose clicks are
    pooled like every consenting user's."""
    n_u, n_i = len(W["users"]["vec"]), len(W["items"]["vec"])
    rows, cols = np.asarray(edges[0]), np.asarray(edges[1])
    ok, consent, private = usable(W), W["users"]["current_consent"], W["graph"]["private"]
    uu = (rows < n_u) & (cols < n_u)
    uc = (rows < n_u) & (cols >= n_u + n_i)
    ui = (rows < n_u) & (cols >= n_u) & (cols < n_u + n_i)
    assert ok[rows[uu]].all() and ok[cols[uu]].all(), "a follow edge with a private or non-consenting account"
    assert ok[rows[uc]].all(), "a creator follow from a private or non-consenting account"
    assert consent[rows[ui]].all(), "a click from a non-consenting user"
    priv = ui & private[np.minimum(rows, n_u - 1)]
    return int(priv.sum()), len(np.unique(rows[priv]))


def row_line(label, W, R, top, base, extra="", rows=None):
    """One results line: NDCG@10, expected satisfied plays per row, the same for later
    requests, and the paired difference against base (users resampled)."""
    rows = np.arange(len(R["user"])) if rows is None else rows
    sub = {k: (v[rows] if isinstance(v, np.ndarray) else [v[r] for r in rows]) for k, v in R.items()}
    on = online(W, sub, top[rows])
    later = sub["place"] >= 1
    against = ""
    if base is not None:
        d, se = paired(on, base[rows], sub["user"])[:2]
        against = f"{d:+.4f} ({se:.4f})"
    print(f"  {label:<36}{extra:>10}{ndcg(sub, top[rows]):>9.4f}{on.mean():>9.4f}{on[later].mean():>9.4f}"
          f"{against:>18}")
    full = np.zeros(len(R["user"]))
    full[rows] = on
    return full


__all__ = ["DAY", "WORLDS", "chapter8_rows", "environment", "exclude_mask", "histories", "known", "ndcg", "online",
           "outcomes", "paired", "requests", "top10", "load", "phase", "click_graph", "usable", "social_edges",
           "own_follows", "own_followers", "check_shared_graph", "row_line"]
