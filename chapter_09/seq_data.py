"""Shared data for Chapter 9: sequences, test requests and the test contract (section 9.2).

Every script in this chapter runs in two worlds (planted_world.WORLDS): Chapter 2's,
where order carries no information, and the planted one, where each visit has an
intent and tastes drift. In both, the pipeline of Chapters 6 to 8 is rebuilt from
that world's logs with the same code and seeds.

  load(name)         - the world, Chapter 8's weekly snapshots, the click log and the
                       hidden truth for the online test
  click_log(W)       - one row per video click of a consenting, logged-in user (after
                       Chapter 2's identity stitching), in estimated action order
  requests(W, a, b)  - every request served in days [a, b) to a consenting user who was
                       logged in, with its visit, its place in the visit and the videos
                       satisfied in it
  histories(W, R, L) - for each request, the last L videos the user clicked that the
                       server had received before the request was served
  online(W, R, top)  - expected satisfied plays of a row of 10 shown at each request,
                       from the hidden taste at that moment (Chapter 5's formula)
  ndcg(R, top)       - NDCG@10 against the videos satisfied in the request

The test contract (table in section 9.2): models are fitted on what had arrived by
day 23 (day 16 on the validation week); each request of days 23 to 30 is then
answered with everything the server had received before it was served.

Needs Python 3.12 or later and PyTorch. Run from the repository root (about three minutes):
    python chapter_09/seq_data.py
"""
import os
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "2"              # NumPy's linear algebra: the same thread count on every run
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "chapter_08"))
from planted_world import WORLDS, set_world, truth  # noqa: E402
from slate_data import DAY, ROW, world  # noqa: E402
from implicit_scores import estimated_action_time, link_same_visit  # noqa: E402
from evaluate import P_COMPLETE_UNSAT, P_PLAY_SAT  # noqa: E402
from tesselo_data import sigmoid  # noqa: E402
from metrics import ndcg_at_k  # noqa: E402

HITS = ("complete", "like")
MAX_LEN = 50
# Section 9.4's validation choice in each world (highest validation NDCG@10 of next_item.py), and its epochs
CHOICE = {"Chapter 2's world": ("BERT4Rec", "masked, full softmax", 80),
          "planted world": ("GRU4Rec", "full softmax", 30)}


def load(name):
    """One world with everything Chapter 9 needs."""
    set_world(name)
    W = world()
    W["name"] = name
    W["truth"] = truth(W, name)
    W["clicks"] = click_log(W)
    return W


def click_log(W):
    """Video clicks of consenting, logged-in users, deduplicated, with the visit
    stitched to the user who logged in (Chapter 2), in estimated action order."""
    events, users, items = W["events"], W["users"], W["items"]
    order = np.lexsort((events["server_time"], events["event_id"]))
    first = order[np.unique(events["event_id"][order], return_index=True)[1]]
    e = {k: v[first] for k, v in events.items()}
    t = estimated_action_time(e)
    user = link_same_visit(e)
    keep = ((e["event"] == "click") & (e["item_id"] >= 0) & (user >= 0) & e["consent_at_time"]
            & users["current_consent"][np.maximum(user, 0)] & ~items["is_product"][np.maximum(e["item_id"], 0)])
    c = {"user": user[keep], "item": e["item_id"][keep], "session": e["session_id"][keep],
         "t": t[keep], "arrived": e["server_time"][keep]}
    o = np.lexsort((c["t"], c["user"]))
    return {k: v[o] for k, v in c.items()}


def requests(W, start_day, end_day):
    """Requests served in [start_day, end_day) to consenting users while logged in."""
    s = W["served"]
    first = s["position"] == 1
    keep = (first & (s["served_time"] >= start_day * DAY) & (s["served_time"] < end_day * DAY)
            & (s["user_id"] >= 0) & s["consent_at_time"])
    keep &= np.isin(s["user_id"], W["consenting"])
    idx = np.flatnonzero(keep)
    R = {"request": s["request_id"][idx], "user": s["user_id"][idx], "session": s["session_id"][idx],
         "time": s["served_time"][idx],
         "intent": s["_intent"][idx] if "_intent" in s else np.full(len(idx), -1)}
    all_first = np.flatnonzero(first)                            # place of each request in its visit
    sess, tt = s["session_id"][all_first], s["served_time"][all_first]
    o = np.lexsort((tt, sess))
    place = np.empty(len(all_first), dtype=np.int64)
    start = np.r_[0, np.flatnonzero(np.diff(sess[o])) + 1]
    place[o] = np.arange(len(o)) - np.repeat(start, np.diff(np.r_[start, len(o)]))
    req_place = np.full(s["request_id"].max() + 1, -1)
    req_place[s["request_id"][all_first]] = place
    R["place"] = req_place[R["request"]]
    ev = W["events"]
    hit = np.isin(ev["event"], HITS) & ~W["items"]["is_product"][np.maximum(ev["item_id"], 0)] & (ev["item_id"] >= 0)
    pos = {}
    for r, i in zip(ev["request_id"][hit].tolist(), ev["item_id"][hit].tolist()):
        pos.setdefault(r, set()).add(i)
    R["pos"] = [pos.get(int(r), set()) for r in R["request"]]
    return R


def histories(W, R, length=MAX_LEN):
    """For each request: the last `length` clicked videos the server had received
    before it was served (left-padded with -1), and how many of them came in the
    same visit."""
    C = W["clicks"]
    starts = np.searchsorted(C["user"], np.arange(len(W["users"]["vec"]) + 1))
    H = np.full((len(R["user"]), length), -1)
    in_visit = np.zeros(len(R["user"]), dtype=np.int64)
    for k, (u, t, sess) in enumerate(zip(R["user"], R["time"], R["session"])):
        a, b = starts[u], starts[u + 1]
        ok = (C["arrived"][a:b] < t) & (C["t"][a:b] < t)
        items, ses = C["item"][a:b][ok][-length:], C["session"][a:b][ok][-length:]
        if len(items):
            H[k, -len(items):] = items
            in_visit[k] = (ses == sess).sum()
    return H, in_visit


def training_sequences(W, cutoff, length=MAX_LEN):
    """Each user's clicked videos that had arrived by the cutoff, in order (for
    fitting); users with at least two clicks."""
    C = W["clicks"]
    ok = (C["arrived"] < cutoff) & (C["t"] < cutoff)
    seqs, sessions = [], []
    for u in np.unique(C["user"][ok]):
        m = ok & (C["user"] == u)
        if m.sum() >= 2:
            seqs.append(C["item"][m])
            sessions.append(C["session"][m])
    return seqs, sessions


def exclude_mask(W, S, R, H):
    """Videos a row may not show: products, section 8.2's rules (videos used before the
    snapshot, blocked creators, licences, age ratings, removals) and videos already
    clicked in the history."""
    from filtering import RuleSets
    from slate_data import rules
    users, items = W["users"], W["items"]
    n = len(items["vec"])
    rs = RuleSets(rules(W["cfg"], users, items), [np.flatnonzero(S["used"][u]) for u in range(len(users["vec"]))],
                  n, "numpy")
    uu, inv = np.unique(R["user"], return_inverse=True)
    ban = np.stack([rs.banned(u) for u in uu])
    bad = ban[inv]
    bad[:, items["is_product"]] = True
    rows = np.repeat(np.arange(len(H)), H.shape[1])
    flat = H.ravel()
    bad[rows[flat >= 0], flat[flat >= 0]] = True
    return bad


def top10(scores, bad, batch=2_048):
    """Each request's 10 best allowed videos. scores is one row for all requests, or a
    function f(a, b) giving the scores of requests a to b; batches keep memory small."""
    out = np.full((len(bad), ROW), -1)
    for a in range(0, len(bad), batch):
        b = min(a + batch, len(bad))
        sc = scores(a, b) if callable(scores) else scores[None, :]
        s = np.where(bad[a:b], -np.inf, sc)
        top = np.argsort(-s, axis=1, kind="stable")[:, :ROW]
        out[a:b] = np.where(np.isfinite(np.take_along_axis(s, top, axis=1)), top, -1)
    return out


def outcomes(W, R, top):
    """Expected clicks and satisfied plays per slot of each request's row, from the
    hidden taste at that moment (Chapter 5's behaviour model, one view of the row)."""
    cfg, items = W["cfg"], W["items"]
    r = np.arange(1, top.shape[1] + 1)
    keep = cfg.scroll_keep
    on = keep ** (r - 1)
    glimpse = np.where(r >= 2, keep ** np.maximum(r - 2, 0) * (1 - keep), 0.0)
    p_look = (on + cfg.glimpse_look * glimpse) / r ** cfg.look_decay
    eff = W["truth"](R["user"], R["time"], R["intent"]).astype(np.float64)
    filled = top >= 0
    i = np.where(filled, top, 0)
    match = np.einsum("nd,nkd->nk", eff, items["vec"][i].astype(np.float64))
    bait = items["clickbait"][i]
    attract, satisfy = sigmoid(5 * match - 3.5 + bait), sigmoid(5 * match - 2.0 - 2 * bait)
    click = np.where(filled, p_look * attract, 0.0)
    play = np.where(items["is_product"][i], 0.0, click * (satisfy * P_PLAY_SAT + (1 - satisfy) * P_COMPLETE_UNSAT))
    return click, play


def online(W, R, top):
    """Expected satisfied plays per row, one number per request."""
    return outcomes(W, R, top)[1].sum(axis=1)


def ndcg(R, top):
    """Mean NDCG@10 over the requests with a satisfied video."""
    has = np.array([len(p) > 0 for p in R["pos"]])
    return float(np.mean([ndcg_at_k(t.tolist(), {i: 1.0 for i in p}) for t, p, h in zip(top, R["pos"], has) if h]))


def paired(a, b):
    d = np.asarray(a) - np.asarray(b)
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))


def chapter8_rows(W, sp, R):
    """Chapter 8's ranker for each request's user: section 8.2's filters, the pool,
    the pre-ranker's 50 and the heavy ranker's score order (one row per user, from
    the day-23 snapshot, so the same for every request of that user)."""
    sys.path.insert(0, str(HERE.parent / "chapter_08"))
    from filtering import RuleSets
    from honest_test import filtered_kept
    from rankers import score_pools
    from slate_data import rules
    from slates import score_order
    users, items, S = W["users"], W["items"], sp["S"]
    rl = rules(W["cfg"], users, items)
    rs = RuleSets(rl, [np.flatnonzero(S["used"][u]) for u in range(len(users["vec"]))], len(items["vec"]), "numpy")
    uu = np.unique(R["user"])
    sp2 = dict(sp, uu=uu)
    kept = filtered_kept(sp2, W, rs, "uu")
    heavy = score_pools(sp["heavy"], S, users, items, uu, kept)
    top = score_order(kept, np.where(kept >= 0, heavy, -np.inf))
    return top[np.searchsorted(uu, R["user"])], rl


if __name__ == "__main__":
    t0 = time.time()
    for name in WORLDS:
        W = load(name)
        C = W["clicks"]
        per_user = np.bincount(C["user"], minlength=len(W["users"]["vec"]))[W["consenting"]]
        R = requests(W, 23, 30)
        H, in_visit = histories(W, R)
        seqs, _ = training_sequences(W, 23 * DAY)
        print(f"{name}: {len(C['item']):,} video clicks of consenting, logged-in users; per consenting user: "
              f"mean {per_user.mean():.1f}, median {np.median(per_user):.0f}; {len(seqs):,} training sequences "
              f"(arrived by day 23)  [{time.time() - t0:.0f} s]")
        later = R["place"] >= 1
        print(f"  test requests (days 23 to 30): {len(R['user']):,}, of which later in their visit: "
              f"{later.mean():.1%}; with a satisfied video: {np.mean([len(p) > 0 for p in R['pos']]):.1%}; "
              f"history length: mean {(H >= 0).sum(1).mean():.1f}; later requests with a click earlier in "
              f"the visit: {(in_visit[later] > 0).mean():.1%}")
