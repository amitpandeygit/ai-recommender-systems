"""Shared data for Chapter 9: sequences, test requests and the test contract (section 9.2).

Every script in this chapter runs in two worlds (planted_world.WORLDS): Chapter 2's,
where each user's taste is fixed, so order carries little information, and the
planted one, where each visit has an intent and tastes drift. In both, the pipeline of Chapters 6 to 8 is rebuilt from
that world's logs with the same code and seeds.

  load(name)         - the world, Chapter 8's weekly snapshots, the click log and the
                       hidden truth for the online test
  click_log(W)       - one row per video click of a consenting user, with the moment
                       the server could credit it to the user (Chapter 2's identity
                       rule, replayed as logins arrive: link_intervals), in each
                       user's order (the phone's own counter on one phone: device_order)
  requests(W, a, b)  - every request served in days [a, b) to a consenting user who was
                       logged in, with its visit, its place in the visit and the videos
                       satisfied in it
  histories(W, R, L) - for each request, the last L videos the user clicked that the
                       server had received, and could credit to the user, before the
                       request was served
  exclude_mask, top10, eligible_kept - what each request may show, and rows that obey it
  paired(a, b, groups) - paired differences with a bootstrap over whole users
  online(W, R, top)  - expected satisfied plays of a row of 10 shown at each request,
                       from the hidden taste at that moment (Chapter 5's formula)
  ndcg(R, top)       - NDCG@10 against the videos satisfied in the request

The test contract (table in section 9.2): models are fitted on what had arrived by
day 23 (day 16 on the validation week); each request of days 23 to 30 is then
answered with everything the server had received, and could credit to the user,
before it was served (Figure 9.2). contract_checks() tests both rules on small cases.

Needs Python 3.12 or later and PyTorch. Run from the repository root (about two minutes;
add 0 or 1 to run one world only):
    python chapter_09/seq_data.py
"""
import os
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "2"              # NumPy's linear algebra: the same thread count on every run
import subprocess
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
CHOICE = {"Chapter 2's world": ("BERT4Rec", "masked, full softmax", 90),
          "planted world": ("GRU4Rec", "full softmax", 30)}


def load(name):
    """One world with everything Chapter 9 needs."""
    set_world(name)
    W = world()
    W["name"] = name
    W["truth"] = truth(W, name)
    W["clicks"] = click_log(W)
    return W


def environment():
    """The machine and library versions of a run, printed at the top of every output."""
    import platform
    import torch
    return (f"Environment: Python {platform.python_version()}, NumPy {np.__version__}, PyTorch "
            f"{torch.__version__.split('+')[0]}, {platform.system()} {platform.machine()}, "
            f"{os.cpu_count()} CPUs")


def link_intervals(e):
    """When each event can be credited to a user, replaying Chapter 2's
    link_same_visit rule as the logins of its visit arrive at the server.

    A logged-in event belongs to its user from the moment it arrives. A logged-out
    event is linked only once a later login of exactly one account, on the same app
    install, has arrived; a login of a second account, or an earlier login on that
    install, ends the link when it arrives (Chapter 2's rule, applied to what the
    server has). Returns (user, linked_from, linked_until); user is -1 if never linked.
    After the last login has arrived, the result is exactly link_same_visit(e)."""
    n = len(e["event"])
    user = e["user_id"].copy()
    start = np.where(user >= 0, e["server_time"], np.inf).astype(np.float64)
    until = np.full(n, np.inf)
    logins = {}
    for j in np.flatnonzero(e["event"] == "login"):
        logins.setdefault(e["session_id"][j], []).append(j)
    for js in logins.values():
        js.sort(key=lambda j: (e["server_time"][j], e["event_id"][j]))   # in order of arrival
    for j in np.flatnonzero(e["user_id"] < 0):
        js = logins.get(e["session_id"][j])
        if not js:
            continue
        linked = -1
        for k in range(1, len(js) + 1):                             # after the k-th login arrives
            found = js[:k]
            accounts = {int(e["user_id"][q]) for q in found}
            here = [e["seq"][q] for q in found if e["anonymous_id"][q] == e["anonymous_id"][j]]
            ok = (len(accounts) == 1 and not any(q < e["seq"][j] for q in here)
                  and any(q > e["seq"][j] for q in here))
            now = e["server_time"][js[k - 1]]
            if ok and linked < 0 and np.isinf(until[j]):
                linked = accounts.pop()
                user[j], start[j] = linked, max(now, e["server_time"][j])
            elif not ok and linked >= 0:
                until[j] = now                                       # the link ends here, for good
                linked = -1
        if user[j] >= 0 and np.isfinite(until[j]) and until[j] <= start[j]:
            user[j] = -1
    return user, start, until


def device_order(user, install, seq, t, event_id):
    """Each user's events in order: on one phone (app
    install) by its counter seq, which the phone keeps exactly; across phones by
    estimated action time, ties broken by event id. Returns the order of the rows."""
    o = np.lexsort((event_id, t, user))                              # by time within each user
    out = o.copy()
    key = np.stack([user[o], install[o]], axis=1)
    _, g = np.unique(key, axis=0, return_inverse=True)
    g = g.ravel()
    for grp in np.unique(g):
        slots = np.flatnonzero(g == grp)                             # where this phone's events sit
        rows = o[slots]
        out[slots] = rows[np.lexsort((event_id[rows], seq[rows]))]   # filled in the phone's own order
    return out


def click_log(W):
    """Video clicks of consenting users, deduplicated, each credited to its user from
    the moment the server could know it (link_intervals), in each user's order
    (device_order). Fields: user, item, session, install, seq, t (estimated action
    time), known (when the server could credit the click to the user: its arrival,
    or the arrival of the login that linked it) and until (when a later login ended
    the link; inf if never)."""
    events, users, items = W["events"], W["users"], W["items"]
    order = np.lexsort((events["server_time"], events["event_id"]))
    first = order[np.unique(events["event_id"][order], return_index=True)[1]]
    e = {k: v[first] for k, v in events.items()}
    t = estimated_action_time(e)
    user, start, until = link_intervals(e)
    keep = ((e["event"] == "click") & (e["item_id"] >= 0) & (user >= 0) & e["consent_at_time"]
            & users["current_consent"][np.maximum(user, 0)] & ~items["is_product"][np.maximum(e["item_id"], 0)])
    c = {"user": user[keep], "item": e["item_id"][keep], "session": e["session_id"][keep],
         "install": e["anonymous_id"][keep], "seq": e["seq"][keep], "t": t[keep],
         "known": np.maximum(start[keep], e["server_time"][keep]), "until": until[keep],
         "arrival": e["server_time"][keep], "event_id": e["event_id"][keep]}
    o = device_order(c["user"], c["install"], c["seq"], c["t"], c["event_id"])
    return {k: v[o] for k, v in c.items()}


def known(C, t):
    """The clicks the server could use at time t: credited to their user by then, the
    link not yet ended, and done before t."""
    return (C["known"] < t) & (C["until"] > t) & (C["t"] < t)


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
        ok = (C["known"][a:b] < t) & (C["until"][a:b] > t) & (C["t"][a:b] < t)
        items, ses = C["item"][a:b][ok][-length:], C["session"][a:b][ok][-length:]
        if len(items):
            H[k, -len(items):] = items
            in_visit[k] = (ses == sess).sum()
    return H, in_visit


def training_sequences(W, cutoff, length=MAX_LEN):
    """Each user's clicked videos that had arrived by the cutoff, in order (for
    fitting); users with at least two clicks."""
    C = W["clicks"]
    ok = known(C, cutoff)
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


def top10(scores, bad, batch=2_048, tie=None):
    """Each request's 10 best allowed videos. scores is one row for all requests, or a
    function f(a, b) giving the scores of requests a to b; batches keep memory small.
    Ties in score (above all, a row with no evidence, where every score is 0) go to
    the higher value of tie, one row for all requests or a function like scores
    (popularity, so that no method falls back on the lowest video ids)."""
    out = np.full((len(bad), ROW), -1)
    for a in range(0, len(bad), batch):
        b = min(a + batch, len(bad))
        sc = scores(a, b) if callable(scores) else scores[None, :]
        s = np.where(bad[a:b], -np.inf, sc)
        if tie is None:
            top = np.argsort(-s, axis=1, kind="stable")[:, :ROW]
        else:
            tb = tie(a, b) if callable(tie) else tie[None, :]
            top = np.lexsort((np.broadcast_to(-tb, s.shape), -s), axis=1)[:, :ROW]
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


def paired(a, b, groups, n_boot=1_000, seed=0):
    """Mean of a - b over requests, with a paired bootstrap that resamples whole groups
    (users, or visits for requests with no profile), since one user's requests share
    a taste, a history and often a row. Returns the mean, its standard error and a
    95% interval."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    _, g = np.unique(np.asarray(groups), return_inverse=True)
    g = g.ravel()
    sums, counts = np.bincount(g, weights=d), np.bincount(g).astype(np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for k in range(0, n_boot, 100):                               # in blocks, to keep memory small
        pick = rng.integers(0, len(sums), size=(min(100, n_boot - k), len(sums)))
        means[k:k + len(pick)] = sums[pick].sum(axis=1) / counts[pick].sum(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(d.mean()), float(means.std(ddof=1)), float(lo), float(hi)


def eligible_kept(ranked, inv, bad, k=50):
    """Chapter 8's pre-ranked pool, cut at request time: for each request, the first k
    videos of its user's ranked pool (ranked: users x pool, best first, -1 padded) that
    the request may show (bad: requests x videos). A video that became forbidden since
    the snapshot, such as one clicked earlier this week, makes room for the next."""
    out = np.full((len(inv), k), -1)
    for a in range(0, len(inv), 2_048):
        r = ranked[inv[a:a + 2_048]]
        ok = (r >= 0) & ~np.take_along_axis(bad[a:a + 2_048], np.maximum(r, 0), axis=1)
        first = np.argsort(~ok, axis=1, kind="stable")[:, :k]       # allowed ones, in pool order
        pick = np.take_along_axis(r, first, axis=1)
        out[a:a + 2_048] = np.where(np.take_along_axis(ok, first, axis=1), pick, -1)
    return out


def ranked_pools(sp, W, rs, rows):
    """Chapter 8's pool of 500 for each user, after section 8.2's filters, in the
    pre-ranker's order (Chapter 8's filtered_kept, kept whole instead of cut at 50)."""
    from pre_ranking import student_scores
    from ranking_data import candidate_pools
    from slate_data import POOL
    users, items, S = W["users"], W["items"], sp["S"]
    pools = candidate_pools(S, users, items, rows, POOL)
    for k, u in enumerate(rows):
        p = pools[k][pools[k] >= 0]
        p = rs.filter(u, p)
        pools[k] = -1
        pools[k, :len(p)] = p
    sc = np.where(pools >= 0, student_scores(sp["pre"], S, users, items, rows, pools), -np.inf)
    order = np.argsort(-sc, axis=1, kind="stable")
    return np.where(np.take_along_axis(sc, order, axis=1) > -np.inf, np.take_along_axis(pools, order, axis=1), -1)


def forbidden_shown(top, bad, items):
    """Videos in the rows that the request was not allowed to show, and the requests
    with at least one (products, such as the ad, are checked by their own rules)."""
    i = np.maximum(top, 0)
    video = (top >= 0) & ~items["is_product"][i]
    hit = video & np.take_along_axis(bad, i, axis=1)
    return int(hit.sum()), int(hit.any(axis=1).sum())


def chapter8_rows(W, sp, R, bad):
    """Chapter 8's ranker for each request: section 8.2's filters and the pool from
    the day-23 snapshot, the pre-ranker's first 50 videos that this request may show
    (eligible_kept), in the heavy ranker's score order. Also returns how many
    videos the snapshot's own 50 would have shown against this week's rules."""
    from filtering import RuleSets
    from rankers import score_pools
    from slate_data import rules
    from slates import score_order
    users, items, S = W["users"], W["items"], sp["S"]
    rl = rules(W["cfg"], users, items)
    rs = RuleSets(rl, [np.flatnonzero(S["used"][u]) for u in range(len(users["vec"]))], len(items["vec"]), "numpy")
    uu, inv = np.unique(R["user"], return_inverse=True)
    ranked = ranked_pools(sp, W, rs, uu)
    old = ranked[:, :50][inv]                                     # the snapshot's 50, the same all week
    old_top = score_order(old, np.where(old >= 0, score_pools(sp["heavy"], S, users, items, R["user"], old), -np.inf))
    kept = eligible_kept(ranked, inv, bad)
    heavy = score_pools(sp["heavy"], S, users, items, R["user"], kept)
    top = score_order(kept, np.where(kept >= 0, heavy, -np.inf))
    return top, rl, forbidden_shown(old_top, bad, items)


def contract_checks():
    """Two small fixtures for the test contract (section 9.2): a late login, and a
    phone whose clicks arrive out of order."""
    # one visit on one phone: anonymous clicks done at 1 and 2, received at 5 and 6;
    # the login (seq 3) is received at 20
    e = {"event": np.array(["click", "click", "login"]), "user_id": np.array([-1, -1, 7]),
         "session_id": np.array([1, 1, 1]), "anonymous_id": np.array([9, 9, 9]),
         "seq": np.array([1, 2, 3]), "server_time": np.array([5.0, 6.0, 20.0]), "event_id": np.array([0, 1, 2])}
    user, start, until = link_intervals(e)
    C = {"known": np.maximum(start, e["server_time"])[:2], "until": until[:2], "t": np.array([1.0, 2.0])}
    assert user[0] == user[1] == 7 and not known(C, 10).any() and known(C, 21).all()
    # a second account logs in on the same visit, received at 30: the link ends there
    e2 = {k: np.append(v, x) for (k, v), x in zip(e.items(), ["login", 8, 1, 9, 4, 30.0, 3])}
    user2, start2, until2 = link_intervals(e2)
    assert until2[0] == 30 and until2[1] == 30
    # the phone did a, b, c (seq 1, 2, 3); b's upload was delayed, so its estimated time is later than c's
    o = device_order(np.array([7, 7, 7]), np.array([9, 9, 9]), np.array([1, 2, 3]),
                     np.array([1.0, 9.0, 3.0]), np.array([0, 1, 2]))
    assert o.tolist() == [0, 1, 2]
    # two phones: phone 9 did x then y; phone 8 did z between them in time
    o = device_order(np.array([7, 7, 7]), np.array([9, 9, 8]), np.array([1, 2, 1]),
                     np.array([1.0, 5.0, 3.0]), np.array([0, 1, 2]))
    assert o.tolist() == [0, 2, 1]
    return "passed: a click counts only after the login that links it has arrived; a later login of a " \
           "second account ends the link; one phone's clicks keep its own order"


if __name__ == "__main__":
    t0 = time.time()
    if not sys.argv[1:]:                                           # each world in its own process, to save memory
        print(environment())
        print("Contract checks " + contract_checks())
        for k in range(len(WORLDS)):
            subprocess.run([sys.executable, "-u", __file__, str(k)], check=True)
        sys.exit()
    for name in [WORLDS[int(a)] for a in sys.argv[1:]]:
        W = load(name)
        C = W["clicks"]
        e_all = W["events"]
        order = np.lexsort((e_all["server_time"], e_all["event_id"]))
        first = order[np.unique(e_all["event_id"][order], return_index=True)[1]]
        e = {k: v[first] for k, v in e_all.items()}
        user, _, until = link_intervals(e)
        same = np.array_equal(np.where(np.isinf(until), user, -1), link_same_visit(e))
        del e, e_all, user, until
        same_phone = (C["user"][1:] == C["user"][:-1]) & (C["install"][1:] == C["install"][:-1])
        swapped = int((same_phone & (C["t"][1:] < C["t"][:-1])).sum())
        print(f"{name}: once every login has arrived, the replayed links equal Chapter 2's link_same_visit: {same}; "
              f"clicks credited later than they arrived, because the login came later: "
              f"{int((C['known'] > C['arrival']).sum()):,}; links ended by a later login: "
              f"{int(np.isfinite(C['until']).sum()):,}; neighbouring clicks on one phone whose estimated times "
              f"are in the wrong order (the counter puts them right): {swapped:,}")
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
