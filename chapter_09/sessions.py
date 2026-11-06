"""Listing 9.5 - Short-term and long-term interest, and visits without a profile (section 9.5).

Two questions, in both worlds.

1. How much of a sequence model's answer comes from the current visit? The model that
   section 9.4's validation week chose in each world (seq_data.CHOICE) answers the
   later requests of each visit from three histories:
     everything     - all clicks received before the request (the default)
     before the visit - the clicks before the visit began: what a system that updates
                      once a visit, or once a day, would know
     this visit only  - the clicks of the current visit, and nothing older

2. Visits with no profile: requests served while logged out, or to users who had
   personalisation off at the time. Tesselo may not use their past, so today they
   get the most popular videos in their app language (Chapter 3). A session model
   reads only the clicks of the current visit, held in memory for the visit and
   kept nowhere afterwards:
     session kNN       - section 9.3's method on the visit's clicks
     session model     - the same kind of model, trained on visits instead of users
                         (visits of consenting, logged-in users only: Tesselo trains only
                         on data it is allowed to use), reading only the current visit
   The rules that can be known without a profile still apply: no 18+ videos (the age
   is unknown), the region's licences, removed videos.

Needs Python 3.12 or later and PyTorch. Run from the repository root (about 10 minutes;
add 0 or 1 to run one world only; with no number, each world runs in a process of
its own, one after the other, to keep memory down):
    python chapter_09/sessions.py
"""
import os
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "2"
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from seq_data import (CHOICE, DAY, MAX_LEN, WORLDS, available_clicks, environment, exclude_mask,  # noqa: E402
                      histories, known, load, ndcg, online, paired, requests, top10, training_sequences)
import baselines as B  # noqa: E402
from seq_models import SeqModel, fit, score  # noqa: E402
from implicit_scores import estimated_action_time  # noqa: E402
from slate_data import rules  # noqa: E402

MAX_EPOCHS = {"GRU4Rec": 50, "SASRec": 50, "BERT4Rec": 100}
EVERY = {"GRU4Rec": 5, "SASRec": 5, "BERT4Rec": 10}


def split_history(W, R, H):
    """Listing 9.5 (first part). The same history cut two ways: before the visit
    began, and the visit alone (left-padded, -1 where empty)."""
    C = W["clicks"]
    starts = np.searchsorted(C["user"], np.arange(len(W["users"]["vec"]) + 1))
    before, visit = np.full(H.shape, -1), np.full(H.shape, -1)
    for k, (u, t, s) in enumerate(zip(R["user"], R["time"], R["session"])):
        idx = available_clicks(C, starts[u], starts[u + 1], t)      # in the user's order (section 9.2)
        items, ses = C["item"][idx], C["session"][idx]
        old, new = items[ses != s][-MAX_LEN:], items[ses == s][-MAX_LEN:]
        if len(old):
            before[k, -len(old):] = old
        if len(new):
            visit[k, -len(new):] = new
    return before, visit


def anonymous_requests(W, start_day, end_day):
    """Listing 9.5 (second part). Requests with no profile to use (logged out, or
    personalisation off), each with the clicks of its own visit that the server had
    received before it, and nothing else."""
    s, ev, items = W["served"], W["events"], W["items"]
    first = s["position"] == 1
    keep = (first & (s["served_time"] >= start_day * DAY) & (s["served_time"] < end_day * DAY)
            & ((s["user_id"] < 0) | ~s["consent_at_time"]))
    idx = np.flatnonzero(keep)
    R = {"request": s["request_id"][idx], "user": s["_true_user"][idx], "session": s["session_id"][idx],
         "time": s["served_time"][idx],
         "intent": s["_intent"][idx] if "_intent" in s else np.full(len(idx), -1)}
    order = np.lexsort((ev["server_time"], ev["event_id"]))
    first_copy = order[np.unique(ev["event_id"][order], return_index=True)[1]]
    e = {k: v[first_copy] for k, v in ev.items()}
    t = estimated_action_time(e)
    clk = (e["event"] == "click") & (e["item_id"] >= 0) & ~items["is_product"][np.maximum(e["item_id"], 0)]
    by_session = {}
    for sess, q, item, tt, arr in zip(e["session_id"][clk], e["seq"][clk], e["item_id"][clk], t[clk],
                                      e["server_time"][clk]):
        by_session.setdefault(int(sess), []).append((q, tt, arr, int(item)))
    H = np.full((len(idx), MAX_LEN), -1)
    for k, (sess, tt) in enumerate(zip(R["session"], R["time"])):
        got = sorted(x for x in by_session.get(int(sess), []) if x[1] < tt and x[2] < tt)   # in the phone's order
        got = [x[3] for x in got][-MAX_LEN:]
        if got:
            H[k, -len(got):] = got
    hit = np.isin(ev["event"], ("complete", "like")) & (ev["item_id"] >= 0)
    pos = {}
    for r, i in zip(ev["request_id"][hit].tolist(), ev["item_id"][hit].tolist()):
        if not items["is_product"][i]:
            pos.setdefault(r, set()).add(i)
    R["pos"] = [pos.get(int(r), set()) for r in R["request"]]
    R["place"] = (H >= 0).sum(1)                                  # clicks so far in the visit
    return R, H


def anonymous_mask(W, R, H):
    """What may be shown without a profile: no products, no 18+ (age unknown), the
    region's licences, nothing removed, nothing clicked earlier in the visit."""
    users, items = W["users"], W["items"]
    rl = rules(W["cfg"], users, items)
    n = len(items["vec"])
    bad = np.zeros((len(R["user"]), n), dtype=bool)
    bad[:, items["is_product"] | rl["adult"] | rl["removed"]] = True
    for g in np.unique(rl["region"][R["user"]]):
        bad[np.ix_(rl["region"][R["user"]] == g, rl["unlicensed"][g])] = True
    rows = np.repeat(np.arange(len(H)), H.shape[1])
    flat = H.ravel()
    bad[rows[flat >= 0], flat[flat >= 0]] = True
    return bad


def session_sequences(W, cutoff):
    """Visits of consenting, logged-in users with 2+ clicks, as sequences."""
    C = W["clicks"]
    ok = known(C, cutoff)
    out = []
    o = np.lexsort((C["seq"][ok], C["session"][ok]))                # a visit is one phone: its own order
    ses, item = C["session"][ok][o], C["item"][ok][o]
    start = np.r_[0, np.flatnonzero(np.diff(ses)) + 1, len(ses)]
    for a, b in zip(start[:-1], start[1:]):
        if b - a >= 2:
            out.append(item[a:b])
    return out


if __name__ == "__main__":
    t0 = time.time()
    if not sys.argv[1:]:                                           # each world in its own process, to save memory
        print(environment())
        for k in range(len(WORLDS)):
            subprocess.run([sys.executable, "-u", __file__, str(k)], check=True)
        sys.exit()
    worlds = [WORLDS[int(a)] for a in sys.argv[1:]]               # one world, by its number
    for name in worlds:
        W = load(name)
        n = len(W["items"]["vec"])
        print(f"\n{name}  [{time.time() - t0:.0f} s]")
        kind, loss, ep = CHOICE[name]
        R = requests(W, 23, 30)
        H, in_visit = histories(W, R)
        bad = exclude_mask(W, W["snaps"][23], R, H)
        pop = B.popularity(W, 23 * DAY)
        before, visit = split_history(W, R, H)
        m = fit(SeqModel(n, kind, W["items"]), training_sequences(W, 23 * DAY)[0], ep, loss)
        later = (R["place"] >= 1) & (in_visit > 0)
        Rl = {k: (v[later] if isinstance(v, np.ndarray) else [x for x, f in zip(v, later) if f]) for k, v in R.items()}
        print(f"Section 9.5 (1): {kind} ({loss}, {ep} epochs) on the {later.sum():,} later requests with a click "
              f"earlier in the visit")
        print(f"  {'history read':<34}{'NDCG@10':>9}{'online':>9}{'against everything':>20}")
        res = {}
        for label, h in (("everything", H), ("before the visit", before), ("this visit only", visit)):
            hl, bl = h[later], bad[later]
            top = top10(lambda a, b: score(m, hl[a:b]), bl, tie=pop)
            res[label] = online(W, Rl, top)
            d = "" if label == "everything" else "{:+.4f} ({:.4f})".format(
                *paired(res[label], res["everything"], Rl["user"])[:2])
            print(f"  {label:<34}{ndcg(Rl, top):>9.4f}{res[label].mean():>9.4f}{d:>20}")

        Ra, Ha = anonymous_requests(W, 23, 30)
        bada = anonymous_mask(W, Ra, Ha)
        has = Ra["place"] >= 1
        print(f"Section 9.5 (2): {len(Ra['user']):,} requests with no profile (logged out or personalisation off); "
              f"{has.mean():.1%} have a click earlier in the visit  [{time.time() - t0:.0f} s]")
        S = W["snaps"][23]
        lang_pop = S["lang_pop"][W["users"]["language"][Ra["user"]]]
        sseq = session_sequences(W, 23 * DAY)
        sseq_v = session_sequences(W, 16 * DAY)
        Rva, Hva = anonymous_requests(W, 16, 23)
        bad_va = anonymous_mask(W, Rva, Hva)
        hv = Rva["place"] >= 1
        Rva_h = {k: (v[hv] if isinstance(v, np.ndarray) else [x for x, f in zip(v, hv) if f]) for k, v in Rva.items()}
        curve = []
        ms = SeqModel(n, kind, W["items"])
        lang_pop_v = W["snaps"][16]["lang_pop"][W["users"]["language"][Rva_h["user"]]]
        fit(ms, sseq_v, MAX_EPOCHS[kind], loss, every=EVERY[kind],
            after=lambda e, mm: curve.append((e, ndcg(Rva_h, top10(lambda a, b: score(mm, Hva[hv][a:b]),
                                                                   bad_va[hv], tie=lambda a, b: lang_pop_v[a:b])))))
        ep_s = max(curve, key=lambda x: round(x[1], 9))[0]
        ms = fit(SeqModel(n, kind, W["items"]), sseq, ep_s, loss)
        M = B.session_matrix(W, 23 * DAY)
        X = B.contexts(Ha, Ra["place"], n, fallback=0)
        lp = lambda a, b: lang_pop[a:b]                            # noqa: E731  the fallback, and the tie-break
        rows = {"popular in language (today)": top10(lp, bada),
                "session kNN": top10(lambda a, b: np.where(Ra["place"][a:b, None] >= 1,
                                                           B.session_knn(M, X[a:b]), lang_pop[a:b]), bada, tie=lp),
                f"session {kind} ({ep_s} epochs)": top10(lambda a, b: np.where(
                    Ra["place"][a:b, None] >= 1, score(ms, Ha[a:b]), lang_pop[a:b]), bada, tie=lp)}
        print(f"  {'':<34}{'NDCG@10':>9}{'online':>9}{'with a click':>14}{'against today':>18}")
        base = None
        for label, top in rows.items():
            on = online(W, Ra, top)
            base = on if base is None else base
            d = "" if on is base else "{:+.4f} ({:.4f})".format(*paired(on[has], base[has], Ra["session"][has])[:2])
            print(f"  {label:<34}{ndcg(Ra, top):>9.4f}{on.mean():>9.4f}{on[has].mean():>14.4f}{d:>18}")
        print(f"  (online: all no-profile requests; with a click: those with a click earlier in the visit, where the "
              f"against column is measured, with its standard error from 1,000 bootstrap draws of whole visits; "
              f"before the first click every method shows the language's popular videos)  [{time.time() - t0:.0f} s]")
