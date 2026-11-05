"""Next-video tests: baselines, GRU4Rec, SASRec and BERT4Rec (sections 9.3 and 9.4).

Every method answers each test request (days 23 to 30) with a row of 10 videos,
using what the server had received before the request. It is compared with
Chapter 8's ranker: section 8.2's filters, Chapter 6's pool, the pre-ranker's 50
and the heavy ranker's order, from the day-23 snapshot (so one row per user for
the whole week). No business rules here: section 9.7 adds them.

Settings are chosen on the validation week (fitted on what had arrived by day 16,
tested on the requests of days 16 to 23), by NDCG@10, Chapter 5's rule: for each
neural model, the number of epochs (checked every 5, or every 10 for BERT4Rec).
Each neural model is then refitted on what had arrived by day 23 from seeds 0, 1
and 2. Results are given in both worlds.

Needs Python 3.12 or later and PyTorch. Run from the repository root (about 95 minutes;
add 0 or 1 to run one world only; with no number, each world runs in a process of
its own, one after the other, to keep memory down):
    python chapter_09/next_item.py
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
from seq_data import (DAY, WORLDS, chapter8_rows, exclude_mask, histories, load, ndcg, online, paired,  # noqa: E402
                      requests, top10, training_sequences)
import baselines as B  # noqa: E402
from seq_models import SeqModel, fit, score  # noqa: E402
from slate_data import split  # noqa: E402

NETS = (("GRU4Rec", "full softmax"), ("SASRec", "BCE, 1 negative"), ("SASRec", "sampled softmax, 100"),
        ("SASRec", "full softmax"), ("BERT4Rec", "masked, full softmax"))
MAX_EPOCHS = {"GRU4Rec": 50, "SASRec": 50, "BERT4Rec": 100}
EVERY = {"GRU4Rec": 5, "SASRec": 5, "BERT4Rec": 10}
SEEDS = (0, 1, 2)


def phase(W, start, S):
    """Requests of [start, start + 7) days, their histories and what may not be shown."""
    R = requests(W, start, start + 7)
    H, in_visit = histories(W, R)
    return R, H, in_visit, exclude_mask(W, S, R, H)


def baseline_rows(W, cutoff, R, H, in_visit, bad):
    n = len(W["items"]["vec"])
    seqs, _ = training_sequences(W, cutoff)
    pop = B.popularity(W, cutoff)
    G = B.cooccurrence(seqs, n)
    T = B.markov(seqs, n)
    M = B.session_matrix(W, cutoff)
    X = B.contexts(H, in_visit, n)
    return {"popularity, last 7 days": top10(pop, bad),
            "co-occurrence": top10(lambda a, b: B.cooc_scores(G, H[a:b]), bad),
            "session kNN": top10(lambda a, b: B.session_knn(M, X[a:b]), bad),
            "Markov chain": top10(lambda a, b: B.markov_scores(T, H[a:b], pop), bad)}


def row_line(label, W, R, top, base, extra=""):
    on = online(W, R, top)
    later = R["place"] >= 1
    d, se = paired(on, base) if base is not None else (0.0, 0.0)
    against = "" if base is None else f"{d:+.4f} ({se:.4f})"
    print(f"  {label:<34}{extra:>8}{ndcg(R, top):>9.4f}{on.mean():>9.4f}{on[later].mean():>9.4f}{against:>18}")
    return on


if __name__ == "__main__":
    t0 = time.time()
    if not sys.argv[1:]:                                           # each world in its own process, to save memory
        for k in range(len(WORLDS)):
            subprocess.run([sys.executable, "-u", __file__, str(k)], check=True)
        sys.exit()
    worlds = [WORLDS[int(a)] for a in sys.argv[1:]]               # one world, by its number
    for name in worlds:
        W = load(name)
        video = ~W["items"]["is_product"]
        n = len(video)
        print(f"\n{name}  [{time.time() - t0:.0f} s]")
        Rv, Hv, ivv, badv = phase(W, 16, W["snaps"][16])
        seqs_v, _ = training_sequences(W, 16 * DAY)
        print("Validation week: NDCG@10 after each check (epochs)")
        chosen = {}
        for kind, loss in NETS:
            curve = []
            m = SeqModel(n, kind, W["items"])
            fit(m, seqs_v, MAX_EPOCHS[kind], loss, every=EVERY[kind],
                after=lambda e, m: curve.append((e, ndcg(Rv, top10(lambda a, b: score(m, Hv[a:b]), badv)))))
            best = max(curve, key=lambda x: round(x[1], 9))
            chosen[(kind, loss)] = best[0]
            print(f"  {kind + ', ' + loss:<34}" + " ".join(f"{e}:{x:.4f}" for e, x in curve)
                  + f"   chosen: {best[0]}  [{time.time() - t0:.0f} s]")
        del Rv, Hv, badv
        sp = split(W, "test")
        R, H, iv, bad = phase(W, 23, sp["S"])
        later = R["place"] >= 1
        print(f"Test week: {len(R['user']):,} requests ({later.mean():.1%} later in their visit; "
              f"{np.mean([len(p) > 0 for p in R['pos']]):.1%} with a satisfied video)  [{time.time() - t0:.0f} s]")
        print(f"  {'':<34}{'epochs':>8}{'NDCG@10':>9}{'online':>9}{'later':>9}{'against Chapter 8':>18}")
        ch8, _ = chapter8_rows(W, sp, R)
        base = row_line("Chapter 8's ranker", W, R, ch8, None)
        for label, top in baseline_rows(W, 23 * DAY, R, H, iv, bad).items():
            row_line(label, W, R, top, base)
        seqs, _ = training_sequences(W, 23 * DAY)
        for kind, loss in NETS:
            ons, nds, ons_l = [], [], []
            for seed in SEEDS:
                m = fit(SeqModel(n, kind, W["items"], seed=seed), seqs, chosen[(kind, loss)], loss, seed=seed)
                top = top10(lambda a, b: score(m, H[a:b]), bad)
                on = online(W, R, top)
                ons.append(on)
                nds.append(ndcg(R, top))
            diffs = np.array([paired(o, base)[0] for o in ons])
            means = [o.mean() for o in ons]
            print(f"  {kind + ', ' + loss:<34}{chosen[(kind, loss)]:>8}{np.mean(nds):>9.4f}{np.mean(means):>9.4f}"
                  f"{np.mean([o[later].mean() for o in ons]):>9.4f}"
                  f"{f'{diffs.mean():+.4f} ({diffs.std(ddof=1) / np.sqrt(len(diffs)):.4f})':>18}"
                  f"   seeds: " + ", ".join(f"{x:.4f}" for x in means) + f"  [{time.time() - t0:.0f} s]")
        print("  (online: expected satisfied plays per row, all test requests; later: requests after the first of "
              "their visit; against: paired difference over the same requests, standard error in brackets, across "
              "seeds for the neural models; NDCG@10 and online of the neural models are means over 3 seeds)")
