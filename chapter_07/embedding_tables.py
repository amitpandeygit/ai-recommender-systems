"""Section 7.7 - Large embedding tables and how to handle billions of ids.

The DLRM ranker of section 7.4, with six ways to store the video-id table
(rankers.ItemTable). The same table also builds each user's vector from their
history, so every change reaches both sides:
  full        - one row per video id (5,000 rows)
  hash 1,000  - ids hashed into 1,000 rows: most videos share a row
  hash 250    - ids hashed into 250 rows
  QR 71 x 71  - the quotient-remainder trick: two tables of 71 rows (71 x 71 > 5,000);
                a video's vector is the product of its two rows, unique to it
  rare shared - videos with fewer than 20 training impressions share one row
  no video id - one row for every video: the model must use the other inputs
Epochs are chosen for each table on the validation week, as in heavy_ranking.py;
each table is then trained from three seeds, because with this little data the
seed moves the online result more than most of the differences between tables.
The script then prints the memory a table needs at production scale.

Needs Python 3.12 or later and PyTorch. Run from the repository root (about four minutes):
    python chapter_07/embedding_tables.py
"""
import os
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "2"              # NumPy's linear algebra: the same thread count on every run
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ranking_data import DAY, Config, build, candidate_pools, snapshot  # noqa: E402
from rankers import (DIM, ItemTable, Ranker, hash_rows, meta_for, predict, rerank,  # noqa: E402
                     score_pools, tensors, train)
from heavy_ranking import EPOCHS, POOL, epoch_losses, impression_auc, rows_for  # noqa: E402
from evaluate import expected_outcomes  # noqa: E402

RARE = 20


def tables(n_items, counts):
    return {"full": lambda: ItemTable(n_items, DIM),
            "hash 1,000": lambda: ItemTable(n_items, DIM, "hash", rows=1_000),
            "hash 250": lambda: ItemTable(n_items, DIM, "hash", rows=250),
            "QR 71 x 71": lambda: ItemTable(n_items, DIM, "qr", rows=71),
            f"rare shared (< {RARE})": lambda: ItemTable(n_items, DIM, "rare", counts=counts, min_count=RARE),
            "no video id": lambda: ItemTable(n_items, DIM, "hash", rows=1)}


def shared_share(table, ids):
    """Share of the given ids whose row (or pair of rows) is used by another of them."""
    if table.mode == "full":
        return 0.0
    if table.mode == "qr":
        key = (ids // table.rows) * table.rows + ids % table.rows     # the pair of rows, unique per id
    elif table.mode == "hash":
        key = hash_rows(ids, table.rows)
    else:
        key = table.index.numpy()[ids]
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    return float(np.mean(cnt[inv] > 1))


if __name__ == "__main__":
    t0 = time.time()
    cfg = Config()
    users, items, served, events, orders, truth = build(cfg)
    n_items = cfg.num_items
    video_ids = np.flatnonzero(~items["is_product"])
    snaps = {d: snapshot(events, served, users, items, d * DAY) for d in (9, 16, 23)}
    week1 = rows_for(events, served, users, items, [(9, 16)])
    week2 = rows_for(events, served, users, items, [(16, 23)])
    weeks12 = rows_for(events, served, users, items, [(9, 16), (16, 23)])
    week3 = rows_for(events, served, users, items, [(23, 30)])
    b1 = tensors([snaps[9]], users, items, week1["user"], week1["item"], week1["snap"])
    b2 = tensors([snaps[16]], users, items, week2["user"], week2["item"], week2["snap"])
    b12 = tensors([snaps[9], snaps[16]], users, items, weeks12["user"], weeks12["item"], weeks12["snap"])
    b3 = tensors([snaps[23]], users, items, week3["user"], week3["item"], week3["snap"])
    meta1 = meta_for(users, items, b1["dense"].numpy())
    meta12 = meta_for(users, items, b12["dense"].numpy())
    count1 = np.bincount(week1["item"], minlength=n_items)
    count12 = np.bincount(weeks12["item"], minlength=n_items)
    print(f"Training impressions per video, days 9 to 23: median {np.median(count12[video_ids]):.0f}, "
          f"fewest {count12[video_ids].min()}, most {count12[video_ids].max():,}; "
          f"{np.mean(count12[video_ids] < RARE):.1%} of videos have fewer than {RARE}")

    S = snaps[23]
    aud = np.flatnonzero(users["current_consent"])
    pools = candidate_pools(S, users, items, aud, POOL)
    c3 = week3["click"]
    print("\nSection 7.7: DLRM with six video-id tables; epochs chosen on the validation week")
    print(f"  {'':<22}{'rows':>7}{'memory':>9}{'videos sharing':>16}{'epochs':>7}{'click AUC':>10}"
          f"{'online, seeds 0, 1, 2':>24}{'mean':>8}")
    for name in tables(n_items, count12):
        torch.manual_seed(0)
        t1 = None if name == "full" else tables(n_items, count1)[name]()
        losses, _ = epoch_losses(meta1, "dlrm", b1, week1, b2, week2, item_table=t1)  # "full" = section 7.4's DLRM
        ep = EPOCHS[int(np.argmin(np.round(losses, 9)))]
        online, auc_c = [], []
        for seed in (0, 1, 2):
            torch.manual_seed(seed)
            table = None if name == "full" else tables(n_items, count12)[name]()
            m = train(Ranker(meta12, "dlrm", item_table=table, seed=seed), b12, weeks12, epochs=ep, seed=seed)
            table = m.inputs.item
            pc, pp = predict(m, b3, len(c3), position=torch.as_tensor(week3["position"]))
            top = rerank(pools, score_pools(m, S, users, items, aud, pools), 10)
            online.append(expected_outcomes(top, aud, users, items, cfg)[1].sum(axis=1).mean())
            auc_c.append(impression_auc(pc, c3))
        n_rows = table.n_rows()
        print(f"  {name:<22}{n_rows:>7,}{n_rows * DIM * 4 / 1024:>7.0f} KB{shared_share(table, video_ids):>16.1%}"
              f"{ep:>7}{np.mean(auc_c):>10.3f}"
              + "".join(f"{x:>8.4f}" for x in online) + f"{np.mean(online):>8.4f}")
    print(f"  (memory = rows x {DIM} numbers x 4 bytes; click AUC is the mean over the three seeds; online = expected "
          f"satisfied plays, {len(aud):,} consenting users)")
    print(f"[{time.time() - t0:.0f} s]")

    # ---- production scale: a table of a billion ids
    n, dim = 1_000_000_000, 64
    print(f"\nA table of {n:,} ids with {dim} numbers each:")
    for label, size in (("32-bit floats", 4), ("16-bit floats", 2), ("8-bit integers", 1)):
        print(f"  {label:<15} {n * dim * size / 1e9:>6.0f} GB")
    print(f"  training with Adam: the weights and Adam's two stored numbers per weight, all 32-bit, "
          f"{n * dim * 4 * 3 / 1e9:.0f} GB that persist; a full 32-bit gradient adds {n * dim * 4 / 1e9:.0f} GB "
          f"during each step, {n * dim * 4 * 4 / 1e9:,.0f} GB in all (sparse updates touch only the rows in a batch)")
    r = math.isqrt(n - 1) + 1
    print(f"  the QR trick with a remainder table of {r:,} rows: {r:,} + {-(-n // r):,} rows, "
          f"{(r + -(-n // r)) * dim * 4 / 1e6:.0f} MB")
    for m_rows in (100_000_000, 1_000_000_000, 10_000_000_000):
        print(f"  hashed into {m_rows:>14,} rows: share of ids that share their row "
              f"about {1 - math.exp(-(n - 1) / m_rows):.1%}")
    print(f"[{time.time() - t0:.0f} s]")
