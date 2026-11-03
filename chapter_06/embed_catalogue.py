"""The vectors that vector_search.py searches (section 6.7), computed with PyTorch.

vector_search.py uses Faiss, and this file uses PyTorch. On some platforms (macOS
with the pip wheels) the two libraries each bring their own copy of the OpenMP
runtime, and loading both into one process aborts. So the PyTorch work runs here,
in its own process, and writes its results to one file:

  - the two-tower model of section 6.6 (MAIN settings), trained on data before day 23;
  - the vectors of all 1,000,000 catalogue items (tesselo_catalogue.py): learned
    ones for the 5,000 items with history, content-and-creator ones (zero id
    vector) for the 995,000 new items;
  - the query vectors of 1,000 users with history, drawn once at random;
  - Chapter 4's iALS vectors for the same users and the 5,000 items (section 6.7's
    first check), and each item's upload day (its filtered search).

vector_search.py runs this file itself, so readers type one command:
    python chapter_06/vector_search.py
It can also be run alone, with the output file as its argument.
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from tesselo_catalogue import build_catalogue  # noqa: E402
from two_tower import MAIN, embed_all, make_data, train_two_tower  # noqa: E402
from tesselo_shop import DAY, Config  # noqa: E402
from similarity_recs import item_features, user_item_scores  # noqa: E402
from implicit_mf import ials  # noqa: E402


def embed_catalogue(out_path):
    t0 = time.time()
    cfg = Config()
    users, items, events, cat, _ = build_catalogue(cfg)
    n_users, n_items = cfg.num_users, len(items["vec"])
    print(f"Catalogue: {len(cat['vec']):,} items, {cat['n_original']:,} with history  [{time.time() - t0:.0f} s]",
          flush=True)
    weight, used = user_item_scores(events, users, 23 * DAY, n_users, n_items)
    feats = item_features(items, n_items)
    data = make_data(weight, items, users, feats, int(items["creator"].max()) + 1)
    pu, pi = np.nonzero(weight > 0)
    model = train_two_tower(data, pu, pi, **MAIN)
    U, V5 = embed_all(model, data)
    hist = np.flatnonzero((weight > 0).any(axis=1))
    qusers = np.sort(np.random.default_rng(cfg.seed + 610).choice(hist, 1_000, replace=False))
    with torch.no_grad():                                             # new items: content and creator only
        V = np.zeros((len(cat["vec"]), V5.shape[1]))
        V[:n_items] = V5
        f = torch.as_tensor(cat["feats"])
        c = torch.as_tensor(cat["creator"])
        for s in range(n_items, len(V), 100_000):
            e = min(s + 100_000, len(V))
            V[s:e] = model.item_from_inputs(torch.zeros(e - s, model.item_id.embedding_dim),
                                            f[s:e], c[s:e]).numpy()
    X4, Y4 = ials(weight, 16, 1.0, 10.0)
    np.save(out_path, dict(V=V, Q=U[qusers], n_items=n_items, q4=X4[qusers], Y4=Y4,
                           upload_day=cat["upload_day"], seed=cfg.seed), allow_pickle=True)
    print(f"Item vectors: {V.shape[0]:,} x {V.shape[1]}; query users: {len(qusers):,}  [{time.time() - t0:.0f} s]",
          flush=True)


if __name__ == "__main__":
    embed_catalogue(sys.argv[1] if len(sys.argv) > 1 else str(HERE / "catalogue_vectors.npy"))
