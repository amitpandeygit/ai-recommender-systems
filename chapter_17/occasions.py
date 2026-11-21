"""Shopping for an occasion: a planted layer of Tesselo's shop for section 17.4.

In Chapter 3's shop a basket follows the buyer's hidden taste, plus planted complements
that were drawn at random. Nothing in it follows what the words mean: a sunscreen is
no more likely to be bought with a hat than with a knife. So an LLM's knowledge of the
world could not help there, and section 17.4 tests it there too, as a control.

Real shoppers also buy for an occasion: a beach holiday, moving into a new home. The
products of one occasion come from different categories, so no category feature can
link them, but their names do, to anyone who knows what the words mean. This layer adds
such baskets. The six occasions and their concepts were written by hand, before any
language model was run, and are fixed here:

  beach holiday     sunscreen, hat, sandals, speaker, kite, ball, bottle
  new home          lamp, curtain, rug, chair, shelf, wall clock, cooking pot, vase
  road trip         car mount, power bank, phone stand, earbuds, pillow, cable, adapter
  child's birthday  doll, teddy bear, building blocks, puzzle, crayons, slime, plate
  winter            jacket, scarf, blanket, kettle, mug, moisturizer, body lotion
  gym               bottle, earbuds, smartwatch, shoes, face wash, soap

A product belongs to an occasion if its title names one of the occasion's concepts.
Each user shops for one or two occasions (their own, fixed for the month), and 8,000
occasion baskets of 2 to 4 products are added over the 30 days, with buyers drawn like
Chapter 3's (heavy buyers order 8 times as often). Within a basket, products are chosen
as Chapter 3 chooses them, by popularity and the buyer's hidden taste, but only among
the occasion's products. The layer has its own generator (seed cfg.seed + 1710) and
changes no earlier order, so Chapter 3's shop, and every result that used it, stays the
same.

Run from the repository root to see the layer (under a minute):
    python chapter_17/occasions.py
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for _ch in ("chapter_17", "chapter_11", "chapter_10", "chapter_03"):
    sys.path.insert(0, str(HERE.parent / _ch))
from tesselo_shop import DAY  # noqa: E402
from title_words import WORDS  # noqa: E402

OCCASIONS = {
    "beach holiday": ["sunscreen", "hat", "sandals", "speaker", "kite", "ball", "bottle"],
    "new home": ["lamp", "curtain", "rug", "chair", "shelf", "wall clock", "cooking pot", "vase"],
    "road trip": ["car mount", "power bank", "phone stand", "earbuds", "pillow", "cable", "adapter"],
    "child's birthday": ["doll", "teddy bear", "building blocks", "puzzle", "crayons", "slime", "plate"],
    "winter": ["jacket", "scarf", "blanket", "kettle", "mug", "moisturizer", "body lotion"],
    "gym": ["bottle", "earbuds", "smartwatch", "shoes", "face wash", "soap"],
}
N_BASKETS = 8_000
SEED_OFFSET = 1710
TOKEN = {ws[0]: t for t, ws in WORDS.items()}                  # English word -> Chapter 3 token


def membership(items, titles=None):
    """products x occasions: True where the product's title names a concept of the
    occasion. titles: the titles to read (the true ones by default)."""
    titles = items["title"] if titles is None else titles
    sets = [{TOKEN[w] for w in ws} for ws in OCCASIONS.values()]
    M = np.zeros((len(items["vec"]), len(OCCASIONS)), dtype=bool)
    for i in np.flatnonzero(items["is_product"]):
        toks = {str(t) for t in titles[i]}
        M[i] = [bool(toks & s) for s in sets]
    return M


def occasion_orders(W, orders, topup):
    """The occasion baskets, as extra order lines in Chapter 3's format, with order ids
    after Chapter 3's. Also returns each user's occasions."""
    cfg, users, items = W["cfg"], W["users"], W["items"]
    rng = np.random.default_rng(cfg.seed + SEED_OFFSET)
    n_users = len(users["vec"])
    M = membership(items)
    M[topup] = False
    n_occ = len(OCCASIONS)
    own = np.zeros((n_users, n_occ), dtype=bool)
    for u in range(n_users):
        own[u, rng.choice(n_occ, 1 + int(rng.random() < 0.5), replace=False)] = True
    heavy = rng.random(n_users) < 0.05
    rate = np.where(heavy, 8.0, 1.0)
    buyer = rng.choice(n_users, N_BASKETS, p=rate / rate.sum())
    when = rng.random(N_BASKETS) * cfg.days * DAY
    base = 0.8 * np.log(items["popularity"])
    first = int(orders["order_id"].max()) + 1
    rows_o, rows_u, rows_t, rows_i, rows_occ = [], [], [], [], []
    for b in range(N_BASKETS):
        u = buyer[b]
        o = int(rng.choice(np.flatnonzero(own[u])))
        cand = np.flatnonzero(M[:, o])
        logit = base[cand] + 4.0 * items["vec"][cand] @ users["vec"][u]
        gumbel = logit - np.log(-np.log(rng.random(len(cand))))
        k = int(rng.integers(2, 5))
        for i in cand[np.argsort(-gumbel)[:k]]:
            rows_o.append(first + b), rows_u.append(u), rows_t.append(when[b]), rows_i.append(i), rows_occ.append(o)
    extra = {"order_id": np.array(rows_o), "user_id": np.array(rows_u), "time": np.array(rows_t),
             "item_id": np.array(rows_i), "_occasion": np.array(rows_occ)}
    return extra, own


def with_occasions(W, orders, topup):
    """Chapter 3's orders plus the occasion baskets, sorted by time."""
    extra, own = occasion_orders(W, orders, topup)
    merged = {k: np.r_[orders[k], extra[k]] for k in ("order_id", "user_id", "time", "item_id")}
    order = np.argsort(merged["time"], kind="stable")
    return {k: v[order] for k, v in merged.items()}, extra, own


if __name__ == "__main__":
    from content_data import load
    from tesselo_shop import build
    W = load()
    _, items_b, _, _, orders, truth = build(W["cfg"])
    assert np.array_equal(items_b["vec"], W["items"]["vec"])
    merged, extra, own = with_occasions(W, orders, truth["topup"])
    M = membership(W["items"])
    prods = W["items"]["is_product"]
    print(f"Chapter 3's shop: {len(np.unique(orders['order_id'])):,} orders, {len(orders['item_id']):,} lines. "
          f"Occasion layer: {N_BASKETS:,} baskets, {len(extra['item_id']):,} lines "
          f"({len(extra['item_id']) / N_BASKETS:.2f} per basket)")
    print(f"Products in at least one occasion: {M[prods].any(axis=1).sum():,} of {prods.sum():,}; in two or more: "
          f"{(M[prods].sum(axis=1) >= 2).sum():,}")
    for k, (name, ws) in enumerate(OCCASIONS.items()):
        print(f"  {name:<18}{M[:, k].sum():>5} products, {own[:, k].sum():>5} users, "
              f"{(extra['_occasion'] == k).sum():>6,} lines")
