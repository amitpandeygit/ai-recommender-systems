"""Tesselo shop and catalogue extension (Chapter 3).

Chapter 2's generator (chapter_02/tesselo_data.py) simulates the home feed. This
file adds what Chapter 3 needs and the feed logs do not have:

  * item attributes   - a category, a language and a short title for every item
  * user languages    - the language each user's app is set to
  * shop orders       - multi-item baskets over the same 30 days

It uses its own random generator (seed cfg.seed + 300), so Chapter 2's logs and
printed numbers do not change. Attributes are derived from the same hidden taste
vectors that drive behaviour in Chapter 2: an item's category and language, and
a user's language, are the directions their vectors point in most. So content
carries real, but partial, information about taste, as in real life.

Planted in the orders, so that later code can be checked against the truth:
  * complements  - 200 "anchor" products each have 2 complements (a phone and its
                   case); a basket with the anchor adds each complement with
                   probability 0.4. The link is one-way: buying the case does
                   not add the phone.
  * a universal item - a mobile data top-up, added to 25% of baskets whatever
                   else is in them
  * heavy buyers - 5% of users order 8 times as often, with bigger baskets
  * a trend      - 5 products from the best-selling 20% sell 4 times more
                   often in the last 3 days

Every number is a toy setting chosen to make effects visible, not a measurement.
Fields whose names start with "_" are hidden truth, kept only for checking.

Needs Python 3.12 or later. Run from the repository root:
    python chapter_03/tesselo_shop.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "chapter_02"))
from tesselo_data import Config, simulate  # noqa: E402

LANGUAGES = ["en", "hi", "pt", "es", "id", "ar"]
VIDEO_CATEGORIES = ["cooking", "music", "comedy", "sport", "travel", "tech", "beauty", "news"]
PRODUCT_CATEGORIES = ["phone gear", "kitchen", "fashion", "skincare", "home", "toys"]
CATEGORIES = VIDEO_CATEGORIES + PRODUCT_CATEGORIES
WORDS_PER_CATEGORY = 10
GENERIC_WORDS = ["new", "best", "easy", "top", "quick", "cheap", "review", "2026"]
DAY = 86_400


def vocabulary():
    """Title words: 10 per category ("cooking_0" ... "cooking_9") plus generic words.

    The words stand in for real title words. Words are shared across languages
    here, as if titles had been translated; real multilingual titles need the
    language-aware encoders of Chapter 11.
    """
    words = [f"{c.replace(' ', '_')}_{k}" for c in CATEGORIES for k in range(WORDS_PER_CATEGORY)]
    return words + GENERIC_WORDS


def closest_direction(vec, n_directions, rng, noise_share):
    """Index of the random direction each vector points to most; a share at random."""
    d = rng.normal(size=(n_directions, vec.shape[1]))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    pick = np.argmax(vec @ d.T, axis=1)
    noisy = rng.random(len(vec)) < noise_share
    pick[noisy] = rng.integers(0, n_directions, noisy.sum())
    return pick


def add_attributes(cfg, users, items):
    rng = np.random.default_rng(cfg.seed + 300)
    lang_dirs_seed = rng.integers(1 << 30)                 # users and items share language directions
    users["language"] = closest_direction(users["vec"], len(LANGUAGES),
                                          np.random.default_rng(lang_dirs_seed), 0.10)
    items["language"] = closest_direction(items["vec"], len(LANGUAGES),
                                          np.random.default_rng(lang_dirs_seed), 0.10)
    n = len(items["vec"])
    cat = np.empty(n, dtype=np.int64)
    prod = items["is_product"]
    cat[~prod] = closest_direction(items["vec"][~prod], len(VIDEO_CATEGORIES), rng, 0.15)
    cat[prod] = len(VIDEO_CATEGORIES) + closest_direction(items["vec"][prod],
                                                          len(PRODUCT_CATEGORIES), rng, 0.15)
    items["category"] = cat

    # Titles: 3 words from the item's own category, 1 generic word, and with
    # probability 0.2 one word from another category (noise). Each word has a hidden
    # direction; within a category, words closer to the item's taste vector are more
    # likely, so titles say something about taste beyond the category.
    words = vocabulary()
    word_dir = rng.normal(size=(len(CATEGORIES) * WORDS_PER_CATEGORY, items["vec"].shape[1]))
    word_dir /= np.linalg.norm(word_dir, axis=1, keepdims=True)
    titles = []
    for i, c in enumerate(cat):
        own = np.arange(c * WORDS_PER_CATEGORY, (c + 1) * WORDS_PER_CATEGORY)
        p = np.exp(4.0 * word_dir[own] @ items["vec"][i])
        t = [words[k] for k in rng.choice(own, 3, replace=False, p=p / p.sum())]
        t.append(rng.choice(GENERIC_WORDS))
        if rng.random() < 0.2:
            other = rng.choice([x for x in range(len(CATEGORIES)) if x != c])
            t.append(words[other * WORDS_PER_CATEGORY + rng.integers(WORDS_PER_CATEGORY)])
        titles.append(t)
    items["title"] = titles
    return rng


def make_orders(cfg, users, items, rng, n_orders=40_000):
    """Shop orders over the same 30 days. Returns one row per (order, item) line."""
    products = np.flatnonzero(items["is_product"])
    n_users = len(users["vec"])
    heavy = rng.random(n_users) < 0.05
    rate = np.where(heavy, 8.0, 1.0)
    order_user = rng.choice(n_users, n_orders, p=rate / rate.sum())
    order_time = np.sort(rng.random(n_orders)) * cfg.days * DAY
    order_user = order_user[rng.permutation(n_orders)]
    size = 1 + np.minimum(rng.poisson(np.where(heavy[order_user], 2.5, 0.8)), 7)

    # The universal item: the most popular product becomes a mobile data top-up.
    topup = products[np.argmax(items["popularity"][products])]
    regular = products[products != topup]

    # Planted complements: anchor -> 2 complements from other product categories.
    pw = items["popularity"][regular] ** 0.3
    anchors = rng.choice(regular, 200, replace=False, p=pw / pw.sum())
    complements = {}
    for a in anchors:
        pool = regular[(items["category"][regular] != items["category"][a]) & (regular != a)]
        complements[int(a)] = rng.choice(pool, 2, replace=False).tolist()

    # A trend: 5 products sell 4 times more often in the last 3 days.
    busy = regular[items["popularity"][regular] >= np.quantile(items["popularity"][regular], 0.8)]
    trend = rng.choice(busy, 5, replace=False)               # products that already sell well
    base = 0.8 * np.log(items["popularity"][regular])
    taste = users["vec"] @ items["vec"][regular].T                 # users x regular products
    is_trend = np.isin(regular, trend)

    rows_order, rows_item = [], []
    for o in range(n_orders):
        u, late = order_user[o], order_time[o] >= (cfg.days - 3) * DAY
        logit = base + 4.0 * taste[u] + np.where(is_trend & late, np.log(4.0), 0.0)
        gumbel = logit + rng.gumbel(size=len(regular))             # sample without replacement
        chosen = regular[np.argsort(-gumbel)[: size[o]]].tolist()
        basket = [chosen[0]]
        for c in complements.get(chosen[0], []):                    # complements of the first item
            if rng.random() < 0.4:
                basket.append(c)
        basket += [x for x in chosen[1:] if x not in basket][: max(size[o] - len(basket), 0)]
        if rng.random() < 0.25:
            basket.append(int(topup))
        rows_order += [o] * len(basket)
        rows_item += basket
    rows_order = np.array(rows_order)
    orders = {"order_id": rows_order, "user_id": order_user[rows_order],
              "time": order_time[rows_order], "item_id": np.array(rows_item)}
    truth = {"complements": complements, "topup": int(topup), "trend": trend, "heavy": heavy}
    return orders, truth


def build(cfg=Config()):
    """Chapter 2's world and logs, plus Chapter 3's attributes and orders."""
    users, items, served, events = simulate(cfg)
    rng = add_attributes(cfg, users, items)
    orders, truth = make_orders(cfg, users, items, rng)
    return users, items, served, events, orders, truth


if __name__ == "__main__":
    cfg = Config()
    users, items, served, events, orders, truth = build(cfg)
    n_orders = len(np.unique(orders["order_id"]))
    print(f"Orders: {n_orders:,}  lines: {len(orders['item_id']):,}  "
          f"average basket: {len(orders['item_id']) / n_orders:.2f} items")
    per_user = np.bincount(orders["user_id"], minlength=cfg.num_users)
    h = truth["heavy"]
    print(f"Heavy buyers: {h.mean():.1%} of users, {per_user[h].sum() / per_user.sum():.1%} of order lines")
    print(f"Baskets with the data top-up: "
          f"{len(np.unique(orders['order_id'][orders['item_id'] == truth['topup']])) / n_orders:.1%}")
    print("Users per language:", dict(zip(LANGUAGES, np.bincount(users["language"]).tolist())))
    print("Items per category:", dict(zip(CATEGORIES, np.bincount(items["category"]).tolist())))
    print("Example titles:", [" ".join(items["title"][i]) for i in range(3)])
    # How much do language and category say about taste? Mean hidden match, same vs different.
    u = np.arange(0, cfg.num_users, 8)
    m = users["vec"][u] @ items["vec"].T
    same = users["language"][u][:, None] == items["language"][None, :]
    print(f"Mean hidden match, same language: {m[same].mean():.3f}, different: {m[~same].mean():.3f}")
