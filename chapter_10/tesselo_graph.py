"""Listing 10.1 - The Tesselo graph: follows, brands, sellers and a knowledge graph (section 10.2).

Chapter 2's logs hold users, items and creators, but no follows, and Chapter 3 gave
every item a category, a language and a title. This file adds what Chapter 10
needs and the logs do not have:

  * user follows   - each user follows other users. Who follows whom depends on
                     similar hidden tastes and a shared app language, plus a share
                     of follows at random; a few users attract many followers.
  * creator follows - each user follows a few creators, more often those whose
                     videos match the user's taste on average
  * follow times   - 70% of follows were made before the month began; the rest at
                     a random moment during it, and they count only from then
  * private accounts - 10% of users; their follow edges are never used to
                     recommend anything to anyone else
  * brands, sellers - each product gets a brand (12 per product category, chosen
                     by the product's hidden vector, 20% at random) and a seller
                     (80 sellers, each carrying a few brands)
  * knowledge graph - triples (item, relation, entity) for an item's title words,
                     category, language and creator, and a product's brand and
                     seller

It uses its own random generator (seed cfg.seed + 1000) and changes nothing that
Chapters 2 to 9 use: the logs, the attributes and every earlier printed number stay
the same. The follows are not used by the simulated users: no one clicks a video
because a friend did, so the follow graph can help only because friends have
similar tastes. Every number is a toy setting chosen to make the effects visible,
not a measurement. Hidden truth keeps a leading "_".

Needs Python 3.12 or later. Run from the repository root (under a minute):
    python chapter_10/tesselo_graph.py
"""
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _ch in ("chapter_02", "chapter_03"):
    sys.path.insert(0, str(ROOT / _ch))
from tesselo_shop import CATEGORIES, DAY, LANGUAGES, PRODUCT_CATEGORIES, Config, build, vocabulary  # noqa: E402

RELATIONS = ("word", "category", "language", "creator", "brand", "seller")


@dataclass
class GraphConfig:
    follows: float = 12.0          # mean number of users a user follows
    creator_follows: float = 4.0   # mean number of creators a user follows
    homophily: float = 5.0         # P(follow v) grows with exp(5 x taste match of u and v)
    same_language: float = 4.0     # ... and 4 times more within the same app language
    random_share: float = 0.3      # share of follows made at random (by attractiveness only)
    before_month: float = 0.7      # share of follows made before day 0
    private_share: float = 0.10
    brands_per_category: int = 12
    sellers: int = 80
    seed_offset: int = 1000


def follow_users(cfg, G, users, rng):
    """Listing 10.1. Who follows whom: similar tastes, the same language, and a few
    accounts that attract many followers; a share of follows at random."""
    n = len(users["vec"])
    attract = rng.lognormal(0.0, 1.0, n)                         # some accounts are followed far more
    taste = users["vec"] @ users["vec"].T                        # hidden match of every pair
    lang = np.where(users["language"][:, None] == users["language"][None, :], G.same_language, 1.0)
    src, dst = [], []
    for u in range(n):
        k = min(1 + rng.poisson(G.follows - 1), n - 1)
        w = attract * np.exp(G.homophily * taste[u]) * lang[u]
        w[u] = 0.0
        n_rand = rng.binomial(k, G.random_share)                 # follows at random
        pick = rng.choice(n, k - n_rand, replace=False, p=w / w.sum())
        w_rand = attract.copy()
        w_rand[np.r_[pick, u]] = 0.0
        pick = np.r_[pick, rng.choice(n, n_rand, replace=False, p=w_rand / w_rand.sum())]
        src.append(np.full(len(pick), u))
        dst.append(pick)
    return np.concatenate(src), np.concatenate(dst)


def follow_creators(cfg, G, users, items, rng):
    """Each user follows a few creators, more often those whose videos match the
    user's taste on average."""
    video = ~items["is_product"]
    n_c = cfg.num_creators
    mean = np.zeros((n_c, items["vec"].shape[1]))
    np.add.at(mean, items["creator"][video], items["vec"][video])
    mean /= np.maximum(np.linalg.norm(mean, axis=1, keepdims=True), 1e-12)   # each creator's average direction
    size = np.bincount(items["creator"][video], minlength=n_c)
    src, dst = [], []
    for u in range(len(users["vec"])):
        k = min(rng.poisson(G.creator_follows), n_c)
        if k == 0:
            continue
        w = size * np.exp(G.homophily * (mean @ users["vec"][u]))
        src.append(np.full(k, u))
        dst.append(rng.choice(n_c, k, replace=False, p=w / w.sum()))
    return np.concatenate(src), np.concatenate(dst)


def follow_times(cfg, G, m, rng):
    """When each follow was made: before day 0 (time 0), or at a random moment of the month."""
    late = rng.random(m) >= G.before_month
    return np.where(late, rng.random(m) * cfg.days * DAY, 0.0)


def brands_and_sellers(cfg, G, items, rng):
    """A brand for every product, by its hidden vector within its category (a share at
    random), and a seller that carries that brand. -1 for videos."""
    n_cat = len(PRODUCT_CATEGORIES)
    n_v = len(CATEGORIES) - n_cat
    prod = np.flatnonzero(items["is_product"])
    brand = np.full(len(items["vec"]), -1)
    dirs = rng.normal(size=(n_cat, G.brands_per_category, items["vec"].shape[1]))
    dirs /= np.linalg.norm(dirs, axis=2, keepdims=True)
    for i in prod:
        c = items["category"][i] - n_v
        b = np.argmax(dirs[c] @ items["vec"][i]) if rng.random() >= 0.2 else rng.integers(G.brands_per_category)
        brand[i] = c * G.brands_per_category + b
    n_brands = n_cat * G.brands_per_category
    carries = [rng.choice(G.sellers, 1 + rng.integers(3), replace=False) for _ in range(n_brands)]
    seller = np.full(len(items["vec"]), -1)
    for i in prod:
        seller[i] = rng.choice(carries[brand[i]])
    return brand, seller, n_brands


def knowledge_graph(cfg, items, brand, seller):
    """Triples (item, relation, entity). Entities of each relation are numbered from 0."""
    word_id = {w: k for k, w in enumerate(vocabulary())}
    h, r, t = [], [], []
    for i in range(len(items["vec"])):
        for w in set(items["title"][i]):
            h.append(i), r.append(0), t.append(word_id[w])
        h += [i, i, i]
        r += [1, 2, 3]
        t += [int(items["category"][i]), int(items["language"][i]), int(items["creator"][i])]
        if brand[i] >= 0:
            h += [i, i]
            r += [4, 5]
            t += [int(brand[i]), int(seller[i])]
    return np.array(h), np.array(r), np.array(t)


def add_graph(cfg, users, items, G=GraphConfig()):
    """Everything Chapter 10 adds, from its own random generator."""
    rng = np.random.default_rng(cfg.seed + G.seed_offset)
    uf_src, uf_dst = follow_users(cfg, G, users, rng)
    cf_src, cf_dst = follow_creators(cfg, G, users, items, rng)
    brand, seller, n_brands = brands_and_sellers(cfg, G, items, rng)
    kg = knowledge_graph(cfg, items, brand, seller)
    return {"user_follows": {"src": uf_src, "dst": uf_dst, "t": follow_times(cfg, G, len(uf_src), rng)},
            "creator_follows": {"src": cf_src, "dst": cf_dst, "t": follow_times(cfg, G, len(cf_src), rng)},
            "private": rng.random(len(users["vec"])) < G.private_share,
            "brand": brand, "seller": seller, "n_brands": n_brands, "n_sellers": G.sellers,
            "kg": {"head": kg[0], "relation": kg[1], "tail": kg[2]},
            "n_entities": [len(vocabulary()), len(CATEGORIES), len(LANGUAGES), cfg.num_creators,
                           n_brands, G.sellers]}


if __name__ == "__main__":
    import os
    import platform
    print(f"Environment: Python {platform.python_version()}, NumPy {np.__version__}, {platform.system()} "
          f"{platform.machine()}, {os.cpu_count()} CPUs")
    cfg = Config()
    users, items, served, events, orders, truth = build(cfg)
    g = add_graph(cfg, users, items)
    uf, cf = g["user_follows"], g["creator_follows"]
    n = cfg.num_users
    out_deg, in_deg = np.bincount(uf["src"], minlength=n), np.bincount(uf["dst"], minlength=n)
    print(f"User follows: {len(uf['src']):,}; per user: mean {out_deg.mean():.1f}, median {np.median(out_deg):.0f}, "
          f"max {out_deg.max()}")
    top = np.sort(in_deg)[::-1]
    print(f"Followers: median {np.median(in_deg):.0f}, max {in_deg.max()}; the most-followed 1% of users have "
          f"{top[:n // 100].sum() / top.sum():.1%} of all follows")
    match = (users["vec"][uf["src"]] * users["vec"][uf["dst"]]).sum(axis=1)
    rng = np.random.default_rng(0)
    a, b = rng.integers(0, n, 100_000), rng.integers(0, n, 100_000)
    rand = (users["vec"][a] * users["vec"][b]).sum(axis=1)
    print(f"Hidden taste match: followed pairs {match.mean():.3f}, random pairs {rand.mean():.3f}")
    same = users["language"][uf["src"]] == users["language"][uf["dst"]]
    print(f"Same app language: followed pairs {same.mean():.1%}, random pairs "
          f"{(users['language'][a] == users['language'][b]).mean():.1%}")
    print(f"Creator follows: {len(cf['src']):,}; users who follow no creator: "
          f"{1 - len(np.unique(cf['src'])) / n:.1%}")
    print(f"Follows made during the month: users {(uf['t'] > 0).mean():.1%}, creators {(cf['t'] > 0).mean():.1%}")
    print(f"Private accounts: {g['private'].mean():.1%}")
    prod = items["is_product"]
    print(f"Brands: {g['n_brands']}, sellers: {g['n_sellers']}; products per brand: mean "
          f"{prod.sum() / g['n_brands']:.1f}")
    rel = np.bincount(g["kg"]["relation"], minlength=len(RELATIONS))
    print("Knowledge-graph triples: " + ", ".join(f"{name} {c:,}" for name, c in zip(RELATIONS, rel))
          + f"; total {rel.sum():,}")
