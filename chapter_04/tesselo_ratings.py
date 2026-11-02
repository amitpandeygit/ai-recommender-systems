"""Tesselo star ratings (Chapter 4, section 4.3).

Adds 1-to-5 star ratings that users give to products they bought, on top of the
Chapter 3 shop orders. It uses its own random generator (seed cfg.seed + 400), so
the Chapter 2 logs and Chapter 3 orders do not change.

How a rating is made (all hidden, for checking):
  * utility  = 3.6 + user bias + product quality + 2 * hidden match + noise
               user bias ~ N(0, 0.4): some users rate everything higher
               quality   ~ N(0, 0.5): some products are simply better
  * stars    = utility rounded and clipped to 1..5
  * who rates: a buyer rates with probability 0.10 + 0.25 * |utility - 3.6|,
               capped at 0.8. Strong feelings, good or bad, are more likely to
               be written down. This is the self-selection of Chapter 2: the
               ratings are not missing at random.
  * when:      1 to 5 days after the order.

Every number is a toy setting chosen to make effects visible, not a measurement.

Needs Python 3.12 or later. Run from the repository root:
    python chapter_04/tesselo_ratings.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "chapter_03"))
from tesselo_shop import DAY, Config, build  # noqa: E402

MEAN = 3.6


def make_ratings(cfg, users, items, orders):
    rng = np.random.default_rng(cfg.seed + 400)
    user_bias = rng.normal(0, 0.4, len(users["vec"]))
    quality = rng.normal(0, 0.5, len(items["vec"]))
    u, i = orders["user_id"], orders["item_id"]
    match = np.einsum("nd,nd->n", users["vec"][u], items["vec"][i])
    utility = MEAN + user_bias[u] + quality[i] + 2.0 * match + rng.normal(0, 0.6, len(u))
    stars = np.clip(np.rint(utility), 1, 5)
    p_rate = np.minimum(0.10 + 0.25 * np.abs(utility - MEAN), 0.8)
    rated = rng.random(len(u)) < p_rate
    time = orders["time"] + rng.uniform(1, 5, len(u)) * DAY
    keep = rated & (time < cfg.days * DAY)
    ratings = {"user_id": u[keep], "item_id": i[keep], "stars": stars[keep], "time": time[keep]}
    truth = {"user_bias": user_bias, "quality": quality, "all_stars": stars}
    return ratings, truth


def build_with_ratings(cfg=Config()):
    users, items, served, events, orders, truth = build(cfg)
    ratings, rtruth = make_ratings(cfg, users, items, orders)
    truth.update(rtruth)
    return users, items, served, events, orders, ratings, truth


if __name__ == "__main__":
    cfg = Config()
    users, items, served, events, orders, ratings, truth = build_with_ratings(cfg)
    n = len(ratings["stars"])
    print(f"Ratings: {n:,} from {len(np.unique(ratings['user_id'])):,} users on "
          f"{len(np.unique(ratings['item_id'])):,} products ({n / len(orders['item_id']):.1%} of order lines)")
    print("Stars given:   " + "  ".join(f"{s}: {np.mean(ratings['stars'] == s):.1%}" for s in range(1, 6)))
    print("Stars if every buyer had rated: "
          + "  ".join(f"{s}: {np.mean(truth['all_stars'] == s):.1%}" for s in range(1, 6)))
    print(f"Mean stars given {ratings['stars'].mean():.3f}; mean if everyone had rated "
          f"{truth['all_stars'].mean():.3f}")
