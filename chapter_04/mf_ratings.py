"""Listing 4.1 - Matrix factorisation with biases for star ratings (section 4.3).

Model:  predicted stars = mu + b_u + b_i + p_u . q_i
Trained with stochastic gradient descent (SGD) on the squared error plus an L2
penalty, as in the Netflix Prize work of Funk (2006) and Koren, Bell and
Volinsky (2009). The penalty sits inside the sum over ratings: each rating
shrinks the parameters it touches, so a user or item with many ratings is
shrunk more often.

Test: a time split, like Chapter 3. Settings are chosen on ratings given in
days 16 to 23 (training on earlier ratings), then the model is retrained on
everything before day 23 and tested once on days 23 to 30. Only ratings from
users who allowed personalisation, at the time and now, are used.

Needs Python 3.12 or later. Run from the repository root:
    python chapter_04/mf_ratings.py
"""
import numpy as np

from tesselo_ratings import DAY, Config, build_with_ratings


def sgd_step(r, mu, bu, bi, p, q, lr, reg):
    """One SGD update for one rating. Returns the new values and the error."""
    err = r - (mu + bu + bi + p @ q)
    bu_new = bu + lr * (err - reg * bu)
    bi_new = bi + lr * (err - reg * bi)
    p_new = p + lr * (err * q - reg * p)
    q_new = q + lr * (err * p - reg * q)                  # uses the old p
    return bu_new, bi_new, p_new, q_new, err


def train_mf(u, i, r, n_users, n_items, k=8, lr=0.01, reg=0.05, epochs=30, seed=0):
    """Biased matrix factorisation by SGD. k = 0 gives the biases-only model."""
    rng = np.random.default_rng(seed)
    mu = r.mean()
    bu, bi = np.zeros(n_users), np.zeros(n_items)
    p = rng.normal(0, 0.1, (n_users, k))
    q = rng.normal(0, 0.1, (n_items, k))
    for _ in range(epochs):
        for j in rng.permutation(len(r)):                 # a new random order every epoch
            a, b = u[j], i[j]
            bu[a], bi[b], p[a], q[b], _ = sgd_step(r[j], mu, bu[a], bi[b], p[a], q[b], lr, reg)
    p[np.bincount(u, minlength=n_users) == 0] = 0.0       # no training ratings: no taste vector,
    q[np.bincount(i, minlength=n_items) == 0] = 0.0       # so the prediction is mu + the biases
    return mu, bu, bi, p, q


def predict(model, u, i):
    mu, bu, bi, p, q = model
    return np.clip(mu + bu[u] + bi[i] + np.einsum("nk,nk->n", p[u], q[i]), 1, 5)


def rmse(pred, r):
    return float(np.sqrt(np.mean((pred - r) ** 2)))


def worked_example():
    out = sgd_step(5.0, 4.11, 0.2, 0.1, np.array([0.3, -0.1]), np.array([0.2, 0.4]), 0.01, 0.05)
    bu, bi, p, q, err = out
    print(f"Worked SGD step: error {err:.2f}; b_u {bu:.4f}; b_i {bi:.5f}; "
          f"p_u ({p[0]:.5f}, {p[1]:.5f}); q_i ({q[0]:.5f}, {q[1]:.5f})")


def split(ratings, users, start, end):
    ok = (ratings["time"] < users["consent_off_from"][ratings["user_id"]]) \
        & users["current_consent"][ratings["user_id"]]
    t = ratings["time"]
    train = ok & (t < start)
    test = ok & (t >= start) & (t < end)
    pick = lambda m: (ratings["user_id"][m], ratings["item_id"][m], ratings["stars"][m])
    return pick(train), pick(test)


if __name__ == "__main__":
    worked_example()
    cfg = Config()
    users, items, served, events, orders, ratings, truth = build_with_ratings(cfg)
    nu, ni = cfg.num_users, cfg.num_items

    # Choose k and the penalty on the validation ratings (days 16 to 23)
    (u, i, r), (vu, vi, vr) = split(ratings, users, 16 * DAY, 23 * DAY)
    print(f"\nValidation: {len(r):,} training ratings, {len(vr):,} validation ratings")
    best = None
    for k in (0, 4, 16):
        for reg in (0.02, 0.1):
            model = train_mf(u, i, r, nu, ni, k=k, reg=reg)
            e = rmse(predict(model, vu, vi), vr)
            print(f"  k = {k:>2}, penalty {reg:.2f}: RMSE {e:.4f}")
            if best is None or e < best[0]:
                best = (e, k, reg)
    print(f"Chosen: k = {best[1]}, penalty {best[2]}")

    # Final test on days 23 to 30
    (u, i, r), (tu, ti, tr) = split(ratings, users, 23 * DAY, 30 * DAY)
    print(f"\nTest: {len(r):,} training ratings, {len(tr):,} test ratings")
    per_user, per_item = np.bincount(u, minlength=nu), np.bincount(i, minlength=ni)
    print(f"  Training ratings come from {(per_user > 0).sum():,} users (median "
          f"{np.median(per_user[per_user > 0]):.0f} each; {(per_user >= 20).sum()} with 20 or more) "
          f"on {(per_item > 0).sum():,} products (median {np.median(per_item[per_item > 0]):.0f} each)")
    mean_only = np.full(len(tr), r.mean())
    biases = train_mf(u, i, r, nu, ni, k=0, reg=best[2])
    models = {f"biases + {k} factors": train_mf(u, i, r, nu, ni, k=k, reg=0.1) for k in (4, 16)}
    heavy = np.bincount(u, minlength=nu)[tu] >= 20            # test ratings from users with 20+ ratings
    print(f"  Test ratings from users with no training ratings: {(per_user[tu] == 0).sum():,}; "
          f"on products with none: {(per_item[ti] == 0).sum():,} (these fall back to mu + biases)")
    print(f"  {'':<22} {'all':>8} {'users with 20+':>15}")
    for name, pred in (("global mean", mean_only), ("biases only", predict(biases, tu, ti)),
                       *((n, predict(m, tu, ti)) for n, m in models.items())):
        print(f"  {name:<22} {rmse(pred, tr):>8.4f} {rmse(pred[heavy], tr[heavy]):>15.4f}")
    print(f"  (test ratings: {len(tr):,} in all, {heavy.sum():,} from users with 20+ training ratings)")
    full = biases

    # What the biases learned, against the hidden truth (simulation only)
    mu, bu, bi, p, q = full
    rated_items = np.unique(i)
    rated_users = np.unique(u)
    print(f"\nLearned item bias vs hidden quality, correlation: "
          f"{np.corrcoef(bi[rated_items], truth['quality'][rated_items])[0, 1]:.2f}")
    print(f"Learned user bias vs hidden user bias, correlation: "
          f"{np.corrcoef(bu[rated_users], truth['user_bias'][rated_users])[0, 1]:.2f}")

    # Self-selection: the same model on every test-week purchase, as if all buyers had rated
    t, ok = orders["time"], users["current_consent"][orders["user_id"]]
    wk = ok & (t >= 23 * DAY) & (t < 30 * DAY)
    ou, oi = orders["user_id"][wk], orders["item_id"][wk]
    allp = predict(full, ou, oi)
    print(f"Test-week purchases: {wk.sum():,}. Mean predicted stars {allp.mean():.2f}; "
          f"mean stars if every buyer had rated {truth['all_stars'][wk].mean():.2f}; "
          f"RMSE against those stars {rmse(allp, truth['all_stars'][wk]):.4f}")
