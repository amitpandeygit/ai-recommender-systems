"""Sections 7.3, 7.4 and 7.9 - Tesselo's heavy ranker, tested honestly.

  7.3  training rows from impressions, the features, and the shallow position tower
  7.4  eight ways to learn interactions, on the same rows and features
  7.9  the honest test of Chapter 5: the time split, the fixed baselines, the
       guardrails, every language, the simulated online test, and the candidate
       pool behind the ranker (Chapter 6's iALS pool against a mixed pool)

The ranker reorders a pool of 500 candidates per user (Chapter 6's chosen pool,
iALS filled by popular in your language) and shows the top 10. Its score is
P(click) x P(satisfied play | click).

Validation week: train on impressions of days 9 to 16 (features as at day 9),
choose the number of epochs by the log loss on days 16 to 23 (features as at day
16), and the architecture by NDCG@10 on the validation week. Test week: retrain on
days 9 to 23 and test once on days 23 to 30, with features as at day 23.

Needs Python 3.12 or later and PyTorch. Run from the repository root (about 11 minutes):
    python chapter_07/heavy_ranking.py
"""
import os
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "2"              # NumPy's linear algebra: the same thread count on every run
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "chapter_06"))
from ranking_data import (DAY, DENSE, LANGUAGES, NO, Config, build, candidate_pools,  # noqa: E402
                          impressions, snapshot)
from rankers import (ARCHITECTURES, Ranker, meta_for, predict, rerank, score_pools,  # noqa: E402
                     tensors, train)
from similarity_recs import (cf_scores, group_popularity, item_features, item_neighbours,  # noqa: E402
                             test_positives, top_k)
from implicit_scores import prepare  # noqa: E402
from evaluate import expected_outcomes, ranked, ranking_table  # noqa: E402
from metrics import ece, log_loss, ndcg_at_k  # noqa: E402
from two_tower import guardrails, recall_table  # noqa: E402
from retrieval_channels import merge, sources_at  # noqa: E402

POOL = 500
EPOCHS = (1, 2, 3, 4, 5, 6)
# The ranker chosen on the validation week by the main block below (which checks that it
# picks this one). The other Chapter 7 scripts import it.
SHIP = {"arch": "mlp, no ids", "epochs": 6}


def rows_for(events, served, users, items, weeks):
    """Impression rows for each (snapshot day, end day) week, tagged with the snapshot index."""
    parts = []
    for s, (a, e) in enumerate(weeks):
        r = impressions(events, served, users, items, a * DAY, e * DAY, e * DAY)
        r["snap"] = np.full(len(r["user"]), s)
        parts.append(r)
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def impression_auc(p, y):
    """Chance that a random positive row scores above a random negative one (ties half)."""
    p, y = np.asarray(p, float), np.asarray(y, bool)
    order = np.argsort(p, kind="stable")
    ranks = np.empty(len(p))
    ranks[order] = np.arange(1, len(p) + 1)
    _, inv, cnt = np.unique(p, return_inverse=True, return_counts=True)       # average ranks for ties
    first = np.cumsum(cnt) - cnt
    ranks = (first[inv] + (cnt[inv] + 1) / 2)
    n1, n0 = y.sum(), (~y).sum()
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def full_order(top, chain):
    """The reranked top of the pool, then the rest of the chain's order, for metrics
    that read the whole list (Chapter 5's ranking_table)."""
    out = np.full(chain.shape, -1)
    for r in range(len(top)):
        head = [i for i in top[r] if i >= 0]
        rest = [i for i in chain[r] if i >= 0 and i not in set(head)]
        lst = (head + rest)[: chain.shape[1]]
        out[r, :len(lst)] = lst
    return out


def epoch_losses(meta, arch, b_tr, R_tr, b_va, R_va, seed=0):
    """Train for up to max(EPOCHS) epochs; after each, the validation log loss of
    the click head (all rows, true slot) plus the play head (clicked rows)."""
    model = Ranker(meta, arch)
    out = []
    for _ in EPOCHS:
        train(model, b_tr, R_tr, epochs=1, seed=seed + len(out))
        pc, pp = predict(model, b_va, len(R_va["click"]), position=torch.as_tensor(R_va["position"]))
        c = R_va["click"]
        out.append(log_loss(pc, c) + log_loss(pp[c], R_va["play"][c]))
    return out, model


if __name__ == "__main__":
    t0 = time.time()
    cfg = Config()
    users, items, served, events, orders, truth = build(cfg)
    n_users, n_items = cfg.num_users, cfg.num_items
    video = ~items["is_product"]
    snaps = {d: snapshot(events, users, items, d * DAY) for d in (9, 16, 23)}
    week1 = rows_for(events, served, users, items, [(9, 16)])
    weeks12 = rows_for(events, served, users, items, [(9, 16), (16, 23)])
    week2 = rows_for(events, served, users, items, [(16, 23)])
    week3 = rows_for(events, served, users, items, [(23, 30)])
    print("Section 7.3: training rows are impressions (on screen, or clicked) of videos, for consenting, "
          "logged-in users")
    for name, R in (("days 9 to 16", week1), ("days 16 to 23", week2), ("days 23 to 30", week3)):
        print(f"  {name}: {len(R['click']):,} rows, {R['click'].sum():,} clicks ({R['click'].mean():.1%}), "
              f"{R['play'].sum():,} satisfied plays ({R['play'][R['click']].mean():.1%} of clicks), "
              f"{len(np.unique(R['user'])):,} users, {len(np.unique(R['item'])):,} videos")
    b1 = tensors([snaps[9]], users, items, week1["user"], week1["item"], week1["snap"])
    b2 = tensors([snaps[16]], users, items, week2["user"], week2["item"], week2["snap"])
    b12 = tensors([snaps[9], snaps[16]], users, items, weeks12["user"], weeks12["item"], weeks12["snap"])
    b3 = tensors([snaps[23]], users, items, week3["user"], week3["item"], week3["snap"])
    meta1 = meta_for(users, items, b1["dense"].numpy())
    meta12 = meta_for(users, items, b12["dense"].numpy())
    print("  dense features, mean over days 9 to 23 rows: "
          + ", ".join(f"{k} {v:.2f}" for k, v in zip(DENSE, meta12['dense_mean'])))
    print(f"[{time.time() - t0:.0f} s]")

    # ---- validation week: epochs by log loss, architecture by NDCG@10
    S16 = snaps[16]
    pos16 = test_positives(events, users, 16 * DAY, 23 * DAY, S16["used"], video)
    r16 = np.array(sorted(pos16))
    pools16 = candidate_pools(S16, users, items, r16, POOL)
    print(f"\nValidation week: {len(r16):,} users with a relevant video; trained on days 9 to 16")
    print(f"  {'':<14}" + "".join(f"{'loss, ' + str(e) + ' ep.':>14}" for e in EPOCHS)
          + f"{'chosen':>8}{'NDCG@10':>9}")
    chosen = {}
    for arch in ARCHITECTURES:
        losses, _ = epoch_losses(meta1, arch, b1, week1, b2, week2)
        ep = EPOCHS[int(np.argmin(np.round(losses, 9)))]
        m = train(Ranker(meta1, arch), b1, week1, epochs=ep)
        top = rerank(pools16, score_pools(m, S16, users, items, r16, pools16), 10)
        nd = ranking_table(full_order(top, pools16), r16, pos16, S16["used"], video)["NDCG@10"]
        chosen[arch] = (ep, nd)
        print(f"  {arch:<14}" + "".join(f"{x:>14.4f}" for x in losses) + f"{ep:>8}{nd:>9.4f}")
    ship = max(chosen, key=lambda a: round(chosen[a][1], 9))
    print(f"Chosen architecture: {ship}, {chosen[ship][0]} epoch(s); equals SHIP: "
          f"{SHIP == {'arch': ship, 'epochs': chosen[ship][0]}}   [{time.time() - t0:.0f} s]")

    # ---- test week
    S = snaps[23]
    weight, used = S["weight"], S["used"]
    pos = test_positives(events, users, 23 * DAY, 30 * DAY, used, video)
    rows = np.array(sorted(pos))
    aud = np.flatnonzero(users["current_consent"])
    hist = (weight > 0).any(axis=1)
    known = np.flatnonzero(users["current_consent"] & hist)
    print(f"\nTest week: {len(rows):,} test users with a relevant video; of the {len(known):,} consenting users "
          f"with history, {len(known) - np.isin(known, rows).sum():,} had none and are left out of offline "
          f"averages; online audience {len(aud):,} users")
    ok = S["ok"]
    n_creators = int(items["creator"].max()) + 1
    engaged = (weight > 0).sum(axis=0)
    vc = np.unique(items["creator"][video])                              # Chapter 5's creator groups
    cpop = np.bincount(items["creator"][video], weights=engaged[video].astype(float), minlength=n_creators)
    top_creators = set(vc[np.argsort(-cpop[vc], kind="stable")][: len(vc) // 10].tolist())
    pools_rows = candidate_pools(S, users, items, rows, POOL)
    pools_aud = candidate_pools(S, users, items, aud, POOL)
    pos_in_pool = np.mean([len(pos[u] & set(p.tolist())) / len(pos[u]) for u, p in zip(rows, pools_rows)])
    print(f"  pool of {POOL}: recall of the test users' relevant videos {pos_in_pool:.3f}")

    # Chapter 5's fixed baselines and Chapter 6's two-tower model
    feats = item_features(items, n_items)
    w2, used2, src = sources_at(events, users, items, 23 * DAY, n_users, n_items, video, feats)
    assert np.array_equal(w2, weight) and np.array_equal(used2, used)
    lang_pop = lambda r: S["lang_pop"][users["language"][r]]                # noqa: E731
    pop = group_popularity(weight, np.zeros(n_users, dtype=np.int64), 1)[0]
    d23 = prepare(events, users, 23 * DAY)
    views = np.bincount(d23["item"][d23["event"] == "viewable"], minlength=n_items).astype(float)
    clicks = np.bincount(d23["item"][d23["event"] == "click"], minlength=n_items).astype(float)
    ctr = (clicks + 20 * clicks.sum() / views.sum()) / (views + 20)
    row = lambda v: (lambda r: np.broadcast_to(v, (len(r), n_items)))        # noqa: E731
    shuffle = lambda r: np.array([np.random.default_rng([cfg.seed, 504, int(u)]).random(n_items)  # noqa: E731
                                  for u in r])
    sim = item_neighbours(weight, k=50, strength=0.0)
    baselines = {"random (eligible videos)": lambda r: np.where(ok, shuffle(r), NO),
                 "most clicked (click-through rate)": row(np.where(ok, ctr, NO)),
                 "popular worldwide": row(pop),
                 "popular in your language": lang_pop,
                 "CF, then popular in language": lambda r: (cf_scores(weight[r], sim), lang_pop(r)),
                 "iALS, then popular in language": lambda r: (src["iALS"](r), lang_pop(r)),
                 "two-tower (Chapter 6), then language": lambda r: (src["two-tower"](r), lang_pop(r))}
    results, per_user = {}, {}
    n_video = int(video.sum())

    def record(name, full, top_aud):
        t = ranking_table(full, rows, pos, used, video)
        t["R@100"] = recall_table(full, rows, pos, (100,))[100]
        t["cov"], t["creators"] = guardrails(top_aud, ok, items["creator"], top_creators, n_items)
        plays_aud = expected_outcomes(top_aud, aud, users, items, cfg)[1].sum(axis=1)
        t["online"] = plays_aud.mean()
        t["online_test"] = expected_outcomes(full[:, :10], rows, users, items, cfg)[1].sum(axis=1).mean()
        results[name] = t
        per_user[name] = plays_aud

    for name, fn in baselines.items():
        record(name, ranked(fn, rows, used, video, k=n_video), ranked(fn, aud, used, video, k=10))
    chain_rows = ranked(baselines["iALS, then popular in language"], rows, used, video, k=n_video)
    assert np.array_equal(chain_rows[:, :POOL], pools_rows)              # the pool is the chain's top 500
    print(f"Baselines rebuilt  [{time.time() - t0:.0f} s]")

    models = {}
    for arch in ARCHITECTURES:
        m = models[arch] = train(Ranker(meta12, arch), b12, weeks12, epochs=chosen[arch][0])
        top_r = rerank(pools_rows, score_pools(m, S, users, items, rows, pools_rows), 10)
        top_a = rerank(pools_aud, score_pools(m, S, users, items, aud, pools_aud), 10)
        record(f"ranker: {arch}", full_order(top_r, chain_rows), top_a)
    print(f"Rankers trained and scored  [{time.time() - t0:.0f} s]")

    print(f"\nSections 7.4 and 7.9: test week, video row, {len(rows):,} test users; each ranker reorders the "
          f"pool of {POOL}")
    print(f"  {'':<40}{'hit@10':>7}{'NDCG@10':>8}{'R@100':>7}{'coverage':>9}{'top-10% cr.':>12}"
          f"{'online':>8}{'online, test':>13}")
    for name, t in results.items():
        print(f"  {name:<40}{t['hit@10']:>7.3f}{t['NDCG@10']:>8.3f}{t['R@100']:>7.3f}{t['cov']:>9.1%}"
              f"{t['creators']:>12.1%}{t['online']:>8.4f}{t['online_test']:>13.4f}")
    print("  (coverage and top-10% creators on the lists all consenting users would see; online = expected "
          "satisfied plays per list)")
    base = per_user["iALS, then popular in language"]
    print("  online, paired difference against iALS (standard error), all consenting users; then the same "
          "ranker trained from seeds 1 and 2:")
    for arch in ARCHITECTURES:
        d = per_user[f"ranker: {arch}"] - base
        other = []
        for seed in (1, 2):
            ms = train(Ranker(meta12, arch, seed=seed), b12, weeks12, epochs=chosen[arch][0], seed=seed)
            top_a = rerank(pools_aud, score_pools(ms, S, users, items, aud, pools_aud), 10)
            other.append(expected_outcomes(top_a, aud, users, items, cfg)[1].sum(axis=1).mean())
        print(f"    {arch:<14} {d.mean():+.4f} ({d.std(ddof=1) / np.sqrt(len(d)):.4f})   online, seeds 0, 1, 2: "
              f"{results[f'ranker: {arch}']['online']:.4f}, {other[0]:.4f}, {other[1]:.4f}")

    # ---- held-out impressions, days 23 to 30: what a scoring model is judged on (Chapter 5)
    c3, x3 = week3["click"], week3["explored"]
    print(f"\nHeld-out impressions, days 23 to 30 ({len(c3):,} rows): click head at the true slot, play head on "
          f"clicked rows")
    print(f"  {'':<14}{'click AUC':>10}{'log loss':>9}{'ECE':>8}{'mean p':>8}{'play AUC':>9}")
    for arch, m in models.items():
        pc, pp = predict(m, b3, len(c3), position=torch.as_tensor(week3["position"]))
        print(f"  {arch:<14}{impression_auc(pc, c3):>10.3f}{log_loss(pc, c3):>9.4f}{ece(pc, c3):>8.4f}"
              f"{pc.mean():>8.4f}{impression_auc(pp[c3], week3['play'][c3]):>9.3f}")
    print(f"  observed click rate {c3.mean():.4f}; exploratory slots (random videos): {x3.sum():,} rows, "
          f"{week3['play'][x3].sum()} satisfied plays among them")
    shown = lambda top: S["engaged"][top[top >= 0]]                         # noqa: E731
    iall = ranked(baselines["iALS, then popular in language"], aud, used, video, k=10)
    rk = rerank(pools_aud, score_pools(models[ship], S, users, items, aud, pools_aud), 10)
    print(f"  videos shown to all consenting users, users engaged with each before day 23: iALS median "
          f"{np.median(shown(iall)):.0f}, ranker ({ship}) median {np.median(shown(rk)):.0f}")

    # ---- the shipped ranker: every language, three seeds, the position tower
    m = models[ship]
    print(f"\nShipped ranker ({ship}), NDCG@10 and online by user language: "
          + "  ".join(f"{LANGUAGES[g]} ({(users['language'][rows] == g).sum()})" for g in range(len(LANGUAGES))))
    top_ship = rerank(pools_rows, score_pools(m, S, users, items, rows, pools_rows), 10)
    nd_ship = [ndcg_at_k(t.tolist(), {i: 1.0 for i in pos[u]}) for u, t in zip(rows, top_ship)]
    nd_ials = [ndcg_at_k(t[:10].tolist(), {i: 1.0 for i in pos[u]}) for u, t in zip(rows, chain_rows)]
    lang_r, lang_a = users["language"][rows], users["language"][aud]
    for label, vals, group in (("NDCG@10, iALS chain", nd_ials, lang_r), ("NDCG@10, ranker", nd_ship, lang_r),
                               ("online, iALS chain", base, lang_a),
                               ("online, ranker", per_user[f"ranker: {ship}"], lang_a)):
        vals = np.asarray(vals)
        print(f"  {label:<24}" + "".join(f"{vals[group == g].mean():>9.4f}" for g in range(len(LANGUAGES))))
    d = np.asarray(nd_ship) - np.asarray(nd_ials)
    print("  NDCG@10 difference     " + "".join(
        f"{d[lang_r == g].mean():>+9.4f}" for g in range(len(LANGUAGES))))
    print("    standard error       " + "".join(
        f"{d[lang_r == g].std(ddof=1) / np.sqrt((lang_r == g).sum()):>9.4f}" for g in range(len(LANGUAGES))))
    print(f"  NDCG@10, ranker minus iALS, all test users: {d.mean():+.4f} (standard error "
          f"{d.std(ddof=1) / np.sqrt(len(d)):.4f})")


    no_pos = train(Ranker(meta12, ship, position=False), b12, weeks12, epochs=chosen[ship][0])
    top_a = rerank(pools_aud, score_pools(no_pos, S, users, items, aud, pools_aud), 10)
    print(f"\nSection 7.3: the shallow position tower. Online, with the tower "
          f"{results[f'ranker: {ship}']['online']:.4f}; "
          f"without position as an input {expected_outcomes(top_a, aud, users, items, cfg)[1].sum(axis=1).mean():.4f}")
    # Each slot's click chance relative to slot 1: the model's (mean over held-out rows of
    # P(click | slot k) / P(click | slot 1)), the exploratory slots' (click rate of random
    # videos in slot k over slot 1, days 0 to 23), and the truth (P(look) = 1 / k ** 0.5)
    ratio = []
    p1 = predict(m, b3, len(c3), position=torch.ones(len(c3), dtype=torch.long))[0]
    for k in range(1, 11):
        pk = predict(m, b3, len(c3), position=torch.full((len(c3),), k, dtype=torch.long))[0]
        ratio.append(float(np.mean(pk / p1)))
    early = impressions(events, served, users, items, 0, 23 * DAY, 23 * DAY)
    ex = early["explored"]
    ctr_k = np.array([early["click"][ex & (early["position"] == k)].mean() for k in range(1, 11)])
    n_k = [int((ex & (early["position"] == k)).sum()) for k in range(1, 11)]
    true = 1 / np.arange(1, 11) ** cfg.look_decay
    print("  click chance relative to slot 1")
    print("  slot:                       " + "".join(f"{k:>7}" for k in range(1, 11)))
    print("  learned tower               " + "".join(f"{x:>7.3f}" for x in ratio))
    print("  exploratory slots, days 0-23" + "".join(f"{x:>7.3f}" for x in ctr_k / ctr_k[0]))
    print("  truth, 1 / slot ** 0.5      " + "".join(f"{x:>7.3f}" for x in true))
    print(f"  exploratory rows per slot: {min(n_k):,} to {max(n_k):,}")

    # ---- the pool behind the ranker: Chapter 6's iALS pool against a mixed pool
    del b1, b2, b12, b3, models, no_pos                              # free memory for the pools below
    names = ("iALS", "two-tower", "item-based CF", "New for you", "popular in your language")
    print(f"\nSection 7.9: which pool should the ranker reorder? Pools of {POOL} for every consenting user")
    lists_aud = {n: top_k(src[n](aud), used[aud], video, POOL) for n in names}
    lists_rows = {n: top_k(src[n](rows), used[rows], video, POOL) for n in names}
    for quotas in ((500, 0, 0, 0, 0), (250, 250, 0, 0, 0), (250, 125, 75, 50, 0)):
        pa = np.array([merge([lists_aud[n][r] for n in names], quotas, pool=POOL)[0] for r in range(len(aud))])
        pr = [merge([lists_rows[n][r] for n in names], quotas, pool=POOL)[0] for r in range(len(rows))]
        rc = np.mean([len(pos[u] & set(p)) / len(pos[u]) for u, p in zip(rows, pr)])
        top_a = rerank(pa, score_pools(m, S, users, items, aud, pa), 10)
        plays = expected_outcomes(top_a, aud, users, items, cfg)[1].sum(axis=1)
        dd = plays - per_user[f"ranker: {ship}"]
        cov, cr = guardrails(top_a, ok, items["creator"], top_creators, n_items)
        print(f"  quotas {str(quotas):<22} pool recall {rc:.3f}   online {plays.mean():.4f} "
              f"(against Chapter 6's pool {dd.mean():+.4f}, standard error {dd.std(ddof=1) / np.sqrt(len(dd)):.4f})"
              f"   coverage {cov:.1%}   top-10% creators {cr:.1%}")
    print(f"[{time.time() - t0:.0f} s]")
