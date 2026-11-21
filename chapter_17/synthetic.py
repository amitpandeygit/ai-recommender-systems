"""Synthetic first interactions for launched products (section 17.5).

A launched product has no purchases, so the purchase graph knows nothing about it. A
language model can play the part of shoppers: shown the product and a few real
shoppers' recent purchases, it picks the ones who would buy it, and each pick becomes a
synthetic purchase edge in the graph that LightGCN + KG trains on (Listing 17.6). The
synthetic edges are used only to train that graph: they are kept out of the logs, the
test, and every other model.

For each of the 150 launched products, the candidates are 12 consenting shoppers with
purchases before day 23: those whose purchases are most similar to the product by
Chapter 3's content similarity. Each arm adds 3 synthetic buyers per product:

  no synthetic buyers  - LightGCN + KG as in section 10.5
  content filter       - the 3 candidates with the most similar purchases (no model)
  LLM shoppers         - the 3 the model picks from the 12 (enrich.WORKHORSE), shown each
                         candidate's 3 latest purchases
  hidden-taste reference, 12 candidates
                       - the 3 candidates whose hidden taste vector has the largest dot
                         product with the product's, a privileged reference only the
                         simulation has. It is not a perfect simulator: real purchases
                         also depend on how often a shopper buys, on chance, on
                         complements and, in the occasion shop, on occasions
  hidden-taste reference, anyone
                       - the same choice among all shoppers who bought before day 23

The test is section 10.5's launch test (NDCG@10 and recall@10, the 978 users who bought a
launched product in days 23 to 30), in Chapter 3's shop and in the shop with occasion
baskets (occasions.py), where what a product is for says who buys it. LightGCN + KG uses
section 10.5's 40 epochs, seeds 0, 1 and 2.

Needs llm_cache (or CH17_RUN=1 and the model). From the repository root:
    python chapter_17/synthetic.py
"""
import os
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "2"
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from occasions import with_occasions  # noqa: E402
from content_data import load  # noqa: E402
from graph_data import DAY, environment, paired  # noqa: E402
from graph_models import fit_lightgcn, graph_degrees, layer_vectors, request_vectors  # noqa: E402
from cold_items import LAYERS, PRODUCT_RELATIONS, kg_edges, launches, product_ranking, product_test  # noqa: E402
from similarity_recs import item_features  # noqa: E402
from title_words import WORDS  # noqa: E402
from listings import CATEGORIES  # noqa: E402
import enrich  # noqa: E402
import llm  # noqa: E402

torch.set_num_threads(2)
MODEL = os.environ.get("CH17_SYNTH_MODEL", enrich.WORKHORSE)
CANDIDATES, PICKS, RECENT = 12, 3, 3
SEEDS = (0, 1, 2)
EPOCHS = 40

SHOPPERS = """You simulate shoppers in an online shop. You are shown a product that has just been launched, and a list of shoppers, each with the products they bought recently. Choose the shoppers who are most likely to buy the new product, judging only from what they bought before: exactly 3 shoppers, by their labels."""


def describe(items, i):
    return f"{CATEGORIES[items['category'][i]]}: {', '.join(WORDS[str(t)][0] for t in items['title'][i])}"


def candidates(W, test, launched, feats):
    """For each launched product, the CANDIDATES shoppers with purchases before the cutoff
    whose purchase profile is most similar to it (Chapter 3's content similarity)."""
    pairs, bought = test[0], test[1]
    shoppers = np.array(sorted(bought))
    n_i = len(W["items"]["vec"])
    hist = np.zeros((len(shoppers), n_i))
    for k, u in enumerate(shoppers):
        hist[k, list(bought[int(u)])] = 1.0
    P = hist @ feats
    P /= np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-12)
    S = P @ feats[np.flatnonzero(launched)].T                  # shoppers x launched products
    out = {}
    for c, i in enumerate(np.flatnonzero(launched)):
        top = np.argsort(-S[:, c], kind="stable")[:CANDIDATES]
        out[int(i)] = [int(u) for u in shoppers[top]]
    return out


def recent(orders, users, cutoff, launched):
    """Each consenting shopper's purchases before the cutoff, latest first (launched products hidden)."""
    ok = users["current_consent"][orders["user_id"]] & (orders["time"] < cutoff) & ~launched[orders["item_id"]]
    out = {}
    for k in np.argsort(-orders["time"][ok], kind="stable"):
        u, i = int(orders["user_id"][ok][k]), int(orders["item_id"][ok][k])
        lst = out.setdefault(u, [])
        if i not in lst:
            lst.append(i)
    return out


def llm_picks(W, cands, history, world):
    """Listing 17.6. The model reads the product and each candidate's recent purchases and
    picks PICKS of them; the schema allows only the candidates' labels."""
    items = W["items"]
    reqs, keys = [], []
    labels = [f"S{k + 1}" for k in range(CANDIDATES)]
    schema = {"type": "object", "properties": {"buyers": {"type": "array", "minItems": PICKS, "maxItems": PICKS,
                                                          "items": {"type": "string", "enum": labels}}},
              "required": ["buyers"]}
    for i, us in cands.items():
        lines = [f"New product: {describe(items, i)}", "Shoppers:"]
        for lab, u in zip(labels, us):
            lines.append(f"{lab}: " + " | ".join(describe(items, j) for j in history[u][:RECENT]))
        reqs.append((SHOPPERS, "\n".join(lines), schema))
        keys.append(i)
    out = llm.ask_many(MODEL, f"synthetic_{world}", reqs, max_tokens=48)
    picks = {}
    for i, r in zip(keys, out):
        chosen = list(dict.fromkeys((r["answer"] or {"buyers": []})["buyers"]))
        picks[i] = [cands[i][labels.index(x)] for x in chosen]
    return picks, out


def scores(W, test, synthetic, seed):
    """LightGCN + KG on the purchase graph plus the synthetic edges."""
    users, items = W["users"], W["items"]
    pairs, bought, _, tu = test
    n_u, n_i = len(users["vec"]), len(items["vec"])
    prods = np.flatnonzero(items["is_product"])
    syn = np.array([(u, i) for i, us in synthetic.items() for u in us], dtype=np.int64).reshape(-1, 2)
    allp = np.vstack([pairs, syn]) if len(syn) else pairs
    rows, cols = allp[:, 0], allp[:, 1] + n_u
    e, n_ent = kg_edges(W, prods, PRODUCT_RELATIONS, n_u + n_i)
    rows, cols, n_nodes = np.r_[rows, e[0]], np.r_[cols, e[1]], n_u + n_i + n_ent
    deg = graph_degrees(rows, cols, n_nodes)
    m, A_hat = fit_lightgcn((rows, cols), n_nodes, n_u, prods + n_u, layers=LAYERS, epochs=EPOCHS, seed=seed)
    L_, final = layer_vectors(m, A_hat)
    nbrs = [np.array(sorted(bought.get(int(x), [])), dtype=np.int64) + n_u for x in tu]
    own = np.where((deg[tu] > 0)[:, None], L_[0][tu], 0.0)
    return request_vectors(L_, deg, nbrs, own) @ final[n_u:n_u + n_i].T


def run(W, orders, world, t0):
    users, items = W["users"], W["items"]
    launched = launches(W)
    feats = item_features(items, len(items["vec"]))
    test = product_test(W, orders, 23 * DAY, 30 * DAY, launched)
    tu = test[3]
    cands = candidates(W, test, launched, feats)
    history = recent(orders, users, 23 * DAY, launched)
    picks, recs = llm_picks(W, cands, history, world)
    filt = {i: us[:PICKS] for i, us in cands.items()}
    taste = {i: [us[k] for k in np.argsort(-(users["vec"][us] @ items["vec"][i]), kind="stable")[:PICKS]]
             for i, us in cands.items()}
    shoppers = np.array(sorted(test[1]))
    anyone = {i: [int(u) for u in shoppers[np.argsort(-(users["vec"][shoppers] @ items["vec"][i]), kind="stable")[:PICKS]]]
              for i in cands}
    arms = {"no synthetic buyers": {}, "content filter": filt, "LLM shoppers": picks,
            "hidden-taste ref., 12 cand.": taste, "hidden-taste ref., anyone": anyone}
    # how often each arm picks a candidate who really bought the product in the test week
    pos_users = {}
    for u, its in test[2].items():
        for i in its:
            pos_users.setdefault(i, set()).add(u)
    u_ = usage_line(recs)
    print(f"\n{world}: {len(tu):,} users bought a launched product in days 23 to 30. Candidates per product: "
          f"{CANDIDATES}; synthetic buyers per product: {PICKS}. {u_}  [{time.time() - t0:.0f} s]", flush=True)
    any_buyer = np.mean([bool(set(us) & pos_users.get(i, set())) for i, us in cands.items()])
    print(f"  products with a real test-week buyer among their {CANDIDATES} candidates: {any_buyer:.1%}")
    schema = {"type": "object", "properties": {"buyers": {"type": "array", "minItems": PICKS, "maxItems": PICKS,
                                                          "items": {"type": "string",
                                                                    "enum": [f"S{k + 1}" for k in range(CANDIDATES)]}}},
              "required": ["buyers"]}
    short = sum(len(v) < PICKS for v in picks.values())
    print(f"  LLM shoppers: {llm.failures(recs, [schema] * len(recs))}; {short} products got fewer than {PICKS} "
          f"distinct buyers (the model named a shopper twice)")
    res = {}
    for arm, syn in arms.items():
        hit = np.mean([len(set(us) & pos_users.get(i, set())) for i, us in syn.items()]) if syn else 0.0
        runs = [product_ranking(scores(W, test, syn, s), tu, launched, test[1], test[2]) for s in SEEDS]
        nd, rc = np.mean([r[0] for r in runs], axis=0), np.mean([r[1] for r in runs], axis=0)
        res[arm] = nd
        against = "" if arm == "no synthetic buyers" else "{:+.4f} ({:.4f}) [{:+.4f}, {:+.4f}]".format(
            *paired(nd, res["no synthetic buyers"], tu))
        print(f"  {arm:<28}{hit:>8.3f}{nd.mean():>9.4f}{rc.mean():>11.4f}   {against}  [{time.time() - t0:.0f} s]",
              flush=True)
    print("  LLM shoppers against content filter: {:+.4f} ({:.4f}) [{:+.4f}, {:+.4f}]".format(
        *paired(res["LLM shoppers"], res["content filter"], tu)))
    return res


def usage_line(recs):
    u = llm.usage(recs, MODEL)
    return (f"Model {llm.model_id(MODEL)}: {u['requests']} requests, {u['input'] / max(u['requests'], 1):,.0f} input "
            f"and {u['output'] / max(u['requests'], 1):.1f} output tokens each, {u['seconds'] / 60:.0f} minutes")


if __name__ == "__main__":
    from tesselo_shop import build
    t0 = time.time()
    print(environment())
    W = load()
    _, items_b, _, _, orders, truth = build(W["cfg"])
    assert np.array_equal(items_b["vec"], W["items"]["vec"])
    print(f"Columns: real test-week buyers among a product's {PICKS} synthetic buyers (mean), NDCG@10, recall@10, and "
          "the paired difference in NDCG@10 against no synthetic buyers")
    run(W, orders, "chapter3", t0)
    merged, _, _ = with_occasions(W, orders, truth["topup"])
    run(W, merged, "occasions", t0)
