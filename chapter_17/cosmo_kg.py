"""A commonsense knowledge graph from co-purchases, after Amazon's COSMO (section 17.4).

COSMO mines pairs of products that people buy together, asks a language model why
someone would buy both, filters the answers with a critic, groups them into intentions
and links products to the intentions. Here, at Tesselo's scale:

  1. pairs      - pairs of products from different categories bought in the same order by
                  consenting users before day 23, in at least 2 orders (the launched
                  products' orders hidden, as in section 10.5); 300 drawn at random
  2. generate   - the model gives each pair's shopping intent in a few words, or says
                  there is no common reason (Listing 17.5)
  3. critic     - the same model judges whether the intent is a typical reason to buy both
  4. intents    - the kept intents are embedded (Qwen3-Embedding) and grouped by
                  agglomerative clustering; each group is one intent node, named by its
                  most common phrase
  5. tag        - the model links each of the 60 product concepts to up to 2 intent nodes;
                  a product is linked to the intents of its title's concepts, so a launched
                  product, which has no purchases, is linked too
  6. test       - section 10.5's launch test with LightGCN + KG, with and without the intent
                  nodes, against the planted occasions as intent nodes (a reference only the
                  simulation has)

It runs twice: in the shop with occasion baskets (occasions.py) and in Chapter 3's shop
alone, the control, where baskets follow hidden tastes and random complements and no
word's meaning says anything about who buys what.

Needs llm_cache (or CH17_RUN=1 and the model) and Qwen3-Embedding for step 4. From the
repository root (about 15 minutes from the cache):
    python chapter_17/cosmo_kg.py
"""
import os
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "2"
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from listings import CATEGORIES, GENERIC_WORDS, N_VIDEO_CATEGORIES  # noqa: E402
from occasions import OCCASIONS, TOKEN, with_occasions  # noqa: E402
from content_data import load  # noqa: E402
from graph_data import DAY, environment, paired  # noqa: E402
from graph_models import fit_lightgcn, graph_degrees, layer_vectors, request_vectors  # noqa: E402
from cold_items import LAYERS, PRODUCT_RELATIONS, kg_edges, launches, product_ranking, product_test  # noqa: E402
from title_words import WORDS  # noqa: E402
import llm  # noqa: E402
import textvec  # noqa: E402

torch.set_num_threads(2)
MODEL = os.environ.get("CH17_COSMO_MODEL", "Qwen3.5-4B")
N_PAIRS, MIN_ORDERS = 300, 2
SEEDS = (0, 1, 2)
EPOCHS = 40                                         # section 10.5's validation choice for products
CLUSTER_DISTANCE = 0.35                             # cosine distance at which two groups merge
MIN_GROUP = 3
PRODUCT_CONCEPTS = [t for t in WORDS if t not in GENERIC_WORDS and
                    CATEGORIES.index(t.rsplit("_", 1)[0].replace("_", " ")) >= N_VIDEO_CATEGORIES]


def describe(items, i):
    """A product as the model sees it: its category and its title's concepts in English."""
    words = [WORDS[str(t)][0] for t in items["title"][i]]
    return f"{CATEGORIES[items['category'][i]]}: {', '.join(words)}"


# ---------------------------------------------------------------- 1. pairs
def co_purchases(W, orders, cutoff, launched, topup, rng):
    """Pairs of products of different categories in the same order, before the cutoff,
    by consenting users, in at least MIN_ORDERS orders; N_PAIRS drawn at random."""
    items = W["items"]
    ok = (W["users"]["current_consent"][orders["user_id"]] & (orders["time"] < cutoff)
          & ~launched[orders["item_id"]] & (orders["item_id"] != topup))
    o, i = orders["order_id"][ok], orders["item_id"][ok]
    s = np.lexsort((i, o))
    o, i = o[s], i[s]
    count = Counter()
    start = np.r_[0, np.flatnonzero(np.diff(o)) + 1, len(o)]
    for a, b in zip(start[:-1], start[1:]):
        basket = np.unique(i[a:b])
        for x in range(len(basket)):
            for y in range(x + 1, len(basket)):
                p, q = int(basket[x]), int(basket[y])
                if items["category"][p] != items["category"][q]:
                    count[(p, q)] += 1
    pairs = sorted(k for k, c in count.items() if c >= MIN_ORDERS)
    pick = rng.choice(len(pairs), min(N_PAIRS, len(pairs)), replace=False)
    return [pairs[k] for k in sorted(pick)], len(pairs)


# ---------------------------------------------------------------- 2 and 3. Listing 17.5: generate and criticise
GENERATE = """You study why people buy products together in an online shop. You are shown two products that one shopper bought in the same order. Give the most likely shopping intent that explains buying both, as a short phrase of 2 to 4 English words, such as "camping trip", "baking at home" or "new school year". If the two products have no common reason to be bought together, answer "no common reason". Do not explain.

Examples:
Product A: home and garden: tent, torch | Product B: kitchen: water bottle, plates
Answer: {"intent": "camping trip"}
Product A: office: printer paper | Product B: beauty: lipstick, nail polish
Answer: {"intent": "no common reason"}"""

CRITIC = """You check statements about why people buy products together in an online shop. You are shown two products and a proposed shopping intent. Answer true only if many shoppers would plausibly buy both products for that intent. Answer false if the intent fits only one of them, is vague, or is not a real reason to buy both."""

INTENT_SCHEMA = {"type": "object", "properties": {"intent": {"type": "string", "maxLength": 32}},
                 "required": ["intent"]}
CRITIC_SCHEMA = {"type": "object", "properties": {"typical": {"type": "boolean"}}, "required": ["typical"]}


def generate(items, pairs, world):
    reqs = [(GENERATE, f"Product A: {describe(items, a)} | Product B: {describe(items, b)}", INTENT_SCHEMA)
            for a, b in pairs]
    return [r["answer"]["intent"].strip().lower() if r["answer"] else "no common reason"
            for r in llm.ask_many(MODEL, f"cosmo_generate_{world}", reqs, max_tokens=24)]


def criticise(items, pairs, intents, world):
    reqs = [(CRITIC, f"Product A: {describe(items, a)} | Product B: {describe(items, b)} | Intent: {t}", CRITIC_SCHEMA)
            for (a, b), t in zip(pairs, intents)]
    return [bool(r["answer"]["typical"]) if r["answer"] else False
            for r in llm.ask_many(MODEL, f"cosmo_critic_{world}", reqs, max_tokens=12)]


# ---------------------------------------------------------------- 4. intents
def group_intents(phrases):
    """Agglomerative clustering of the phrases' unit vectors (average linkage, cosine
    distance); groups of at least MIN_GROUP phrases become intent nodes, each named by
    its most common phrase. Returns the names and each phrase's node (-1 for none)."""
    from scipy.cluster.hierarchy import fcluster, linkage
    uniq = sorted(set(phrases))
    if len(uniq) < 2:
        return [], np.full(len(phrases), -1)
    V = textvec.vectors(uniq)
    lab = fcluster(linkage(V, method="average", metric="cosine"), CLUSTER_DISTANCE, criterion="distance")
    of = dict(zip(uniq, lab))
    sizes = Counter(of[p] for p in phrases)
    keep = sorted((c for c, n in sizes.items() if n >= MIN_GROUP), key=lambda c: (-sizes[c], c))
    names = [Counter(p for p in phrases if of[p] == c).most_common(1)[0][0] for c in keep]
    node = {c: k for k, c in enumerate(keep)}
    return names, np.array([node.get(of[p], -1) for p in phrases])


# ---------------------------------------------------------------- 5. tag the concepts
TAG = """You link product types to shopping intents for an online shop. You are shown one product type and a list of shopping intents. Choose the intents that people commonly buy this product type for: at most two, or none if no intent in the list fits."""


def tag_concepts(names, world):
    schema = {"type": "object", "properties": {"intents": {"type": "array", "maxItems": 2,
                                                           "items": {"type": "string", "enum": names}}},
              "required": ["intents"]}
    reqs = [(TAG, f"Intents: {'; '.join(names)}\nProduct type: {WORDS[t][0]}", schema) for t in PRODUCT_CONCEPTS]
    out = llm.ask_many(MODEL, f"cosmo_tag_{world}", reqs, max_tokens=40)
    return {t: sorted({names.index(x) for x in (r["answer"] or {"intents": []})["intents"]})
            for t, r in zip(PRODUCT_CONCEPTS, out)}


def product_intents(items, concept_tags, n_nodes):
    """products x intent nodes: True where the product's title has a concept tagged with it."""
    M = np.zeros((len(items["vec"]), n_nodes), dtype=bool)
    for i in np.flatnonzero(items["is_product"]):
        for t in items["title"][i]:
            for k in concept_tags.get(str(t), []):
                M[i, k] = True
    return M


def planted(items):
    """The planted occasions as intent nodes: a product is linked to every occasion one of
    its concepts belongs to (occasions.membership)."""
    from occasions import membership
    return membership(items)


# ---------------------------------------------------------------- 6. the launch test
def launch_scores(W, test, extra_intents, seed):
    """LightGCN + KG on the purchase graph, optionally with product-intent edges."""
    users, items = W["users"], W["items"]
    pairs, bought, _, tu = test
    n_u, n_i = len(users["vec"]), len(items["vec"])
    prods = np.flatnonzero(items["is_product"])
    rows, cols = pairs[:, 0], pairs[:, 1] + n_u
    e, n_ent = kg_edges(W, prods, PRODUCT_RELATIONS, n_u + n_i)
    rows, cols, n_nodes = np.r_[rows, e[0]], np.r_[cols, e[1]], n_u + n_i + n_ent
    if extra_intents is not None:
        p, k = np.nonzero(extra_intents)
        rows, cols = np.r_[rows, p + n_u], np.r_[cols, n_nodes + k]
        n_nodes += extra_intents.shape[1]
    deg = graph_degrees(rows, cols, n_nodes)
    m, A_hat = fit_lightgcn((rows, cols), n_nodes, n_u, prods + n_u, layers=LAYERS, epochs=EPOCHS, seed=seed)
    L_, final = layer_vectors(m, A_hat)
    nbrs = [np.array(sorted(bought.get(int(x), [])), dtype=np.int64) + n_u for x in tu]
    own = np.where((deg[tu] > 0)[:, None], L_[0][tu], 0.0)
    return request_vectors(L_, deg, nbrs, own) @ final[n_u:n_u + n_i].T


def run_world(W, orders, topup, world, t0, occ_extra=None):
    items = W["items"]
    launched = launches(W)
    rng = np.random.default_rng(W["cfg"].seed + 1730)
    pairs, n_all = co_purchases(W, orders, 23 * DAY, launched, topup, rng)
    intents = generate(items, pairs, world)
    typical = criticise(items, pairs, intents, world)
    none = np.array([t == "no common reason" for t in intents])
    kept = [t for t, ok, n in zip(intents, typical, none) if ok and not n]
    print(f"\n{world}: {n_all:,} cross-category pairs bought together in at least {MIN_ORDERS} orders before day 23; "
          f"{len(pairs)} drawn. The model ({llm.model_id(MODEL)}) gave 'no common reason' for {none.sum()} "
          f"({none.mean():.1%}); the critic kept {len(kept)} of the other {len(pairs) - none.sum()}  "
          f"[{time.time() - t0:.0f} s]", flush=True)
    if occ_extra is not None:                                    # which pairs came from occasion baskets
        o_orders = set(zip(occ_extra["order_id"].tolist(), occ_extra["item_id"].tolist()))
        by_order = {}
        for o, i in o_orders:
            by_order.setdefault(o, set()).add(i)
        occ_pair = {(min(a, b), max(a, b)) for s in by_order.values() for a in s for b in s if a != b}
        is_occ = np.array([p in occ_pair for p in pairs])
        ok = np.array(typical) & ~none
        print(f"  pairs bought together in an occasion basket: {is_occ.sum()}; an intent kept for "
              f"{ok[is_occ].mean():.1%} of them and for {ok[~is_occ].mean():.1%} of the others")
    names, node = group_intents(kept)
    print(f"  {len(set(kept))} distinct kept phrases; {len(names)} intent nodes of at least {MIN_GROUP} phrases: "
          + "; ".join(f"{n} ({(node == k).sum()})" for k, n in enumerate(names)))
    for k, (a, b) in enumerate(pairs[:8]):
        print(f"    e.g. [{describe(items, a)}] + [{describe(items, b)}] -> {intents[k]!r}"
              f"{'' if typical[k] else ' (rejected by the critic)'}")
    tags = tag_concepts(names, world) if names else {}
    M = product_intents(items, tags, len(names))
    print(f"  concepts tagged: {sum(bool(v) for v in tags.values())} of {len(PRODUCT_CONCEPTS)}; products linked to "
          f"an intent: {M[items['is_product']].any(axis=1).sum():,} of {items['is_product'].sum():,}; launched: "
          f"{M[launched].any(axis=1).sum()} of {launched.sum()}")
    if names:
        P = planted(items)
        print("  each intent node's concepts, and the planted occasion they overlap most (Jaccard of concept sets):")
        occ_sets = [{TOKEN[w] for w in ws} for ws in OCCASIONS.values()]
        for k, n in enumerate(names):
            cs = {t for t, v in tags.items() if k in v}
            j = [len(cs & s) / max(len(cs | s), 1) for s in occ_sets]
            best = int(np.argmax(j))
            print(f"    {n:<28}{', '.join(sorted(WORDS[t][0] for t in cs)) or '(no concept)'}  ->  "
                  f"{list(OCCASIONS)[best]} {j[best]:.2f}")
    if os.environ.get("CH17_COSMO_LLM_ONLY") == "1":                # fill the cache only
        return None
    test = product_test(W, orders, 23 * DAY, 30 * DAY, launched)
    tu = test[3]
    arms = {"LightGCN + KG (Chapter 10)": None}
    if names:
        arms["+ LLM intent nodes"] = M
    arms["+ planted occasions (reference)"] = planted(items)
    res = {}
    for arm, extra in arms.items():
        runs = [product_ranking(launch_scores(W, test, extra, s), tu, launched, test[1], test[2]) for s in SEEDS]
        res[arm] = (np.mean([r[0] for r in runs], axis=0), np.mean([r[1] for r in runs], axis=0))
    print(f"  Launch test, {len(tu):,} users who bought a launched product in days 23 to 30 (NDCG@10, recall@10; "
          f"against: paired difference in NDCG@10)  [{time.time() - t0:.0f} s]")
    base = res["LightGCN + KG (Chapter 10)"][0]
    for arm, (nd, rc) in res.items():
        against = "" if arm.startswith("LightGCN") else "{:+.4f} ({:.4f}) [{:+.4f}, {:+.4f}]".format(
            *paired(nd, base, tu))
        print(f"    {arm:<34}{nd.mean():>9.4f}{rc.mean():>11.4f}   {against}")
    return res


if __name__ == "__main__":
    from tesselo_shop import build
    t0 = time.time()
    print(environment())
    W = load()
    _, items_b, _, _, orders, truth = build(W["cfg"])
    assert np.array_equal(items_b["vec"], W["items"]["vec"])
    merged, extra, _ = with_occasions(W, orders, truth["topup"])
    run_world(W, merged, truth["topup"], "occasions", t0, occ_extra=extra)
    run_world(W, orders, truth["topup"], "control", t0)
    print(f"\nModel: {llm.model_id(MODEL)}; {llm.engine_version(MODEL)}; greedy decoding, answers constrained by JSON "
          f"schemas. LightGCN + KG: 3 layers, {EPOCHS} epochs, means over seeds 0, 1 and 2.  [{time.time() - t0:.0f} s]")
