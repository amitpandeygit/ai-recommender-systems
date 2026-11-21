"""What cleaning is worth to new items (section 17.3).

Section 10.5's and section 11.7's tests of new videos and launched products, run with
the records that Tesselo actually has for its new uploads. Each arm gives every new
upload a record (a category and title concepts) and a text for the text encoder; every
established item keeps its clean record. The arms:

  as written        - the exact dictionary's record (the declared category and the words
                      found exactly), and the listing as written for the encoder
  fuzzy dictionary  - the fuzzy dictionary's record, rendered in the item's language
  dictionary + LLM  - the fuzzy dictionary, with the words it could not read sent to the
                      language model (enrich.hybrid, enrich.WORKHORSE), rendered
  true record       - the item's real category and title, rendered the same way (a
                      reference that no real platform has)

New videos (the 1,724 new at day 23, the 31,331 test requests): the "New for you"
video at slot 4, chosen by Chapter 8's source (the user's language, then Chapter 3's
content similarity of title words, category and language), by LightGCN + KG (section
10.5, 240 epochs) and by the content map of section 11.7 (penalty 10), each with the
user's language first. New products (the 150 launched, the 978 test users): content
similarity, LightGCN + KG (40 epochs) and the map into the purchase graph (penalty 1).
Settings are Chapters 10 and 11's validation choices; models with seeds are means over
seeds 0, 1 and 2. Differences are paired against "as written", with standard errors
from 1,000 bootstrap draws of whole users.

Needs the cleaning answers in llm_cache (enrich.py full) and text_vectors.npz (made here
if missing, with Qwen3-Embedding). Run from the repository root (about 50 minutes):
    python chapter_17/new_items.py
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
from listings import LANGUAGES, _forms, as_items, exact_dictionary, fuzzy_dictionary, listings, new_uploads  # noqa: E402
from content_data import MODELS, load, modalities, text_embeddings  # noqa: E402
from content_models import fuse, profiles  # noqa: E402
from graph_data import DAY, click_graph, environment, paired, phase  # noqa: E402
from graph_models import fit_lightgcn, graph_degrees, history_nodes, layer_vectors, request_vectors  # noqa: E402
from cold_items import (LAYERS, PRODUCT_RELATIONS, VIDEO_RELATIONS, chapter8_new_scores, kg_edges, launches,  # noqa: E402
                        new_video_choice, product_ranking, product_test, slot_plays)
from cold_start import fit_map, mapped  # noqa: E402
from similarity_recs import item_features  # noqa: E402
from tesselo_graph import knowledge_graph  # noqa: E402
from title_words import WORDS  # noqa: E402
import enrich  # noqa: E402
import textvec  # noqa: E402

torch.set_num_threads(2)
SEEDS = (0, 1, 2)
EPOCHS = {"videos": 240, "products": 40}
PENALTY = {"videos": 10.0, "products": 1.0}
ARMS = ("as written", "fuzzy dictionary", "dictionary + LLM", "true record")
VOCAB = list(WORDS)


def render(tokens, language):
    """A record's concepts as text in the item's language, in the catalogue's order."""
    return " ".join(WORDS[t][language] for t in sorted(tokens, key=VOCAB.index))


def arm_records(W, L, ids):
    """Each arm's record and encoder text for the listed items."""
    forms_all = _forms(range(len(LANGUAGES)))
    llm_recs, _ = enrich.hybrid(enrich.WORKHORSE, L, ids)
    items = W["items"]
    out = {}
    for arm in ARMS:
        recs, texts = {}, {}
        for i in ids:
            if arm == "as written":
                recs[i], texts[i] = exact_dictionary(L[i]), L[i]["text"]
                continue
            if arm == "fuzzy dictionary":
                r = fuzzy_dictionary(L[i], forms_all)
            elif arm == "dictionary + LLM":
                r = llm_recs[i]
            else:
                r = {"category": int(items["category"][i]), "concepts": sorted({str(t) for t in items["title"][i]})}
            recs[i], texts[i] = r, render(r["concepts"], L[i]["language"])
        out[arm] = (recs, texts)
    return out


def arm_world(W, recs):
    """W with the arm's records for its items: titles, categories and the knowledge graph."""
    Wa = dict(W)
    Wa["items"] = as_items(W, recs)
    g = dict(W["graph"])
    h, r, t = knowledge_graph(W["cfg"], Wa["items"], g["brand"], g["seller"])
    g["kg"] = {"head": h, "relation": r, "tail": t}
    Wa["graph"] = g
    return Wa


def text_block(W, texts):
    """Every item's text vector: Chapter 11's clean title for established items, the
    arm's text for its items."""
    E, idx = text_embeddings(W, MODELS[-1])
    n = len(W["items"]["vec"])
    T = E[idx["titles"][np.arange(n), W["items"]["language"]]].copy()
    ids = sorted(texts)
    T[ids] = textvec.vectors([texts[i] for i in ids])
    return T


def lightgcn(Wa, cutoff, edges_user, n_u, n_i, item_nodes, relations, kg, epochs, seed):
    rows, cols, n_nodes = edges_user[0], edges_user[1], n_u + n_i
    if kg:
        e, n_ent = kg_edges(Wa, item_nodes - n_u, relations, n_u + n_i)
        rows, cols, n_nodes = np.r_[rows, e[0]], np.r_[cols, e[1]], n_nodes + n_ent
    deg = graph_degrees(rows, cols, n_nodes)
    m, A_hat = fit_lightgcn((rows, cols), n_nodes, n_u, item_nodes, layers=LAYERS, epochs=epochs, seed=seed)
    L_, final = layer_vectors(m, A_hat)
    return L_, final, deg


def video_test(W, arms, t0):
    S = W["snaps"][23]
    cols = np.flatnonzero(W["video"] & ~S["ok"])
    R, H, _, bad = phase(W, 23, S)
    bn = bad[:, cols]
    del bad
    users, items = W["users"], W["items"]
    n_u, n_i = len(users["vec"]), len(items["vec"])
    same = 1_000.0 * (users["language"][R["user"]][:, None] == items["language"][cols][None, :])
    u, i = click_graph(W, 23 * DAY).nonzero()
    vn = np.flatnonzero(W["video"]) + n_u
    mods = list(modalities(W).values())
    established = np.flatnonzero(W["video"] & S["ok"])
    plain = []                                                   # LightGCN on clicks: the same in every arm
    for seed in SEEDS:
        L_, final, deg = lightgcn(W, 23 * DAY, (u, i + n_u), n_u, n_i, vn, VIDEO_RELATIONS, False,
                                  EPOCHS["videos"], seed)
        own = np.where((deg[R["user"]] > 0)[:, None], L_[0][R["user"]], 0.0)
        plain.append((request_vectors(L_, deg, history_nodes(H, n_u), own), final[n_u:n_u + n_i]))
    print(f"  LightGCN on clicks fitted  [{time.time() - t0:.0f} s]", flush=True)
    res = {}
    for arm, (recs, texts) in arms.items():
        Wa = arm_world(W, {i: r for i, r in recs.items() if i in set(cols.tolist())})
        res[(arm, "Chapter 8's source")] = [slot_plays(W, R, new_video_choice(chapter8_new_scores(Wa, S, R, cols),
                                                                              bn, cols))]
        runs = []
        for seed in SEEDS:
            L_, final, deg = lightgcn(Wa, 23 * DAY, (u, i + n_u), n_u, n_i, vn, VIDEO_RELATIONS, True,
                                      EPOCHS["videos"], seed)
            own = np.where((deg[R["user"]] > 0)[:, None], L_[0][R["user"]], 0.0)
            s = request_vectors(L_, deg, history_nodes(H, n_u), own) @ final[n_u + cols].T + same
            runs.append(slot_plays(W, R, new_video_choice(s, bn, cols)))
        res[(arm, "LightGCN + KG")] = runs
        T = text_block(W, {i: t for i, t in texts.items() if i in set(cols.tolist())})
        parts = [(T, np.ones(n_i, dtype=bool))] + mods
        X = fuse(parts, [1.0] * len(parts))
        runs = []
        for U, V in plain:
            M = fit_map(X[established], V[established], PENALTY["videos"])
            runs.append(slot_plays(W, R, new_video_choice(U @ mapped(X[cols], M).T + same, bn, cols)))
        res[(arm, "content map")] = runs
        print(f"  {arm}: done  [{time.time() - t0:.0f} s]", flush=True)
    return res, R


def product_test_arms(W, arms, orders, t0):
    users, items = W["users"], W["items"]
    n_u, n_i = len(users["vec"]), len(items["vec"])
    new = launches(W)
    pairs, bought, pos, tu = product_test(W, orders, 23 * DAY, 30 * DAY, new)
    hist = np.zeros((len(tu), n_i))
    for k, x in enumerate(tu):
        hist[k, list(bought.get(int(x), []))] = 1.0
    hist /= np.maximum(hist.sum(axis=1, keepdims=True), 1.0)
    prods = np.flatnonzero(items["is_product"])
    established = prods[~new[prods]]
    nbrs = [np.array(sorted(bought.get(int(x), [])), dtype=np.int64) + n_u for x in tu]
    buy = (pairs[:, 0], pairs[:, 1] + n_u)
    mods = list(modalities(W).values())
    plain = []
    for seed in SEEDS:
        L_, final, deg = lightgcn(W, 23 * DAY, buy, n_u, n_i, prods + n_u, PRODUCT_RELATIONS, False, 200, seed)
        own = np.where((deg[tu] > 0)[:, None], L_[0][tu], 0.0)
        plain.append((request_vectors(L_, deg, nbrs, own), final[n_u:n_u + n_i]))
    res = {}
    launched = set(np.flatnonzero(new).tolist())
    for arm, (recs, texts) in arms.items():
        Wa = arm_world(W, {i: r for i, r in recs.items() if i in launched})
        feats = item_features(Wa["items"], n_i)
        res[(arm, "content similarity")] = [product_ranking((hist @ feats) @ feats.T, tu, new, bought, pos)]
        runs = []
        for seed in SEEDS:
            L_, final, deg = lightgcn(Wa, 23 * DAY, buy, n_u, n_i, prods + n_u, PRODUCT_RELATIONS, True,
                                      EPOCHS["products"], seed)
            own = np.where((deg[tu] > 0)[:, None], L_[0][tu], 0.0)
            runs.append(product_ranking(request_vectors(L_, deg, nbrs, own) @ final[n_u:n_u + n_i].T, tu, new,
                                        bought, pos))
        res[(arm, "LightGCN + KG")] = runs
        T = text_block(W, {i: t for i, t in texts.items() if i in launched})
        parts = [(T, np.ones(n_i, dtype=bool))] + mods
        X = fuse(parts, [1.0] * len(parts))
        runs = []
        for U, V in plain:
            M = fit_map(X[established], V[established], PENALTY["products"])
            runs.append(product_ranking(U @ mapped(X, M).T, tu, new, bought, pos))
        res[(arm, "content map")] = runs
    return res, tu


if __name__ == "__main__":
    t0 = time.time()
    print(environment())
    W = load()
    L = listings(W)
    vids = new_uploads(W, 23)
    arms = arm_records(W, L, sorted(L))
    textvec.vectors(sorted({t for _, texts in arms.values() for t in texts.values()}))   # embed every new text once
    print(f"Chapter 2's world; {len(L):,} messy listings; the cleaning model is {enrich.WORKHORSE}  "
          f"[{time.time() - t0:.0f} s]", flush=True)
    res, R = video_test(W, arms, t0)
    print(f"\nNew videos, test week: the slot-4 video among the {(W['video'] & ~W['snaps'][23]['ok']).sum():,} new "
          f"videos, for {len(R['user']):,} requests. Slot-4 plays; against: paired difference against 'as written' "
          f"for the same method, standard error in brackets")
    print(f"  {'':<20}" + "".join(f"{m:>32}" for m in ("Chapter 8's source", "LightGCN + KG", "content map")))
    for arm in ARMS:
        cells = []
        for m in ("Chapter 8's source", "LightGCN + KG", "content map"):
            p = np.mean(res[(arm, m)], axis=0)
            if arm == "as written":
                cells.append(f"{p.mean():.4f}")
            else:
                d, se = paired(p, np.mean(res[("as written", m)], axis=0), R["user"])[:2]
                cells.append(f"{p.mean():.4f} {d:+.4f} ({se:.4f})")
        print(f"  {arm:<20}" + "".join(f"{c:>32}" for c in cells))
    del R
    from tesselo_shop import build
    _, items_b, _, _, orders, _ = build(W["cfg"])
    assert np.array_equal(items_b["vec"], W["items"]["vec"])
    res, tu = product_test_arms(W, arms, orders, t0)
    print(f"\nLaunched products, test week: the 150 ranked for {len(tu):,} users. NDCG@10, recall@10; against: paired "
          f"difference in NDCG@10 against 'as written' for the same method  [{time.time() - t0:.0f} s]")
    print(f"  {'':<20}" + "".join(f"{m:>36}" for m in ("content similarity", "LightGCN + KG", "content map")))
    for arm in ARMS:
        cells = []
        for m in ("content similarity", "LightGCN + KG", "content map"):
            nd = np.mean([r[0] for r in res[(arm, m)]], axis=0)
            rc = np.mean([r[1] for r in res[(arm, m)]], axis=0)
            if arm == "as written":
                cells.append(f"{nd.mean():.4f} {rc.mean():.4f}")
            else:
                base = np.mean([r[0] for r in res[("as written", m)]], axis=0)
                d, se = paired(nd, base, tu)[:2]
                cells.append(f"{nd.mean():.4f} {rc.mean():.4f} {d:+.4f} ({se:.4f})")
        print(f"  {arm:<20}" + "".join(f"{c:>36}" for c in cells))
    print(f"  (models with seeds are means over seeds 0, 1 and 2; a random order has recall@10 {10 / 150:.4f})  "
          f"[{time.time() - t0:.0f} s]")
