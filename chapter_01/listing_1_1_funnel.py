"""Listing 1.1 - The five-stage funnel in miniature (Tesselo home feed, composite teaching example).

Every model here is a toy. Embeddings are random and the scores are hand-written formulas,
so the numbers show the mechanics of the funnel, not real relevance or real probabilities.

Tested with Python 3.13 and 3.14, NumPy 2.5.4. Run from the repository root:
    pip install -r requirements.txt -c constraints.txt
    python chapter_01/listing_1_1_funnel.py
Repository: https://github.com/amitpandeygit/ai-recommender-systems (folder chapter_01)
"""
from collections import Counter

import numpy as np

rng = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# Setup: a toy Tesselo catalogue of 100,000 items and one user
# ---------------------------------------------------------------------------
NUM_ITEMS, DIM, NUM_CREATORS = 100_000, 32, 2_000
item_vecs = rng.normal(size=(NUM_ITEMS, DIM)).astype(np.float32)
item_vecs /= np.linalg.norm(item_vecs, axis=1, keepdims=True)   # unit length, so dot product = cosine
in_stock = rng.random(NUM_ITEMS) > 0.05                         # about 5% of items are unavailable
popularity = rng.random(NUM_ITEMS)                              # stand-in for a popularity feature, 0 to 1
creator = rng.integers(0, NUM_CREATORS, NUM_ITEMS)              # which creator or seller made each item

user_vec = rng.normal(size=DIM).astype(np.float32)
user_vec /= np.linalg.norm(user_vec)
history = rng.choice(NUM_ITEMS, size=500, replace=False)        # 500 items this user has already seen
seen = set(history.tolist())
views_per_creator = Counter(creator[history].tolist())          # how often this user watched each creator

# ---------------------------------------------------------------------------
# Stage 1. Retrieval (two-tower scoring): one dot product per item, keep the top 1,000
# Brute force here; Chapter 6 replaces it with an approximate nearest-neighbour index.
# ---------------------------------------------------------------------------
scores = item_vecs @ user_vec
candidates = np.argpartition(-scores, 1_000)[:1_000]            # top 1,000, not sorted
print("After retrieval:   ", len(candidates))

# ---------------------------------------------------------------------------
# Stage 2. Filtering: hard eligibility rules, no machine learning
# ---------------------------------------------------------------------------
candidates = np.array([i for i in candidates if in_stock[i] and i not in seen], dtype=np.intp)

# Underfill policy: if fewer than 200 candidates survive, top up with the most popular
# eligible items from the whole catalogue, skipping duplicates. If the whole eligible
# catalogue is too small, we accept a shorter slate rather than break a hard rule.
# Fine for this toy; a real system would use a ready-made popular list and a deadline,
# never a sort of the whole catalogue during a request.
MIN_POOL = 200
if len(candidates) < MIN_POOL:
    already = set(candidates.tolist())
    extra = [i for i in np.argsort(-popularity)
             if in_stock[i] and i not in seen and i not in already][: MIN_POOL - len(candidates)]
    candidates = np.concatenate([candidates, np.array(extra, dtype=np.intp)])
print("After filtering:   ", len(candidates))

# ---------------------------------------------------------------------------
# Stage 3. Pre-ranking: a cheap score, keep the top 200
# ---------------------------------------------------------------------------
pre_score = scores[candidates] + 0.1 * popularity[candidates]
candidates = candidates[np.argsort(-pre_score)[:200]]
print("After pre-ranking: ", len(candidates))

# ---------------------------------------------------------------------------
# Stage 4. Heavy ranking (joint scoring): computes a pair feature for each candidate from this
# user's history with the item's creator, and combines it with the match score.
# ---------------------------------------------------------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))                                  # squashes any number into the range 0 to 1

def joint_score(i):
    match = float(item_vecs[i] @ user_vec)                      # how well user and item vectors align
    creator_views = min(views_per_creator[creator[i]], 3)       # pair feature: past views of this creator, capped at 3
    boost = creator_views * match                               # pair feature combined with the match
    return sigmoid(3 * match + 2 * boost + 0.5 * popularity[i])

relevance_scores = np.array([joint_score(i) for i in candidates])  # bounded 0 to 1, NOT probabilities

# ---------------------------------------------------------------------------
# Stage 5. Reranking: MMR for variety, a preferred cap of 2 items per creator, a slate of 10.
# The creator cap is a soft rule: if it would leave the slate short, we relax it and backfill by score.
# ---------------------------------------------------------------------------
def mmr_rerank(cands, rel, k=10, lam=0.7, max_per_creator=2):
    chosen, per_creator = [], {}
    pool = list(range(len(cands)))
    while pool and len(chosen) < k:
        best, best_val = None, -np.inf
        for p in pool:
            item = cands[p]
            if per_creator.get(creator[item], 0) >= max_per_creator:
                continue
            sim = max((float(item_vecs[item] @ item_vecs[cands[c]]) for c in chosen), default=0.0)
            val = lam * rel[p] - (1 - lam) * sim
            if val > best_val:
                best, best_val = p, val
        if best is None:                                          # creator cap blocks every remaining item
            break
        chosen.append(best)
        c = creator[cands[best]]
        per_creator[c] = per_creator.get(c, 0) + 1
        pool.remove(best)
    if len(chosen) < k:                                           # backfill: relax the soft creator cap only
        rest = sorted(pool, key=lambda p: -rel[p])
        chosen += rest[: k - len(chosen)]
    return [int(cands[c]) for c in chosen]

slate = mmr_rerank(candidates, relevance_scores)
print("Final slate:       ", slate)
