"""Listings 6.1 and 6.3 - Two-tower retrieval for Tesselo (sections 6.3 to 6.6).

  6.3  the two towers: a user tower that reads the user's history and app
       language, and an item tower that reads the item's id, category,
       language, title words and creator; both end in a unit-length vector,
       and the score is their dot product divided by a temperature
  6.4  training with a softmax over sampled candidates: uniform, popular,
       in-batch and mixed negatives (from warm items, or from the whole catalogue),
       the logQ correction for how often each item is sampled, and hard
       negatives mined from the model itself
  6.5  exposure bias: negatives taken from what the old system showed
  6.6  the honest test of Chapter 5: the time split, the fixed baselines,
       recall@100 and recall@200 for the retrieval job, the guardrails, the
       simulated online test, new users and new videos

Training uses only what was known at the cutoff (day 16 for the validation
week, day 23 for the test week), through Chapter 3's user_item_scores(), so the
consent rules, the eligible items and the exclusions are the same as for iALS.
Settings are chosen on the validation week, never on the test week.

This is a laptop-scale PyTorch implementation for teaching (CPU only).

Needs Python 3.12 or later. Run from the repository root (about 30 minutes on a two-core machine):
    pip install -r requirements.txt -c constraints.txt
    python chapter_06/two_tower.py
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
for folder in ("chapter_03", "chapter_04", "chapter_05"):
    sys.path.insert(0, str(ROOT / folder))
from tesselo_shop import DAY, LANGUAGES, Config, build  # noqa: E402
from similarity_recs import (MIN_BUYERS, NO, cf_scores, content_scores, group_popularity,  # noqa: E402
                             hit_rate, item_features, item_neighbours, mcnemar_exact,
                             test_positives, top_k, user_item_scores)
from implicit_scores import EVENT_WEIGHTS, active_blocks, prepare  # noqa: E402
from implicit_mf import bpr, eligible_and_scored, fold_in, ials, taste_correlations  # noqa: E402
from evaluate import expected_outcomes, ranked, ranking_table  # noqa: E402
from metrics import exposure, ndcg_at_k, recall_at_k  # noqa: E402

DIM = 32                                     # length of the user and item vectors
# The settings chosen on the validation week by the grid at the start of the main block
# (which checks that it picks these). Other files import them.
MAIN = dict(negatives="mixed", logq=True, bias=True, temperature=0.2, batch=256, epochs=25)
HITS = ("complete", "like")                  # a satisfied play, as the logs record it
torch.set_num_threads(2)                     # the same thread count on every machine


# ---------------------------------------------------------------- section 6.3: the towers
class TwoTower(torch.nn.Module):
    """score(u, i) = user_tower(u) . item_tower(i) / temperature, plus an optional item bias."""

    def __init__(self, n_items, n_creators, n_content, n_languages, dim=DIM, bias=False):
        super().__init__()
        self.bias = bias                                           # an item bias, learned from the item's inputs
        self.history = torch.nn.Embedding(n_items, dim)            # an item as part of a history
        self.language = torch.nn.Embedding(n_languages, 8)
        self.user_mlp = torch.nn.Sequential(torch.nn.Linear(dim + 8 + 1, 64), torch.nn.ReLU(),
                                            torch.nn.Linear(64, dim))
        self.item_id = torch.nn.Embedding(n_items, dim)             # what behaviour says about the item
        torch.nn.init.zeros_(self.item_id.weight)                  # an unseen item starts at zero
        self.content = torch.nn.Linear(n_content, dim, bias=False)  # title words, category, language
        self.creator = torch.nn.Embedding(n_creators, 16)
        self.item_mlp = torch.nn.Sequential(torch.nn.Linear(dim + dim + 16, 64), torch.nn.ReLU(),
                                            torch.nn.Linear(64, dim + int(bias)))

    def user_tower(self, hist, lang):
        """hist: users x items, each row the user's history weights summing to 1 (or all 0)."""
        bag = hist @ self.history.weight                           # weighted mean of history vectors
        size = torch.log1p((hist > 0).sum(dim=1, keepdim=True).float())
        x = torch.cat([bag, self.language(lang), size], dim=1)
        return F.normalize(self.user_mlp(x), dim=1)

    def item_tower(self, items, feats, creator, id_dropout=0.0):
        """id_dropout (training only): the share of items whose id vector is hidden, so
        that the content and creator inputs learn to place an item on their own."""
        idv = self.item_id(items)
        if id_dropout > 0:
            idv = idv * (torch.rand(len(items), 1) >= id_dropout)
        return self.item_from_inputs(idv, feats[items], creator[items])

    def item_from_inputs(self, idv, feat, creator):
        """The item tower on raw inputs. A new item, which has no id vector yet, passes zeros."""
        x = torch.cat([idv, self.content(feat), self.creator(creator)], dim=1)
        out = self.item_mlp(x)
        if not self.bias:
            return F.normalize(out, dim=1)
        return torch.cat([F.normalize(out[:, :-1], dim=1), out[:, -1:]], dim=1)   # unit vector, then the bias


def history_rows(weight, rows, drop=None):
    """History weights for the given users, normalised to sum to 1. drop[r] removes one
    item from row r: the training target, which the user tower must not see."""
    h = weight[rows].clone()
    if drop is not None:
        h[torch.arange(len(rows)), drop] = 0.0
    total = h.sum(dim=1, keepdim=True)
    return torch.where(total > 0, h / total.clamp_min(1e-12), h)


# ---------------------------------------------------------------- section 6.4: the loss
def scores(u, v, temperature, paired=False):
    """u . v / temperature, plus the item bias when the item vector carries one (its last
    number). paired=True scores row r of u with row r of v only; otherwise every pair."""
    d = u.shape[-1]
    s = ((u * v[:, :d]).sum(dim=1) if paired else u @ v[:, :d].T) / temperature
    if v.shape[-1] > d:
        s = s + (v[:, d] if paired else v[:, d][None, :])
    return s


def sampled_softmax_loss(u, v_pos, v_cand, logq_pos, logq_cand, temperature, mask=None):
    """Softmax over the positive (column 0) and the candidates, each logit corrected
    by subtracting log(expected number of times that item is among the candidates).
    mask[r, c] = True removes candidate c from row r (an accidental hit)."""
    pos = scores(u, v_pos, temperature, paired=True) - logq_pos
    cand = scores(u, v_cand, temperature) - logq_cand[None, :]
    if mask is not None:
        cand = cand.masked_fill(mask, float("-inf"))
    logits = torch.cat([pos[:, None], cand], dim=1)
    return F.cross_entropy(logits, torch.zeros(len(u), dtype=torch.long))


def train_two_tower(data, pos_u, pos_i, negatives="mixed", logq=True, temperature=0.1,
                    epochs=15, batch=512, n_uniform=512, n_hard=0, lr=0.003, bias=False, id_dropout=0.0,
                    seed=0):
    """Train on (user, item) positive pairs.

    negatives: "uniform"  n_uniform items drawn uniformly from the items with history
               "popular"  n_uniform items drawn in proportion to their positives
               "in-batch" the other positives in the same batch
               "mixed"    in-batch plus n_uniform uniform items from the items with history: a
                          version of Yang and colleagues' (2020) mixed sampling limited to warm items
               "mixed-all" in-batch plus n_uniform uniform items from the whole catalogue, as
                          Yang and colleagues drew them (items with no positives included)
               "impressions" items the old system showed the user that they did not engage with
    logq:      subtract log(expected count) of each item among the candidates
    n_hard:    per user, add this many hard negatives mined from the model's own ranking
               (ranks 11 to 100 of items the user has not engaged with), refreshed every epoch
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = TwoTower(data["n_items"], data["n_creators"], data["feats"].shape[1], len(LANGUAGES), bias=bias)
    model.temperature = temperature
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    W, feats, creator, lang = data["W"], data["feats"], data["creator"], data["lang"]
    n_items = data["n_items"]
    freq = np.bincount(pos_i, minlength=n_items).astype(float)
    known = np.flatnonzero(freq > 0)                               # items the model has history for
    p_batch = freq / freq.sum()                                    # chance a batch slot holds item j
    p_pop = freq[known] / freq[known].sum()
    pos_set = data["pos_sets"]                                     # users x items, True = positive
    hard = {}
    for epoch in range(epochs):
        if n_hard and epoch >= 2:                                  # mine after two warm-up epochs
            hard = mine_hard(model, data, np.unique(pos_u), pos_set)
        order = rng.permutation(len(pos_u))
        for s in range(0, len(order), batch):
            o = order[s:s + batch]
            u_idx, i_idx = pos_u[o], pos_i[o]
            B = len(o)
            u = model.user_tower(history_rows(W, torch.as_tensor(u_idx), torch.as_tensor(i_idx)),
                                 lang[u_idx])
            v_pos = model.item_tower(torch.as_tensor(i_idx), feats, creator, id_dropout)
            if negatives == "impressions":
                loss = impression_loss(model, data, u, v_pos, u_idx, temperature, rng)
            else:
                cand, expected, mask = candidates(negatives, i_idx, u_idx, known, p_pop, p_batch,
                                                  n_uniform, B, n_items, pos_set, rng)
                if hard:
                    cand, expected, mask = add_hard(cand, expected, mask, u_idx, hard, n_hard, rng)
                v_cand = model.item_tower(torch.as_tensor(cand), feats, creator, id_dropout)
                lq_c = torch.as_tensor(np.log(expected) if logq else np.zeros(len(cand)), dtype=torch.float32)
                lq_p = torch.as_tensor(np.log(expected_of(i_idx, negatives, p_batch, known, p_pop,
                                                          n_uniform, B)) if logq else np.zeros(B),
                                       dtype=torch.float32)
                loss = sampled_softmax_loss(u, v_pos, v_cand, lq_p, lq_c, temperature,
                                            torch.as_tensor(mask))
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return model


def expected_of(items, negatives, p_batch, known, p_pop, n_uniform, B):
    """Expected number of times each item appears among the shared candidates."""
    e = np.zeros(len(items))
    if negatives in ("in-batch", "mixed"):
        e += B * p_batch[items]
    if negatives in ("uniform", "mixed"):
        e += n_uniform / len(known)
    if negatives == "mixed-all":
        e += B * p_batch[items] + n_uniform / len(p_batch)
    if negatives == "popular":
        q = np.zeros(len(p_batch))
        q[known] = p_pop
        e += n_uniform * q[items]
    return np.maximum(e, 1e-12)


def candidates(negatives, i_idx, u_idx, known, p_pop, p_batch, n_uniform, B, n_items, pos_set, rng):
    """Shared candidate items for one batch, their expected counts, and the
    accidental-hit mask (a candidate that is a positive of that row's user)."""
    cand = np.empty(0, dtype=np.int64)
    if negatives in ("in-batch", "mixed"):
        cand = np.concatenate([cand, i_idx])                       # the batch's own positives
    if negatives == "mixed-all":
        cand = np.concatenate([cand, i_idx, rng.choice(n_items, n_uniform)])   # any item can be drawn
    if negatives in ("uniform", "mixed"):
        cand = np.concatenate([cand, rng.choice(known, n_uniform)])
    if negatives == "popular":
        cand = np.concatenate([cand, rng.choice(known, n_uniform, p=p_pop)])
    expected = expected_of(cand, negatives, p_batch, known, p_pop, n_uniform, B)
    mask = pos_set[u_idx][:, cand]                                 # pos_set: users x items, True = positive
    return cand, expected, mask


def add_hard(cand, expected, mask, u_idx, hard, n_hard, rng):
    """Append n_hard mined negatives per row. Each is a candidate for its own row only,
    with no logQ correction (it was not drawn from a known distribution)."""
    rows = [rng.choice(hard[u], n_hard) for u in u_idx.tolist()]
    extra = np.concatenate(rows)
    own = np.zeros((len(u_idx), len(extra)), dtype=bool)
    for r in range(len(u_idx)):
        own[r, r * n_hard:(r + 1) * n_hard] = True
    mask = np.concatenate([mask, ~own], axis=1)                    # other rows' hard items are hidden
    return np.concatenate([cand, extra]), np.concatenate([expected, np.ones(len(extra))]), mask


def mine_hard(model, data, users, pos_set, lo=10, hi=100):
    """Each user's unengaged items at ranks lo+1 to hi under the current model."""
    with torch.no_grad():
        U, V = embed_all(model, data)
    out = {}
    for c in range(0, len(users), 1000):
        rows = users[c:c + 1000]
        s = U[rows] @ V.T
        s[:, ~data["known"]] = -np.inf                             # only items the model knows
        s[pos_set[rows]] = -np.inf
        best = np.argsort(-s, axis=1)[:, lo:hi]
        out.update({int(u): b for u, b in zip(rows, best)})
    return out


def impression_loss(model, data, u, v_pos, u_idx, temperature, rng, k=20):
    """Section 6.5: softmax over the positive and up to k impressions the user saw and
    did not engage with. No candidate the old system never showed is ever a negative."""
    imp = data["impressions"]
    picks = [rng.choice(imp[x], k) if len(imp.get(x, ())) else np.zeros(k, dtype=np.int64)
             for x in u_idx.tolist()]
    has = torch.as_tensor([len(imp.get(x, ())) > 0 for x in u_idx.tolist()])
    neg = torch.as_tensor(np.stack(picks))
    v = model.item_tower(neg.reshape(-1), data["feats"], data["creator"])
    pos = scores(u, v_pos, temperature, paired=True)[:, None]
    cand = scores(u.repeat_interleave(k, dim=0), v, temperature, paired=True).reshape(len(u), k)
    cand = cand.masked_fill(~has[:, None], float("-inf"))
    return F.cross_entropy(torch.cat([pos, cand], dim=1), torch.zeros(len(u_idx), dtype=torch.long))


# ---------------------------------------------------------------- data and serving
def make_data(weight, items, users, feats, n_creators):
    n_users, n_items = weight.shape
    W = torch.as_tensor(weight, dtype=torch.float32)
    return {"W": W, "n_items": n_items, "n_creators": n_creators,
            "feats": torch.as_tensor(feats, dtype=torch.float32),
            "creator": torch.as_tensor(items["creator"]), "lang": torch.as_tensor(users["language"]),
            "pos_sets": weight > 0, "known": (weight > 0).any(axis=0)}


def embed_all(model, data, hist=None):
    """Every user's vector from their full history, and every item's vector. With an
    item bias, the user vector gets one more number, the temperature, so that a plain
    dot product U @ V.T ranks items exactly as the model does (it equals the model's
    score times the temperature)."""
    with torch.no_grad():
        W = data["W"] if hist is None else hist
        U = model.user_tower(history_rows(W, torch.arange(len(W))), data["lang"][:len(W)])
        V = model.item_tower(torch.arange(data["n_items"]), data["feats"], data["creator"])
        if model.bias:
            U = torch.cat([U, torch.full((len(U), 1), model.temperature)], dim=1)
    return U.numpy().astype(np.float64), V.numpy().astype(np.float64)


def satisfied_pairs(events, users, cutoff, n_users, n_items, video):
    """Section 6.6, model B: (user, video) pairs with a completion or like before the
    cutoff, unblocked, under the same consent and identity rules as user_item_scores()."""
    d = prepare(events, users, cutoff)
    m = np.isin(d["event"], HITS) & video[d["item"]]
    blocks = active_blocks(events, users, cutoff)
    pairs = {(u, i) for u, i in zip(d["user"][m].tolist(), d["item"][m].tolist()) if (u, i) not in blocks}
    pairs = np.array(sorted(pairs), dtype=np.int64).reshape(-1, 2)
    return pairs[:, 0], pairs[:, 1]


def impression_negatives(events, users, cutoff, weight):
    """Section 6.5: per user, items they were shown on screen and did not engage with."""
    d = prepare(events, users, cutoff)
    m = d["event"] == "viewable"
    out = {}
    for u, i in set(zip(d["user"][m].tolist(), d["item"][m].tolist())):
        if weight[u, i] <= 0:
            out.setdefault(u, []).append(i)
    return {u: np.array(sorted(v)) for u, v in out.items()}


def recall_table(top, rows, pos, ks=(10, 100, 200)):
    """Mean recall@k over users (users with nothing relevant are never passed in)."""
    return {k: float(np.mean([recall_at_k(t.tolist(), pos[u], k=k) for u, t in zip(rows, top)]))
            for k in ks}


def guardrails(tops, ok, creators, top_creators, n_items):
    """Catalogue coverage of eligible videos, and exposure share of the top-10% creators."""
    exp = np.zeros(n_items)
    for lst in tops:
        for i, e in exposure(lst.tolist()).items():
            exp[i] += e
    share = sum(exp[i] for i in np.flatnonzero(exp) if creators[i] in top_creators) / exp.sum()
    return float(np.mean(exp[ok] > 0)), float(share)


# ---------------------------------------------------------------- main
def chain_lists(score, fallback, rows, used, video, k):
    """Section 6.6: the two-tower list, filled by the named fallback (Chapter 3's chain)."""
    return ranked(lambda r: (score[r], fallback(r)), rows, used, video, k=k)


if __name__ == "__main__":
    import time
    t0 = time.time()
    cfg = Config()
    users, items, served, events, orders, truth = build(cfg)
    n_users, n_items = cfg.num_users, cfg.num_items
    video = ~items["is_product"]
    feats = item_features(items, n_items)
    n_creators = int(items["creator"].max()) + 1
    n_video = int(video.sum())
    print(f"Item content inputs: {feats.shape[1]} TF-IDF columns (title words, category, language); "
          f"{n_creators} creators")

    # ---- validation week: train to day 16, test days 16 to 23; choose the settings
    w16, used16 = user_item_scores(events, users, 16 * DAY, n_users, n_items)
    pos16 = test_positives(events, users, 16 * DAY, 23 * DAY, used16, video)
    r16 = np.array(sorted(pos16))
    ok16 = video & ((w16 > 0).sum(axis=0) >= MIN_BUYERS)
    d16 = make_data(w16, items, users, feats, n_creators)
    pu16, pi16 = np.nonzero(w16 > 0)
    print(f"\nValidation week: {len(r16):,} users with a hit; {len(pu16):,} positive pairs. "
          f"NDCG@10 of the two-tower list (mixed negatives with logQ, batch 256, 25 epochs):")
    grid = {}
    for bias in (False, True):
        for tau in (0.05, 0.1, 0.2):
            m = train_two_tower(d16, pu16, pi16, negatives="mixed", logq=True, bias=bias, temperature=tau,
                                batch=256, epochs=25)
            U, V = embed_all(m, d16)
            t = top_k(eligible_and_scored(U[r16] @ V.T, w16[r16], ok16), used16[r16], video, 10)
            per_user = [ndcg_at_k(x.tolist(), {i: 1.0 for i in pos16[u]}) for u, x in zip(r16, t)]
            grid[(bias, tau)] = np.mean(per_user)
            se = np.std(per_user, ddof=1) / np.sqrt(len(per_user))
            print(f"  item bias {'yes' if bias else 'no ':<4} temperature {tau:<5} NDCG@10 {grid[(bias, tau)]:.4f}"
                  f" (standard error {se:.4f})")
    best = max(grid, key=lambda b: round(grid[b], 9))
    print(f"Chosen: item bias {'yes' if best[0] else 'no'}, temperature {best[1]}; "
          f"equals MAIN: {best == (MAIN['bias'], MAIN['temperature'])}   [{time.time() - t0:.0f} s]")

    # ---- the test week of Chapters 3 to 5: train to day 23, test days 23 to 30
    weight, used = user_item_scores(events, users, 23 * DAY, n_users, n_items)
    pos = test_positives(events, users, 23 * DAY, 30 * DAY, used, video)
    rows = np.array(sorted(pos))
    engaged = (weight > 0).sum(axis=0)
    ok = video & (engaged >= MIN_BUYERS)
    hist = (weight > 0).any(axis=1)
    known = np.flatnonzero(users["current_consent"] & hist)
    aud = np.flatnonzero(users["current_consent"])                      # the simulated online audience
    print(f"\nTest week: {len(rows):,} test users with a relevant video; of the {len(known):,} consenting "
          f"users with history, {len(known) - np.isin(known, rows).sum():,} had none and are left out of "
          f"offline averages; online audience {len(aud):,} users")

    # Chapter 5's fixed baselines, rebuilt with the settings chosen in Chapters 3 and 4
    pop = group_popularity(weight, np.zeros(n_users, dtype=np.int64), 1)[0]
    by_lang = group_popularity(weight, users["language"], len(LANGUAGES))
    lang_pop = lambda r: by_lang[users["language"][r]]                        # noqa: E731
    sim = item_neighbours(weight, k=50, strength=0.0)
    X, Y = ials(weight, 16, 1.0, 10.0)
    Pb, Qb, bb = bpr(weight, 32, reg=0.01, neg_power=1.0)
    d23 = prepare(events, users, 23 * DAY)
    views = np.bincount(d23["item"][d23["event"] == "viewable"], minlength=n_items).astype(float)
    clicks = np.bincount(d23["item"][d23["event"] == "click"], minlength=n_items).astype(float)
    ctr = (clicks + 20 * clicks.sum() / views.sum()) / (views + 20)
    row = lambda v: (lambda r: np.broadcast_to(v, (len(r), n_items)))        # noqa: E731
    shuffle = lambda r: np.array([np.random.default_rng([cfg.seed, 504, int(u)]).random(n_items)  # noqa: E731
                                  for u in r])
    match = lambda r: users["vec"][r].astype(np.float64) @ items["vec"].T.astype(np.float64)  # noqa: E731
    methods = {"random (eligible videos)": lambda r: np.where(ok, shuffle(r), NO),
               "most clicked (click-through rate)": row(np.where(ok, ctr, NO)),
               "popular worldwide": row(pop),
               "popular in your language": lang_pop,
               "CF, then popular in language": lambda r: (cf_scores(weight[r], sim), lang_pop(r)),
               "iALS, then popular in language":
                   lambda r: (eligible_and_scored(X[r] @ Y.T, weight[r], ok), lang_pop(r)),
               "BPR popular negatives, then language":
                   lambda r: (eligible_and_scored(Pb[r] @ Qb.T + bb, weight[r], ok), lang_pop(r))}

    vc = np.unique(items["creator"][video])                              # Chapter 5's creator groups
    cpop = np.bincount(items["creator"][video], weights=engaged[video].astype(float), minlength=n_creators)
    top_creators = set(vc[np.argsort(-cpop[vc], kind="stable")][: len(vc) // 10].tolist())

    def evaluate(score_fn, keep=False):
        """Every measure of section 6.6 for one method (score_fn as in Chapter 5's ranked()).
        keep=True also returns the full ordering for each test user."""
        full = ranked(score_fn, rows, used, video, k=n_video)
        t = ranking_table(full, rows, pos, used, video)
        t.update({f"R@{k}": v for k, v in recall_table(full, rows, pos, (100, 200)).items()})
        t["cov"], t["creators"] = guardrails(full[:, :10], ok, items["creator"], top_creators, n_items)
        lists_all = ranked(score_fn, aud, used, video, k=10)
        t["online"] = expected_outcomes(lists_all, aud, users, items, cfg)[1].sum(axis=1).mean()
        t["online_test"] = expected_outcomes(full[:, :10], rows, users, items, cfg)[1].sum(axis=1).mean()
        t["cov_all"], t["creators_all"] = guardrails(lists_all, ok, items["creator"], top_creators, n_items)
        if keep:
            t["full"] = full
        return t

    results = {name: evaluate(fn, keep=name in ("popular in your language", "iALS, then popular in language"))
               for name, fn in methods.items()}
    print(f"Baselines rebuilt  [{time.time() - t0:.0f} s]")

    # ---- section 6.4 and 6.5: the same two-tower model, only the negatives change
    data = make_data(weight, items, users, feats, n_creators)
    data["impressions"] = impression_negatives(events, users, 23 * DAY, weight)
    pu, pi = np.nonzero(weight > 0)
    base = {k: v for k, v in MAIN.items() if k not in ("negatives", "logq")}
    variants = {"uniform negatives": dict(negatives="uniform", logq=True),
                "popular negatives, no correction": dict(negatives="popular", logq=False),
                "popular negatives, logQ": dict(negatives="popular", logq=True),
                "in-batch, no correction": dict(negatives="in-batch", logq=False),
                "in-batch, logQ": dict(negatives="in-batch", logq=True),
                "mixed, logQ (MAIN)": dict(negatives="mixed", logq=True),
                "mixed over all items, logQ": dict(negatives="mixed-all", logq=True),
                "mixed, logQ, 4 hard negatives": dict(negatives="mixed", logq=True, n_hard=4),
                "impression negatives (6.5)": dict(negatives="impressions", logq=False)}
    models, taste = {}, {}
    hist_users = np.flatnonzero(hist)
    for name, kw in variants.items():
        m = models[name] = train_two_tower(data, pu, pi, **base, **kw)
        U, V = embed_all(m, data)
        s = eligible_and_scored(U @ V.T, weight, ok)
        results[f"two-tower: {name}"] = evaluate(lambda r, s=s: (s[r], lang_pop(r)), keep=name == "mixed, logQ (MAIN)")
        del s
        taste[name] = taste_correlations({"m": lambda u, i, U=U, V=V: np.einsum("nk,nk->n", U[u], V[i])},
                                         users, items, hist_users, ok)[0]["m"][1]
    print(f"\nSections 6.4 and 6.5: one two-tower model, nine ways to choose negatives "
          f"(each list filled by popular in your language)  [{time.time() - t0:.0f} s]")
    print(f"  {'':<42}{'NDCG@10':>8}{'R@200':>8}{'coverage':>10}{'top-10% cr.':>12}{'online':>8}{'taste':>7}")
    for name in variants:
        t = results[f"two-tower: {name}"]
        print(f"  {name:<42}{t['NDCG@10']:>8.3f}{t['R@200']:>8.3f}{t['cov']:>10.1%}{t['creators']:>12.1%}"
              f"{t['online']:>8.4f}{taste[name]:>7.2f}")
    print("  (offline on the test users; online = expected satisfied plays per list, all consenting users; "
          "taste = pooled correlation with the hidden match)")
    print(f"  items with no positive pair before day 23 (only 'mixed over all items' can draw them): "
          f"{int((~data['known']).sum()):,} of {n_items:,}")

    # what the negatives are, in the hidden truth (simulation only)
    def p_enjoy(u, i):
        m_ = np.einsum("nd,nd->n", users["vec"][u].astype(np.float64), items["vec"][i].astype(np.float64))
        return 1 / (1 + np.exp(-(5 * m_ - 2.0 - 2 * items["clickbait"][i])))
    rng = np.random.default_rng(cfg.seed + 606)
    su = rng.choice(pu.size, 20_000, replace=False)
    hard = mine_hard(models["mixed, logQ (MAIN)"], data, np.unique(pu), data["pos_sets"])
    hu = np.array([u for u in pu[su]])
    hi_ = np.array([hard[u][rng.integers(len(hard[u]))] for u in hu])
    iu = np.array([u for u in pu[su] if u in data["impressions"]])
    ii = np.array([data["impressions"][u][rng.integers(len(data["impressions"][u]))] for u in iu])
    kn = np.flatnonzero(data["known"])
    print("\nWhat each kind of negative is, in the hidden truth: mean chance the user would be satisfied by "
          "the video if they clicked it")
    for label, (u_, i_) in (("positives (training pairs)", (pu[su], pi[su])),
                            ("uniform negatives", (pu[su], rng.choice(kn, len(su)))),
                            ("impression negatives", (iu, ii)),
                            ("mined hard negatives (ranks 11-100)", (hu, hi_))):
        p = p_enjoy(u_, i_)
        print(f"  {label:<38} {p.mean():.3f}   share above 0.5: {np.mean(p > 0.5):.1%}")
    shown = set()
    for v in data["impressions"].values():
        shown.update(v.tolist())
    print(f"  distinct items that ever serve as impression negatives: {len(shown):,} of {n_items:,}")

    # ---- section 6.6: model B, trained only on satisfied plays; and three seeds of MAIN
    bu, bi = satisfied_pairs(events, users, 23 * DAY, n_users, n_items, video)
    data_b = dict(data)
    data_b["pos_sets"] = data["pos_sets"].copy()
    data_b["pos_sets"][bu, bi] = True
    data_b["known"] = np.zeros(n_items, dtype=bool)
    data_b["known"][bi] = True
    mb = train_two_tower(data_b, bu, bi, **MAIN)
    U, V = embed_all(mb, data_b)
    sb = eligible_and_scored(U @ V.T, weight, ok)
    results["two-tower trained on satisfied plays"] = evaluate(lambda r: (sb[r], lang_pop(r)))
    print(f"\nPositive pairs: engagement (any positive implicit score) {len(pu):,}; "
          f"satisfied plays (completion or like, videos) {len(bu):,}")
    seed_runs = []
    for seed in (1, 2):
        m = train_two_tower(data, pu, pi, **MAIN, seed=seed)
        U, V = embed_all(m, data)
        s = eligible_and_scored(U @ V.T, weight, ok)
        seed_runs.append(evaluate(lambda r, s=s: (s[r], lang_pop(r))))
    for name, fn in (("hidden taste (simulation only)", match),
                     ("taste + popularity (simulation only)",
                      lambda r: 5 * match(r) + np.log(items["popularity"]))):
        results[name] = evaluate(fn)
    main = results["two-tower: mixed, logQ (MAIN)"]
    print("Seeds 0, 1, 2 of the MAIN model: NDCG@10 "
          + ", ".join(f"{t['NDCG@10']:.4f}" for t in [main] + seed_runs)
          + "; satisfied plays " + ", ".join(f"{t['online']:.4f}" for t in [main] + seed_runs))

    # ---- section 6.6: the honest test against the fixed baselines
    show = list(methods) + ["two-tower: mixed, logQ (MAIN)", "two-tower trained on satisfied plays",
                            "hidden taste (simulation only)", "taste + popularity (simulation only)"]
    print(f"\nSection 6.6: test week, video row, {len(rows):,} users; lists filled by popular in your language")
    print(f"  {'':<40}{'hit@10':>8}{'NDCG@10':>8}{'R@100':>8}{'R@200':>8}"
          f"{'coverage':>10}{'top-10% cr.':>12}{'online':>8}{'online, test':>13}")
    for name in show:
        t = results[name]
        print(f"  {name:<40}{t['hit@10']:>8.3f}{t['NDCG@10']:>8.3f}{t['R@100']:>8.3f}{t['R@200']:>8.3f}"
              f"{t['cov']:>10.1%}{t['creators']:>12.1%}{t['online']:>8.4f}{t['online_test']:>13.4f}")
    print("  guardrails on the lists all consenting users would see:")
    for name in ("iALS, then popular in language", "BPR popular negatives, then language",
                 "two-tower: mixed, logQ (MAIN)", "two-tower trained on satisfied plays"):
        t = results[name]
        print(f"    {name:<40} coverage {t['cov_all']:6.1%}   top-10% creators {t['creators_all']:6.1%}")
    hit = {n: {u for u, t in zip(rows.tolist(), results[n]["full"][:, :10]) if pos[u] & set(t.tolist())}
           for n in ("two-tower: mixed, logQ (MAIN)", "iALS, then popular in language")}
    a_only = len(hit["two-tower: mixed, logQ (MAIN)"] - hit["iALS, then popular in language"])
    b_only = len(hit["iALS, then popular in language"] - hit["two-tower: mixed, logQ (MAIN)"])
    print(f"  hit@10, two-tower against iALS: only two-tower {a_only}, only iALS {b_only}; "
          f"exact McNemar p = {mcnemar_exact(a_only, b_only):.2g}")
    per_user = {name: np.array([ndcg_at_k(t.tolist(), {i: 1.0 for i in pos[u]})
                                for u, t in zip(rows, results[name]["full"])])
                for name in ("popular in your language", "iALS, then popular in language",
                             "two-tower: mixed, logQ (MAIN)")}
    se = lambda x: x.std(ddof=1) / np.sqrt(len(x))                                  # noqa: E731
    for name in ("iALS, then popular in language", "two-tower: mixed, logQ (MAIN)"):
        h = np.array([u in hit[name] for u in rows.tolist()], dtype=float)
        print(f"  standard errors, {name}: hit@10 {h.mean():.3f} ({se(h):.3f}), "
              f"NDCG@10 {per_user[name].mean():.3f} ({se(per_user[name]):.3f})")
    lang_rows = users["language"][rows]
    print("  NDCG@10 by user language:", "  ".join(f"{g} ({(lang_rows == j).sum()})" for j, g in enumerate(LANGUAGES)))
    for name, v in per_user.items():
        print(f"    {name:<40}" + "".join(f"{v[lang_rows == j].mean():>8.3f}" for j in range(len(LANGUAGES))))
    diff = per_user["two-tower: mixed, logQ (MAIN)"] - per_user["iALS, then popular in language"]
    print(f"    {'two-tower minus iALS':<40}" + "".join(f"{diff[lang_rows == j].mean():>8.3f}"
                                                         for j in range(len(LANGUAGES))))
    print(f"    {'  standard error of the difference':<40}" + "".join(f"{se(diff[lang_rows == j]):>8.3f}"
                                                                     for j in range(len(LANGUAGES))))
    print(f"[{time.time() - t0:.0f} s]")
    del results, models, data, data_b, sb, U, V

    # ---- section 6.6: new videos - Chapter 3 and 4's history-masking test
    new = (np.random.default_rng(cfg.seed + 303).random(n_items) < 0.10) & video
    to01 = lambda x: (x - x.min(axis=1, keepdims=True)) / (np.ptp(x, axis=1, keepdims=True) + 1e-12)  # noqa: E731

    def masked_two_tower(w, id_dropout):
        """Train on data with the new videos' history hidden; their id vectors stay zero."""
        w_old = np.where(new[None, :], 0.0, w)
        d_old = make_data(w_old, items, users, feats, n_creators)
        po, pio = np.nonzero(w_old > 0)
        m_old = train_two_tower(d_old, po, pio, **MAIN, id_dropout=id_dropout)
        return m_old, embed_all(m_old, d_old), w_old

    def new_video_rates(rows_new, pos_new, used_, ranks):
        rng = np.random.default_rng(1)                      # the same draws as Chapters 3 and 4
        return {name: np.mean([hit_rate(top_k(b_ + 1e-3 * rng.random((len(rows_new), n_items)), used_[rows_new], new),
                                        rows_new, pos_new) for _ in range(20)]) for name, b_ in ranks}

    def only_new(positives):
        out = {u: s_ & set(np.flatnonzero(new).tolist()) for u, s_ in positives.items()}
        return {u: s_ for u, s_ in out.items() if s_}

    # id dropout, checked on the validation week (train to day 16, hits in days 16 to 23)
    pn16 = only_new(pos16)
    rn16 = np.array(sorted(pn16))
    lang16 = (users["language"][rn16, None] == items["language"][None, :]).astype(np.float64)
    print(f"\nNew videos, validation week: {len(rn16):,} users; hiding the id vector of a share of items in training")
    for p_drop in (0.0, 0.5):
        _, (Uv, Vv), _ = masked_two_tower(w16, p_drop)
        r_ = new_video_rates(rn16, pn16, used16, (("two-tower item tower", to01(Uv[rn16] @ Vv.T)),
                                                 ("in your language, by two-tower",
                                                  lang16 + 0.5 * to01(Uv[rn16] @ Vv.T))))
        print(f"  id dropout {p_drop}: " + "; ".join(f"{k} {v:.3f}" for k, v in r_.items()))

    m_old, (Uo, Vo), w_old = masked_two_tower(weight, 0.0)
    Xo, Yo = ials(w_old, 16, 1.0, 10.0)
    from implicit_mf import content_to_factors
    Yhat, _ = content_to_factors(feats, Yo, ~new & (engaged >= MIN_BUYERS))
    pos_new = only_new(pos)
    rn = np.array(sorted(pos_new))
    same_lang = (users["language"][rn, None] == items["language"][None, :]).astype(np.float64)
    content, _ = content_scores(w_old[rn], feats)
    pred01, tt01 = to01(Xo[rn] @ Yhat.T), to01(Uo[rn] @ Vo.T)
    print(f"\nNew videos, test week (history-masking test of Chapters 3 and 4): {new.sum()} videos, {len(rn):,} users; "
          f"two-tower item vectors from content only (largest id-vector entry of a new video: "
          f"{np.abs(m_old.item_id.weight.detach().numpy()[new]).max():.1f})")
    rates = new_video_rates(rn, pos_new, used, (
        ("random order", 0.0), ("in your language, random order", same_lang),
        ("in your language, by content (Chapter 3)", same_lang + 0.5 * content),
        ("predicted vectors (Chapter 4)", pred01),
        ("in your language, by predicted vectors (Chapter 4)", same_lang + 0.5 * pred01),
        ("two-tower item tower", tt01), ("in your language, by two-tower", same_lang + 0.5 * tt01)))
    for name, v in rates.items():
        print(f"  {name:<52} hit rate@10 {v:.3f} (standard error {np.sqrt(v * (1 - v) / len(rn)):.3f})")
    del m_old, Uo, Vo, w_old, Xo, Yo, Yhat, content, pred01, tt01, same_lang

    # ---- section 6.6: new users - Chapter 4's onboarding test, without fold-in
    positive = [e for e, w in EVENT_WEIGHTS.items() if w > 0]
    good = np.isin(d23["event"], positive) & video[d23["item"]]
    firsts = {}
    for _, u, i, e in sorted(zip((-d23["age_days"][good]).tolist(), d23["user"][good].tolist(),
                                 d23["item"][good].tolist(), d23["event"][good].tolist())):
        f = firsts.setdefault(u, {})
        if i not in f and len(f) < 5:
            f[i] = float(np.log1p(EVENT_WEIGHTS[e]))
    onboard = np.array([u for u in rows if len(firsts.get(u, {})) >= 5])
    w_train = weight.copy()
    w_train[onboard] = 0.0
    d_tr = make_data(w_train, items, users, feats, n_creators)
    pt, pit = np.nonzero(w_train > 0)
    m_tr = train_two_tower(d_tr, pt, pit, **MAIN)
    _, Vh = embed_all(m_tr, d_tr)
    _, Yh = ials(w_train, 16, 1.0, 10.0)
    YtY = Yh.T @ Yh
    ok_h = video & ((w_train > 0).sum(axis=0) >= MIN_BUYERS)
    lang_h = group_popularity(w_train, users["language"], len(LANGUAGES))[users["language"]]
    print(f"\nNew users (Chapter 4's onboarding test): {len(onboard):,} held-out users; the two-tower user "
          f"tower reads their first videos directly, with no retraining and no fold-in")
    print(f"  {'videos known':<36}{'two-tower':>10}{'iALS fold-in':>14}{'chart':>8}")
    for n in (1, 3, 5, None):
        H = np.zeros((len(onboard), n_items), dtype=np.float32)
        Vf = np.zeros((len(onboard), 16))
        seen = np.zeros((len(onboard), n_items), dtype=bool)
        for r_, u in enumerate(onboard):
            if n is None:
                idx = np.flatnonzero(weight[u] > 0)
                H[r_, idx] = weight[u, idx]
                Vf[r_] = fold_in(Yh, YtY, idx, weight[u, idx], 1.0, 10.0)
                seen[r_] = used[u]
            else:
                idx, wts = list(firsts[u])[:n], list(firsts[u].values())[:n]
                H[r_, idx] = wts
                Vf[r_] = fold_in(Yh, YtY, idx, wts, 1.0, 10.0)
                seen[r_, idx] = True
        with torch.no_grad():
            Ut = m_tr.user_tower(history_rows(torch.as_tensor(H, dtype=torch.float32), torch.arange(len(H))),
                                 d_tr["lang"][onboard])
            Ut = torch.cat([Ut, torch.full((len(Ut), 1), m_tr.temperature)], dim=1) if m_tr.bias else Ut
        lists_ = [top_k(np.where(ok_h[None, :], Ut.numpy().astype(np.float64) @ Vh.T, NO), seen, video),
                  top_k(np.where(ok_h[None, :] & Vf.any(axis=1)[:, None], Vf @ Yh.T, NO), seen, video),
                  top_k(lang_h[onboard], seen, video)]
        rates = [np.mean([bool(pos[u] & set(t.tolist())) for u, t in zip(onboard, L)]) for L in lists_]
        label = "full history at day 23 (reference)" if n is None else f"first {n}, in time order"
        print(f"  {label:<36}{rates[0]:>10.3f}{rates[1]:>14.3f}{rates[2]:>8.3f}")
    print(f"[{time.time() - t0:.0f} s]")
