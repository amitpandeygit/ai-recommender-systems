"""One model in place of retrieval and ranking, session-wise generation, advertising, and
the honest test (sections 14.4 to 14.8).

Every test answers the held-out users' requests of days 23 to 30 in Chapter 12's scale
world (gen_data.requests), from the clicks each user had made before the request. Each
row has 10 videos and may not show a video the user has already clicked. It is scored
online from the hidden taste at the request (Chapter 5's expected satisfied plays) and
offline by NDCG@10 against the videos the user satisfied-played in that request, and
it is checked against Chapter 5's guardrails: at least 10% of the 3,480 videos shown,
and at most 60% of position-weighted exposure to the top 10% of creators.

  cascade   - Tesselo's funnel in miniature: iALS (Chapter 4) retrieves 500 videos and
              a ranker in the style of Chapter 7 orders them by P(click) x P(satisfied
              play | click). The ranker reads Chapter 7's eight dense features, with the
              user's features from the history at the request and the video's from the
              last weekly snapshot (days 9, 16, 23), and is trained on the shown rows of a
              quarter of the training users (gen_data)
  baselines - Chapter 5's fixed baselines, re-run in this world: random, most clicked,
              popular worldwide, popular in your language, CF then popular in language,
              iALS then popular in language
  arms      - the cascade, with iALS's 500 or 100 videos; the cascade with half its pool
              from the generator (section 14.4); Chapter 13's model alone, scoring every
              video; the generator alone (section 14.5); the generator with the ranker as
              a reward model choosing 10 of its 100 or 500 beams (OneRec's reward-model
              selection); the generator trained on satisfied plays only; and the list-wise
              generator trained on visits with a satisfied play, with the creator limit
              applied inside decoding (section 14.6). Every arm except the baselines keeps
              at most 2 videos per creator.
  ads       - section 14.7: an ad slot (slot 3) for sponsored videos, chosen by the
              ranker's P(click) x bid (the existing stack), its P(click) alone, or its
              P(click) x P(play | click) x bid; by the generator over a trie of sponsored
              videos; by the generator with value-aware selection (P x bid); by a generator
              trained with value-aware sample weights (GR4AD's VSL in miniature); with a
              beam that widens level by level (GR4AD's dynamic beam width); and by the
              ranker's P(click) x bid among the generator's 64 beams

Needs beam_runs.py's trained models (chapter_14/cache). Run from the repository root
(about 35 minutes on 2 CPUs; about 16 minutes once its own models are kept in the cache):
    python chapter_14/e2e_runs.py
"""
import os
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "2"
import gc
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.flop_counter import FlopCounterMode

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
for _ch in ("chapter_13", "chapter_12", "chapter_09", "chapter_07", "chapter_04", "chapter_03"):
    sys.path.insert(0, str(ROOT / _ch))
from gen_data import CFG, DAY, load, online, requests  # noqa: E402
from gen_models import (Generator, ListGenerator, Trie, beam_search, fit_generator,  # noqa: E402
                        generate_row, weighted_batches)
from beam_runs import CACHE, cached, generator, reference, setup, trained  # noqa: E402
from id_runs import boot  # noqa: E402
from large_models import tensors, windows  # noqa: E402
from scale_world import sequences  # noqa: E402
from implicit_mf import ials, solve_row  # noqa: E402
from similarity_recs import item_neighbours  # noqa: E402
from rankers import DenseMLP  # noqa: E402
from tesselo_shop import LANGUAGES, VIDEO_CATEGORIES  # noqa: E402
from seq_data import environment  # noqa: E402

ROW, CAP, POOL, REWARD_BEAMS, AD_SLOT = 10, 2, 500, 100, 3
K_ALS, REG, ALPHA, PRIOR = 16, 1.0, 10.0, 20            # Chapter 4's iALS, Chapter 7's smoothing
DENSE = ["iALS score", "popularity (log)", "click rate (log)", "plays per click (log)", "same language",
         "share of history in category", "history size (log)", "has history"]


# ---------------------------------------------------------------- clicks, with their satisfied plays
def kept_clicks(R, W, new):
    """Every click of a consenting user except the new videos' clicks before day 23
    (Chapter 13's rule), with its video number, request time, visit and satisfied play."""
    video = np.flatnonzero(~W["is_product"])
    renum = np.full(len(W["is_product"]), -1)
    renum[video] = np.arange(len(video))
    v = renum[W["item"]]
    keep = ~(new[v] & (W["time"] < 23 * DAY))
    return {"user": W["user"][keep], "video": v[keep], "time": W["time"][keep],
            "request_time": W["request_time"][keep], "session": W["session"][keep],
            "request": R["request"][keep], "play": R["play"][keep], "video_items": video}


def click_windows(C, W, k=64):
    """Chapter 13's training and validation windows (sid_data.data), built from the
    positions of the clicks instead of their videos, so that each target's satisfied play
    and visit are known. The videos are checked against Chapter 13's windows."""
    n = len(C["user"])
    V = {"user": C["user"], "item": np.arange(n), "time": C["time"], "request_time": C["request_time"],
         "session": C["session"], "is_product": np.zeros(n, dtype=bool), "n_users": W["n_users"]}
    S = sequences(V)
    return windows(*S["train"], k), windows(*S["held"], k, hi=23 * DAY)


def as_videos(win, C):
    return [[C["video"][x] for x in win[0]], win[1], win[2], win[3], [C["video"][y] for y in win[4]], win[5]]


def play_only(win, C):
    """The same windows, with only the targets that ended in a satisfied play scored."""
    v = as_videos(win, C)
    v[5] = [ok & C["play"][y] for ok, y in zip(win[5], win[4])]
    keep = [i for i, ok in enumerate(v[5]) if ok.any()]
    return [[p[i] for i in keep] for p in v]


def session_lists(win, C, quality=True, m=ROW):
    """Section 14.6. For each place a row is predicted, the list of videos the user went
    on to click in the rest of that visit (up to m, in order), kept only if one of them
    ended in a satisfied play (OneRec's high-quality sessions). Returns windows with one
    target per list (its first video) and the lists, aligned."""
    out, lists = [[], [], [], [], [], []], []
    for x, t, mm, at, y, ok in zip(*win):
        rows, firsts = [], []
        for p in np.unique(at[ok]):
            c0 = y[ok][at[ok] == p].min()
            yy = y[ok]                                          # scored targets only: no later period
            later = yy[(C["session"][yy] == C["session"][c0]) & (C["time"][yy] >= C["time"][c0])]
            later = np.sort(later)[:m]
            if quality and not C["play"][later].any():
                continue
            rows.append(np.r_[C["video"][later], np.full(m - len(later), -1)])
            firsts.append(p)
        if rows:
            for part, val in zip(out, (C["video"][x], t, mm, np.array(firsts), np.array(rows)[:, 0],
                                       np.ones(len(rows), dtype=bool))):
                part.append(val)
            lists.append(np.array(rows))
    return out, lists


def fit_lists(model, train, tl, valid, vl, epochs=30, lr=1e-3, batch=128, seed=0, patience=2):
    """Adam on the list loss (ListGenerator.list_loss), keeping the best epoch."""
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def run(data, lists, g=None):
        for x, t, tn, where, y, L in weighted_batches(data, None if lists is None else lists, model.pad, batch, g):
            yield model.list_loss(model.encode(x, t, tn)[where], L.long())

    curve, best, state = [], math.inf, None
    for e in range(1, epochs + 1):
        model.train()
        for loss in run(train, [l.astype(np.float32) for l in tl], gen):
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vals = [loss.item() for loss in run(valid, [l.astype(np.float32) for l in vl])]
        curve.append(float(np.mean(vals)))
        if curve[-1] < best:
            best, state = curve[-1], {k: p.detach().clone() for k, p in model.state_dict().items()}
        elif e - (int(np.argmin(curve)) + 1) >= patience:
            break
    model.load_state_dict(state)
    model.eval()
    return model, curve


# ---------------------------------------------------------------- the requests
def histories(C, Q, k=64):
    """Each request's history: the user's last k clicks made before the request, with
    their times and the moment each position predicts at (the request time of the click
    after it, and the request itself at the last position), as Chapter 12's windows."""
    starts = np.searchsorted(C["user"], np.arange(int(C["user"].max()) + 2))
    out = []
    for u, t in zip(Q["user"], Q["time"]):
        a = starts[u]
        b = a + np.searchsorted(C["time"][a:starts[u + 1]], t, side="left")
        s = slice(max(a, b - k), b)
        out.append((C["video"][s], C["time"][s], np.r_[C["request_time"][s][1:], t] if b > a else np.zeros(0),
                    C["video"][a:b]))
    return out


@torch.no_grad()
def request_states(model, H, batch=256):
    """The encoder's state at each request with a history (zeros otherwise)."""
    model.eval()
    model.uploaded = True
    out = torch.zeros(len(H), model.d)
    has = np.array([len(h[0]) > 0 for h in H])
    idx = np.flatnonzero(has)
    idx = idx[np.argsort([len(H[i][0]) for i in idx], kind="stable")]
    for a in range(0, len(idx), batch):
        sel = idx[a:a + batch]
        part = ([H[i][0] for i in sel], [H[i][1] for i in sel], [H[i][2] for i in sel],
                [np.array([len(H[i][0]) - 1]) for i in sel], [np.zeros(1, dtype=np.int64) for _ in sel],
                [np.ones(1, dtype=bool) for _ in sel])
        x, t, tn, where, _ = tensors(*part, model.pad)
        out[torch.as_tensor(sel)] = model.encode(x, t, tn)[where]
    model.uploaded = False
    return out, has


def banned_matrix(H, sel, V):
    b = torch.zeros(len(sel), V, dtype=torch.bool)
    for r, i in enumerate(sel):
        b[r, torch.as_tensor(H[i][3], dtype=torch.long)] = True
    return b


# ---------------------------------------------------------------- the ranker (Chapter 7 in miniature)
def snapshot(C, train_users, R, day, V, category):
    """What the ranker may use at the start of `day`: iALS item vectors, popularity, and
    each video's smoothed click rate and plays per click over the last 30 days, from the
    training users' clicks and shown rows."""
    m = train_users[C["user"]] & (C["time"] < day * DAY)
    users, inv = np.unique(C["user"][m], return_inverse=True)
    Wm = np.zeros((len(users), V), dtype=np.float32)
    np.add.at(Wm, (inv, C["video"][m]), 1.0)
    _, Y = ials(Wm, K_ALS, REG, ALPHA)
    engaged = (Wm > 0).sum(0)
    shown, clicked, played = (R["stats"][j][max(0, day - 30):day].sum(0) for j in range(3))
    ctr = (clicked + PRIOR * clicked.sum() / shown.sum()) / (shown + PRIOR)
    ppc = (played + PRIOR * played.sum() / clicked.sum()) / (clicked + PRIOR)
    return {"day": day, "Y": Y, "YtY": Y.T @ Y, "engaged": engaged, "ctr": ctr, "ppc": ppc, "W": Wm,
            "users": users, "category": category}


def user_side(S, hist):
    """The user's iALS vector (fold-in, Chapter 4) and share of history in each category,
    from the clicks before the request."""
    n_cat = int(S["category"].max()) + 1
    X = np.zeros((len(hist), K_ALS))
    share = np.zeros((len(hist), n_cat))
    size = np.zeros(len(hist))
    for r, h in enumerate(hist):
        if len(h):
            items, w = np.unique(h, return_counts=True)
            X[r] = solve_row(S["Y"], S["YtY"], items, w.astype(float), REG, ALPHA)
            share[r] = np.bincount(S["category"][h], minlength=n_cat) / len(h)
            size[r] = len(h)
    return X, share, size


def features(S, X, share, size, ulang, vids, vlang):
    """Chapter 7's eight dense features (ranking_data.pair_features) for each request's
    candidates (n x m videos)."""
    has = (size > 0)[:, None]
    f = [np.where(has, np.einsum("nk,nmk->nm", X, S["Y"][vids]), 0.0), np.log1p(S["engaged"][vids]),
         np.log(S["ctr"][vids]), np.log(S["ppc"][vids]), (ulang[:, None] == vlang[vids]).astype(float),
         np.take_along_axis(share, S["category"][vids], 1), np.broadcast_to(np.log1p(size)[:, None], vids.shape),
         np.broadcast_to(has, vids.shape).astype(float)]
    return np.stack(f, -1).astype(np.float32)


class Ranker(nn.Module):
    """Chapter 7's heavy ranker without id vectors (rankers.DenseMLP) and its two heads,
    P(click) and P(satisfied play | click), with a shallow position tower for the click."""
    def __init__(self, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.body = DenseMLP(None)
        self.click, self.play = nn.Linear(64, 1), nn.Linear(64, 1)
        self.position = nn.Embedding(ROW + 1, 1)
        nn.init.zeros_(self.position.weight)

    def forward(self, x, slot=None):
        h = self.body(None, None, x)
        c = self.click(h)[..., 0]
        if slot is not None:
            c = c + self.position(slot)[..., 0] - self.position.weight[1, 0]
        return c, self.play(h)[..., 0]

    def score(self, x):
        c, p = self(x)
        return torch.sigmoid(c) * torch.sigmoid(p)


def ranker_rows(R, C, snaps, lo, hi, ulang, vlang):
    """The shown rows of [lo, hi) days, with each row's features: the video's from the
    last snapshot before the row, the user's from the clicks before the request."""
    m = (R["shown_time"] >= lo * DAY) & (R["shown_time"] < hi * DAY)
    u, v, t = R["shown_user"][m], R["shown_video"][m], R["shown_time"][m]
    key, inv = np.unique(np.stack([u, t]), axis=1, return_inverse=True)
    Q = {"user": key[0].astype(np.int64), "time": key[1]}
    H = histories(C, Q)
    out = np.zeros((len(u), len(DENSE)), dtype=np.float32)
    S = max((s for s in snaps if s["day"] * DAY <= lo * DAY), key=lambda s: s["day"])
    X, share, size = user_side(S, [h[3] for h in H])
    inv = inv.ravel()
    out[:] = features(S, X[inv], share[inv], size[inv], ulang[u], v[:, None], vlang)[:, 0]
    return out, R["shown_slot"][m], R["shown_click"][m], R["shown_play"][m]


def fit_ranker(x, slot, click, play, epochs, seed=0, batch=1024, lr=1e-3, valid=None):
    m = Ranker(seed)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)
    X, S = torch.as_tensor(x), torch.as_tensor(slot.astype(np.int64))
    Cl, Pl = torch.as_tensor(click.astype(np.float32)), torch.as_tensor(play.astype(np.float32))
    curve = []
    for e in range(epochs):
        m.train()
        for b in torch.split(torch.randperm(len(X), generator=gen), batch):
            c, p = m(X[b], S[b])
            ck = Cl[b] > 0
            loss = F.binary_cross_entropy_with_logits(c, Cl[b])
            if ck.any():
                loss = loss + F.binary_cross_entropy_with_logits(p[ck], Pl[b][ck])
            opt.zero_grad()
            loss.backward()
            opt.step()
        if valid is not None:
            m.eval()
            with torch.no_grad():
                vx, vs, vc, vp = (torch.as_tensor(a) for a in valid)
                c, p = m(vx, vs.long())
                ck = vc > 0
                curve.append((F.binary_cross_entropy_with_logits(c, vc.float())
                              + F.binary_cross_entropy_with_logits(p[ck], vp[ck].float())).item())
    m.eval()
    return m, curve


# ---------------------------------------------------------------- rows and measures
def row_checks(rows, banned, creator, cap=CAP):
    """Checks of each row, independent of how it was made: slots holding a banned video,
    repeated videos, rows with more than `cap` videos of one creator, and empty slots."""
    rows = np.asarray(rows)
    ok = rows >= 0
    shown = np.zeros(rows.shape, dtype=bool)
    shown[ok] = banned.numpy()[np.nonzero(ok)[0], rows[ok]]
    dup = over = 0
    for r in rows:
        r = r[r >= 0]
        dup += len(r) - len(np.unique(r))
        over += int(len(r) > 0 and np.bincount(creator[r]).max() > cap)
    return np.array([shown.sum(), dup, over, (~ok).sum()])


def cap_rows(cands, creator, k=ROW, cap=CAP):
    """The first k candidates (in order) with at most `cap` videos per creator."""
    out = np.full((len(cands), k), -1)
    for r, c in enumerate(cands):
        n, seen = 0, {}
        for v in c:
            if v < 0:
                continue
            cr = creator[v]
            if seen.get(cr, 0) >= cap:
                continue
            seen[cr] = seen.get(cr, 0) + 1
            out[r, n] = v
            n += 1
            if n == k:
                break
    return out


def reward_pick(ranker, S, user, pool, vlang, creator):
    """Listing 14.4. OneRec's reward-model selection in miniature: the cascade's ranker,
    P(click) x P(satisfied play | click), scores each request's candidates (n x m videos,
    -1 = none), here the generator's beams, and the row takes the best, at most 2 per
    creator."""
    f = features(S, *user, np.maximum(pool, 0), vlang)
    with torch.no_grad():
        s = ranker.score(torch.as_tensor(f)).numpy()
    s = np.where(pool >= 0, s, -np.inf)
    order = np.argsort(-s, axis=1, kind="stable")
    return cap_rows(np.take_along_axis(pool, order, 1), creator)


def value_pick(v, score, bid):
    """Listing 14.6. The ad among each request's beams (videos v, log probabilities
    score) with the highest value score, the generator's probability times the bid. The
    probability is of the next video, not a calibrated click rate, so this ranks by
    value; it does not estimate revenue."""
    val = torch.exp(score) * torch.as_tensor(bid, dtype=torch.float32)[v.clamp(min=0)]
    val = torch.where(v >= 0, val, -1.0)
    return torch.gather(v, 1, val.argmax(1, keepdim=True))[:, 0]


def top_allowed(scores, banned, k):
    s = scores.masked_fill(banned, -math.inf)
    v = s.topk(k, dim=1)
    return torch.where(torch.isfinite(v.values), v.indices, -1)


def ndcg(top, rel):
    """NDCG@10 against the videos satisfied-played in each request (requests with one)."""
    out = []
    for row, r in zip(top, rel):
        if not r:
            continue
        dcg = sum(1 / math.log2(p + 2) for p, v in enumerate(row) if v in r)
        idcg = sum(1 / math.log2(p + 2) for p in range(min(len(r), len(row))))
        out.append(dcg / idcg)
    return float(np.mean(out))


def guardrails(top, creator, top_creators, V):
    shown = np.unique(top[top >= 0])
    w = 1 / np.log2(np.arange(2, top.shape[1] + 2))[None].repeat(len(top), 0)
    w = np.where(top >= 0, w, 0.0)
    cr = creator[np.maximum(top, 0)]
    return len(shown) / V, (w * np.isin(cr, top_creators)).sum() / w.sum()


# ---------------------------------------------------------------- the run
if __name__ == "__main__":
    t0 = time.time()
    parts = sys.argv[1:] or ["arms", "ads"]
    print(environment())
    R, W = load()
    _, D, new, sl_v, Z = setup()
    codes = Z["rq_vae_co_clicks"]
    V = D["n_videos"]
    C = kept_clicks(R, W, new)
    vid_items = C["video_items"]
    creator = W["creator"][vid_items]
    category = W["category"][vid_items]
    vlang = W["language"][vid_items]
    ulang = R["users_language"]
    held = R["held"]
    train_users = ~held & R["consent"]
    trw, vaw = click_windows(C, W)
    assert all(np.array_equal(a, b) for a, b in zip(as_videos(trw, C)[0][:2000], D["train"][0][:2000]))
    assert sum(len(y) for y in trw[4]) == sum(len(y) for y in D["train"][4])

    # models: beam_runs.py's, and three more generators
    ref = reference(D, new, codes)[0]
    gen = generator("generator", D, new, codes)[0]
    gplay, c_play, s_play = trained("generator_plays", lambda: Generator(V, codes, new),
                                    lambda m: fit_generator(m, play_only(trw, C), play_only(vaw, C)))
    tl_data, tl = session_lists(trw, C)
    vl_data, vl = session_lists(vaw, C)
    lens = np.concatenate([(l >= 0).sum(1) for l in tl])
    print(f"\nSession lists for the list-wise generator: {len(lens):,} training lists with a satisfied play "
          f"(mean length {lens.mean():.2f}; {np.mean(lens >= 2):.1%} with 2 or more videos)")
    glist, c_list, s_list = trained("generator_lists", lambda: ListGenerator(V, codes, new),
                                    lambda m: fit_lists(m, tl_data, tl, vl_data, vl))
    print(f"  generator trained on satisfied plays only: validation " + " ".join(f"{v:.4f}" for v in c_play)
          + f"  [{s_play:.0f} s]")
    print(f"  list-wise generator: validation " + " ".join(f"{v:.4f}" for v in c_list) + f"  [{s_list:.0f} s]")

    # the cascade's ranker
    def build_snaps():
        return [snapshot(C, train_users, R, d, V, category) for d in (9, 16, 23)]
    snaps, s_snap = cached("snapshots", build_snaps)
    for S in snaps[:2]:
        S.pop("W")                                             # only day 23's click matrix is needed below
    print(f"\nSnapshots (iALS, video statistics) at days 9, 16, 23  [{s_snap:.0f} s]")

    def ranker_data():
        return {name: ranker_rows(R, C, snaps, lo, hi, ulang, vlang)
                for name, lo, hi in (("a", 9, 16), ("b", 16, 23))}
    RD, s_rd = cached("ranker_rows", ranker_data)
    print(f"  ranker rows: days 9 to 16 {len(RD['a'][0]):,}, days 16 to 23 {len(RD['b'][0]):,}  [{s_rd:.0f} s]")
    _, curve = fit_ranker(*RD["a"], 6, valid=RD["b"])
    epochs = int(np.argmin(curve)) + 1
    print(f"  ranker validation loss (click + play | click) after each epoch: " + " ".join(f"{v:.4f}" for v in curve)
          + f"; chosen {epochs}")
    allx = [np.concatenate([RD["a"][j], RD["b"][j]]) for j in range(4)]
    ranker, _ = fit_ranker(*allx, epochs)
    S23 = snaps[2]
    del RD, allx, trw, vaw, tl_data, tl, vl_data, vl            # training data no longer needed: free memory
    gc.collect()

    # the test requests
    Q = requests(R, 23, 30)
    H = histories(C, Q)
    has = np.array([len(h[3]) > 0 for h in H])
    rel_map = {}
    tm = (held[C["user"]]) & (C["time"] >= 23 * DAY) & C["play"]
    for r, v in zip(C["request"][tm], C["video"][tm]):
        rel_map.setdefault(int(r), set()).add(int(v))
    rel = [rel_map.get(int(r), set()) for r in Q["request"]]
    users_q = Q["user"]
    print(f"\nTest requests (held-out users, days 23 to 30): {len(Q['user']):,}; with a history "
          f"{has.mean():.1%}; with a satisfied play {np.mean([len(r) > 0 for r in rel]):.1%}")
    X_u, share_u, size_u = user_side(S23, [h[3] for h in H])
    lang_q = ulang[users_q]

    # popularity charts and CF, from the training users' clicks before day 23
    Wm = S23["W"]
    pop = (Wm > 0).sum(0).astype(float)
    pop_lang = np.zeros((len(LANGUAGES), V))
    ul = ulang[S23["users"]]
    for g in range(len(LANGUAGES)):
        pop_lang[g] = (Wm[ul == g] > 0).sum(0)
    sim = item_neighbours(Wm)
    rng = np.random.default_rng(0)
    trie = Trie(codes, np.ones(V, dtype=bool), late=new)
    beam_w = int((CACHE / "beam_width.txt").read_text())    # chosen by beam_runs.py on the validation clicks
    states = {name: request_states(m, H)[0] for name, m in
              (("ref", ref), ("gen", gen), ("gplay", gplay), ("glist", glist))}

    B = 1024                                                   # requests per batch
    if "arms" in parts:
        arms, checks = {}, {}
        for a in range(0, len(Q["user"]), B):
            sel = np.arange(a, min(a + B, len(Q["user"])))
            ban = banned_matrix(H, sel, V)
            lq = lang_q[sel]
            fallback = top_allowed(torch.as_tensor(pop_lang[lq]), ban, POOL).numpy()   # no history: the language chart
            nh = ~has[sel]

            def put(name, rows):
                rows = np.where(nh[:, None], fallback[:, :ROW], rows)
                arms.setdefault(name, []).append(rows)
                checks[name] = checks.get(name, 0) + row_checks(rows, ban, creator)

            put("random", top_allowed(torch.as_tensor(rng.random((len(sel), V))), ban, ROW).numpy())
            put("most clicked", top_allowed(torch.as_tensor(np.broadcast_to(S23["ctr"], (len(sel), V)).copy()), ban,
                                             ROW).numpy())
            put("popular worldwide", top_allowed(torch.as_tensor(np.broadcast_to(pop, (len(sel), V)).copy()), ban,
                                                  ROW).numpy())
            put("popular in your language", fallback[:, :ROW])
            hw = np.zeros((len(sel), V))
            for r, i in enumerate(sel):
                np.add.at(hw[r], H[i][3], 1.0)
            cf = hw @ sim
            put("CF, then popular in language",                    # CF's videos first, then the language chart
                top_allowed(torch.as_tensor(np.where(cf > 0, 1e6 + cf, pop_lang[lq])), ban, ROW).numpy())
            als = X_u[sel] @ S23["Y"].T
            als_chain = np.where(nh[:, None], pop_lang[lq], als)
            put("iALS, then popular in language", top_allowed(torch.as_tensor(als_chain), ban, ROW).numpy())

            # the cascade: iALS's 500, ordered by the ranker
            pool = top_allowed(torch.as_tensor(als), ban, POOL).numpy()

            u = (X_u[sel], share_u[sel], size_u[sel], lq)        # the user's side of the features

            put("cascade: iALS 500, ranker", reward_pick(ranker, S23, u, pool, vlang, creator))
            put("cascade: iALS 100, ranker", reward_pick(ranker, S23, u, pool[:, :REWARD_BEAMS], vlang, creator))
            hs = {k: v[torch.as_tensor(sel)] for k, v in states.items()}
            g250 = beam_search(gen, hs["gen"], trie, (256, 256, 256, 256), ban)[0].numpy()
            half = []
            for r in range(len(sel)):
                c = list(pool[r][:250])
                s = set(c)
                for v in g250[r]:
                    if len(c) == POOL:
                        break
                    if v >= 0 and v not in s:
                        c.append(v)
                        s.add(v)
                for v in pool[r][250:]:
                    if len(c) == POOL:
                        break
                    if v not in s:
                        c.append(v)
                        s.add(v)
                half.append(np.array(c + [-1] * (POOL - len(c))))
            put("cascade, half the pool generated", reward_pick(ranker, S23, u, np.array(half), vlang, creator))
            ref.uploaded = True                                    # the test: the new videos exist
            with torch.no_grad():
                lr_ = ref.logits(hs["ref"])
            ref.uploaded = False
            put("Chapter 13's model alone, every video scored", cap_rows(top_allowed(lr_, ban, 30).numpy(), creator))
            gb = beam_search(gen, hs["gen"], trie, (beam_w,) * 3 + (max(beam_w, 30),), ban)[0].numpy()
            put("generator alone", cap_rows(gb, creator))
            gr = beam_search(gen, hs["gen"], trie, (REWARD_BEAMS,) * 4, ban)[0].numpy()
            put("generator, reward model picks 10 of 100", reward_pick(ranker, S23, u, gr, vlang, creator))
            g500 = beam_search(gen, hs["gen"], trie, (512, 512, 512, POOL), ban)[0].numpy()
            put("generator, reward model picks 10 of 500", reward_pick(ranker, S23, u, g500, vlang, creator))
            gp = beam_search(gplay, hs["gplay"], trie, (beam_w,) * 3 + (max(beam_w, 30),), ban)[0].numpy()
            put("generator trained on satisfied plays", cap_rows(gp, creator))
            put("list-wise generator, limit inside decoding",
                generate_row(glist, hs["glist"], trie, ban, torch.as_tensor(creator), beam=beam_w).numpy())
            print(f"  requests {sel[-1] + 1:,}  [{time.time() - t0:.0f} s]", flush=True)
        arms = {k: np.concatenate(v) for k, v in arms.items()}

        top_creators = np.argsort(-np.bincount(creator, weights=pop, minlength=int(creator.max()) + 1))[:30]
        user_of = users_q
        res = {}
        print(f"\nThe honest test: {len(users_q):,} requests; rows of {ROW}; online = expected satisfied plays per "
              f"row from the hidden taste at the request")
        print(f"  {'':<46}{'NDCG@10':>8}{'online':>9}{'clicks':>8}{'bait':>7}{'new':>7}{'cover':>8}{'top10%':>8}"
              f"   against the cascade (95% interval)")
        base = online(arms["cascade: iALS 500, ranker"], Q, R, vid_items)[1].sum(1)
        for name, top in arms.items():
            click, play = online(top, Q, R, vid_items)
            cov, conc = guardrails(top, creator, top_creators, V)
            bait = R["clickbait"][vid_items][np.maximum(top, 0)][top >= 0].astype(bool).mean()
            nw = new[np.maximum(top, 0)][top >= 0].mean()
            res[name] = play.sum(1)
            d = ""
            if name != "cascade: iALS 500, ranker":
                m, lo, hi = boot(res[name], base, user_of, np.ones(len(base), dtype=bool))
                d = f"   {m:+.4f} ({lo:+.4f} to {hi:+.4f})"
            print(f"  {name:<46}{ndcg(top, rel):>8.4f}{play.sum(1).mean():>9.4f}{click.sum(1).mean():>8.3f}"
                  f"{bait:>7.1%}{nw:>7.1%}{cov:>8.1%}{conc:>8.1%}{d}", flush=True)
        print("  (NDCG@10: against the videos satisfied-played in each request, over requests with one; online: "
              "expected satisfied plays per row; clicks: expected clicks per row; bait: share of shown videos that are "
              "clickbait; new: share of shown videos uploaded on day 23; cover: share of the 3,480 videos shown at least "
              "once; top10%: the top 10% of creators' share of position-weighted exposure)")
        print(f"\nChecks of every row ({len(users_q) * ROW:,} slots): banned videos shown (already clicked), "
              f"repeated videos, rows with more than {CAP} videos of one creator, empty slots")
        for name, c in checks.items():
            print(f"  {name:<46}" + "".join(f"{int(x):>8,}" for x in c))
        print("  (the baselines before the cascade have no creator limit)")
        print("\nPaired differences in expected satisfied plays per row (95% interval, 1,000 draws of whole users):")
        for a_, b_ in (("generator, reward model picks 10 of 100", "cascade: iALS 100, ranker"),
                       ("generator, reward model picks 10 of 500", "cascade: iALS 500, ranker"),
                       ("generator, reward model picks 10 of 500", "generator, reward model picks 10 of 100"),
                       ("generator alone", "Chapter 13's model alone, every video scored"),
                       ("generator trained on satisfied plays", "generator alone"),
                       ("list-wise generator, limit inside decoding", "generator alone")):
            m, lo, hi = boot(res[a_], res[b_], user_of, np.ones(len(user_of), dtype=bool))
            print(f"  {a_} minus {b_}: {m:+.4f} ({lo:+.4f} to {hi:+.4f})")
        print("\nOnline, by app language:")
        print("  " + f"{'':<46}" + "".join(f"{l:>8}" for l in LANGUAGES))
        for name in ("cascade: iALS 500, ranker", "generator alone", "generator, reward model picks 10 of 100",
                     "generator, reward model picks 10 of 500", "list-wise generator, limit inside decoding"):
            print(f"  {name:<46}" + "".join(f"{res[name][lang_q == g].mean():>8.4f}" for g in range(len(LANGUAGES))))

    if "ads" in parts:
        ad_rng = np.random.default_rng(CFG.seed + 1500)
        sponsor = ad_rng.random(int(creator.max()) + 1) < 0.25            # creators who promote their videos
        bid = ad_rng.lognormal(0.0, 0.5, V)                                 # coins per click
        spons = sponsor[creator]
        print(f"\nAds (section 14.7): {sponsor.sum()} of {len(sponsor)} creators promote their videos; "
              f"{spons.sum():,} sponsored videos; bid per click: median {np.median(bid[spons]):.3f}, "
              f"mean {bid[spons].mean():.3f} coins; the ad takes slot {AD_SLOT}")
        wts = [np.where(spons[y], 1.0 + bid[y], 1.0) for y in D["train"][4]]
        gvsl, c_vsl, s_vsl = trained("generator_vsl", lambda: Generator(V, codes, new),
                                     lambda m: fit_generator(m, D["train"], D["valid"], weights=wts))
        print("  generator with value-aware sample weights (1 + bid for a sponsored target): validation "
              + " ".join(f"{v:.4f}" for v in c_vsl) + f"  [{s_vsl:.0f} s]")
        ad_trie = Trie(codes, spons, late=new)
        hv = request_states(gvsl, H)[0]
        hg = states["gen"]
        picks = {}

        for a in range(0, len(Q["user"]), B):
            sel = np.arange(a, min(a + B, len(Q["user"])))
            ban = banned_matrix(H, sel, V) | torch.as_tensor(~spons)[None]
            lq = lang_q[sel]
            cand = top_allowed(torch.zeros(len(sel), V), ban, int(spons.sum())).numpy()
            f = features(S23, X_u[sel], share_u[sel], size_u[sel], lq, np.maximum(cand, 0), vlang)
            with torch.no_grad():
                pc = torch.sigmoid(ranker(torch.as_tensor(f))[0]).numpy()
            pc = np.where(cand >= 0, pc, -1)
            picks.setdefault("ranker: P(click) x bid", []).append(
                np.take_along_axis(cand, (pc * bid[np.maximum(cand, 0)]).argmax(1)[:, None], 1)[:, 0])
            picks.setdefault("ranker: P(click) only", []).append(np.take_along_axis(cand, pc.argmax(1)[:, None], 1)[:, 0])
            with torch.no_grad():
                pp = ranker.score(torch.as_tensor(f)).numpy()
            pp = np.where(cand >= 0, pp, -1)
            picks.setdefault("ranker: P(click) x P(play | click) x bid", []).append(
                np.take_along_axis(cand, (pp * bid[np.maximum(cand, 0)]).argmax(1)[:, None], 1)[:, 0])
            s_sel = torch.as_tensor(sel)
            v, sc = beam_search(gen, hg[s_sel], ad_trie, (beam_w,) * 4, ban)
            picks.setdefault("generator: most probable", []).append(v[:, 0].numpy())
            picks.setdefault("generator: P x bid among its beams", []).append(value_pick(v, sc, bid).numpy())
            v, sc = beam_search(gvsl, hv[s_sel], ad_trie, (beam_w,) * 4, ban)
            picks.setdefault("value-weighted generator: P x bid", []).append(value_pick(v, sc, bid).numpy())
            v64 = beam_search(gen, hg[s_sel], ad_trie, (64,) * 4, ban)[0].numpy()
            f64 = features(S23, X_u[sel], share_u[sel], size_u[sel], lq, np.maximum(v64, 0), vlang)
            with torch.no_grad():
                p64 = torch.sigmoid(ranker(torch.as_tensor(f64))[0]).numpy()
            p64 = np.where(v64 >= 0, p64 * bid[np.maximum(v64, 0)], -1)
            picks.setdefault("generator's 64 beams, ranker's P(click) x bid", []).append(
                np.take_along_axis(v64, p64.argmax(1)[:, None], 1)[:, 0])
            for w in ((64, 64, 64, 64), (8, 16, 32, 64)):
                v, sc = beam_search(gvsl, hv[s_sel], ad_trie, w, ban)
                picks.setdefault(f"value-weighted generator, beams {'-'.join(map(str, w))}", []).append(
                    value_pick(v, sc, bid).numpy())
        nb = ~has
        allban = torch.cat([banned_matrix(H, np.arange(a, min(a + B, len(Q["user"]))), V)
                            for a in range(0, len(Q["user"]), B)]) | torch.as_tensor(~spons)[None]
        print(f"  {'':<52}{'revenue':>9}{'plays':>9}{'clicks':>8}{'bait':>7}   revenue against the ranker (95% interval)")
        base_rev = None
        for name, p in picks.items():
            p = np.concatenate(p)
            top = np.full((len(p), AD_SLOT), -1)
            top[:, AD_SLOT - 1] = p
            top[nb] = -1                                               # no history: no personalised ad
            click, play = online(top, Q, R, vid_items)
            rev = click[:, AD_SLOT - 1] * bid[np.maximum(top[:, AD_SLOT - 1], 0)]
            if base_rev is None:
                base_rev = rev
                d = ""
            else:
                m, lo, hi = boot(rev, base_rev, users_q, has)
                d = f"   {m:+.4f} ({lo:+.4f} to {hi:+.4f})"
            bait = R["clickbait"][vid_items][np.maximum(p[has], 0)].astype(bool).mean()
            print(f"  {name:<52}{rev[has].mean():>9.4f}{play[has, AD_SLOT - 1].mean():>9.4f}"
                  f"{click[has, AD_SLOT - 1].mean():>8.4f}{bait:>7.1%}{d}", flush=True)
        print("  (per request with a history: revenue = expected coins from the ad slot; plays and clicks: expected "
              "satisfied plays and clicks there; bait: share of ads that are clickbait)")
        print("  checks over requests with a history: ads that are banned (already clicked or not sponsored), "
              "empty ad slots")
        for name, p in picks.items():
            p = np.concatenate(p)[has]
            okp = p >= 0
            bad = int(allban[torch.as_tensor(np.nonzero(has)[0][okp]), torch.as_tensor(p[okp])].sum())
            print(f"  {name:<52}{bad:>8,}{int((~okp).sum()):>8,}")
        hb = hv[:64]
        for w in ((64, 64, 64, 64), (8, 16, 32, 64), (beam_w,) * 4):
            with FlopCounterMode(display=False) as fc:
                beam_search(gvsl, hb, ad_trie, w)
            print(f"  beam widths {'-'.join(map(str, w))}: {fc.get_total_flops() / 64:,.0f} FLOPs per request")
    print(f"[{time.time() - t0:.0f} s]")
