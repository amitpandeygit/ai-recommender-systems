"""Section 24.7 - what Tesselo's bandits log, in a form Chapter 25 can use directly.

Chapter 2 logged two of its four probabilities: the exploration rate and the chance of an
item within exploration. It could not log the third, the logging-policy probability (the
full chance that the system showed that item in that slot), because its ranker was not
randomised in a known way. A bandit is. Every decision it makes is logged with the chance
it gave every candidate.

One file per surface, chapter_24/cache/bandit_logs_<surface>.npz, made by fresh_content.py
(the slot-4 bandit of the home row, surface "home_slot4") and slate_bandits.py (the "New
for you" row, surface "new_for_you_row"). Arrays, one row per decision unless noted, in
Chapter 2's style: whole numbers for ids, -1 for "none", hidden truth only in fields whose
names start with "_".

  decision_id, request_id, session_id, user_id, anonymous_id, served_time (server clock),
  consent_at_time                        - who and when
  slot                                   - 1 to 10, or 0 for a whole row
  policy, policy_version, model_time     - the policy, and the time of the last update it
                                           used (all its data had reached the server by then)
  settings                               - one text: the policy's settings, floor and prior
  cand_ptr                               - candidates of decision d are rows
                                           cand_ptr[d]:cand_ptr[d + 1] of the cand_* arrays
  cand_item, cand_prob, cand_prob_se     - each candidate, the chance the policy gave it,
                                           and the Monte Carlo standard error of that chance
  cand_mean, cand_sd                     - the policy's predicted satisfied play and its
                                           spread, for each candidate
  cand_x                                 - the policy's base features for each candidate:
                                           the constant and the content gap
  week_row                               - home_slot4 only: the request's row in
                                           new_video_test.npz, whose hidden array holds the
                                           64 hidden numbers the neural-linear policy also
                                           read (logged by reference, not copied: 31,000
                                           decisions x 100 candidates x 64 numbers would be
                                           0.8 GB)
  _cand_p                                - hidden truth: the exact chance of a satisfied
                                           play for each candidate in that slot
  chosen_item, chosen_prob, chosen_pos   - what was shown, its chance, its place in the list
  prob_method, prob_draws                - "exact" or "monte_carlo", and the number of draws
  slate, slate_marginal                  - rows only: the 10 videos shown, and for every
                                           candidate the chance of being in each slot
                                           (candidates x 10, per decision, float16)
  snap_time, snap_kappa, snap_alpha,     - rows only: the policy itself, after every hourly
  snap_A0, snap_b0, snap_a, snap_B,        update (snap_time is its model_time). With it and
  snap_bv                                  cand_x, any chance of the logged policy can be
                                           recomputed by drawing, such as the chance that two
                                           videos are shown in two given slots together,
                                           which the slot marginals do not hold
  impression_id                          - the shown slots' impression ids
                                           (request_id x 10 + slot - 1, as in Chapter 2)
  events                                 - a separate table, prefix ev_, in Chapter 2's
                                           format: ev_event_id, ev_event, ev_impression_id,
                                           ev_request_id, ev_session_id, ev_user_id,
                                           ev_item_id, ev_value, ev_event_time (on the
                                           server's clock), ev_server_time (arrival),
                                           ev_consent_at_time

  load(surface)      - the log as a dict, with a check that it is complete
  candidates(L, d)   - the candidates of decision d as a dict of arrays
  check(L)           - the checks of section 24.7: chances sum to 1, every chance is at least
                       the floor, the chosen chance matches its candidate, and the chances
                       agree with how often the policy actually made each choice

Needs Python 3.12 or later. Imported by fresh_content.py and slate_bandits.py; run from the
repository root to check both logs:
    python chapter_24/bandit_logs.py
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
VERSION = "chapter-24.1"
SURFACES = ("home_slot4", "new_for_you_row")
EVENT_NAMES = ("viewable", "click", "watch_end", "complete", "skip", "like", "not_interested")


class Logger:
    """Collects decisions while a policy runs; save() writes the file."""
    def __init__(self, surface, policy, settings):
        self.surface, self.policy, self.settings = surface, policy, settings
        self.rows, self.snaps = [], []

    def add(self, **row):
        self.rows.append(row)

    def snapshot(self, t, pol):
        """Keep the policy's state as of model time t (once per update)."""
        if not self.snaps or self.snaps[-1][0] != t:
            self.snaps.append((t, pol.A0.copy(), pol.b0.copy(), pol.a.copy(), pol.B.astype(np.float32),
                               pol.bv.copy()))
        self.kappa, self.alpha = pol.kappa, pol.alpha

    def save(self, events):
        R = self.rows
        n = len(R)
        sizes = np.array([len(r["cand_item"]) for r in R])
        out = {"surface": np.array(self.surface), "policy": np.array(self.policy),
               "policy_version": np.array(VERSION), "settings": np.array(self.settings),
               "decision_id": np.arange(n), "cand_ptr": np.r_[0, np.cumsum(sizes)]}
        for key in ("request_id", "session_id", "user_id", "anonymous_id", "served_time", "consent_at_time",
                    "slot", "model_time", "chosen_item", "chosen_prob", "chosen_pos", "prob_method",
                    "prob_draws") + (("week_row",) if "week_row" in R[0] else ()):
            out[key] = np.array([r[key] for r in R])
        for key in ("cand_item", "cand_prob", "cand_prob_se", "cand_mean", "cand_sd", "_cand_p"):
            out[key] = np.concatenate([np.asarray(r[key]) for r in R])
        out["cand_x"] = np.concatenate([np.asarray(r["cand_x"], dtype=np.float32) for r in R])
        if "slate" in R[0]:
            out["slate"] = np.stack([r["slate"] for r in R])
            out["slate_marginal"] = np.concatenate([np.asarray(r["slate_marginal"], dtype=np.float16) for r in R])
            out["impression_id"] = out["request_id"][:, None] * 10 + np.arange(10)[None, :]
        else:
            out["impression_id"] = out["request_id"] * 10 + out["slot"] - 1
        if self.snaps:
            for j, key in enumerate(("time", "A0", "b0", "a", "B", "bv")):
                out[f"snap_{key}"] = np.stack([sn[j] for sn in self.snaps])
            out["snap_kappa"], out["snap_alpha"] = np.array(self.kappa), np.array(self.alpha)
        out.update({f"ev_{k}": v for k, v in events.items()})
        CACHE.mkdir(exist_ok=True)
        np.savez_compressed(CACHE / f"bandit_logs_{self.surface}.npz", **out)
        return out


def events_table(request_id, session_id, user_id, served_time, consent, slot, item, ev, duration, reconnect,
                 first_event_id=0):
    """Chapter 2-style events for shown slots: one row per event. The app logs viewable one
    second after the slot comes on screen (2.5 seconds per slot), a click 3.5 seconds after
    that, and the end of the watch (with seconds watched) when the share watched has passed.
    An event reaches the server about a second later, or when an offline phone reconnects."""
    t_screen = served_time + (slot - 1) * 2.5
    t_click = t_screen + 3.5
    t_end = t_click + ev["watch"] * duration
    times = {"viewable": t_screen + 1.0, "click": t_click, "watch_end": t_end, "complete": t_end,
             "skip": t_end, "like": t_end + 1.5, "not_interested": t_end + 3.0}
    flags = dict(ev)
    flags["watch_end"] = ev["click"] & (duration > 0)
    rows = []
    for name in EVENT_NAMES:
        m = np.flatnonzero(flags[name])
        rows.append((m, np.full(len(m), name, dtype=object), times[name][m]))
    idx = np.concatenate([r[0] for r in rows])
    t = np.concatenate([r[2] for r in rows])
    rc = reconnect[session_id[idx]]
    arrive = np.where(np.isnan(rc), t + 1.0, np.maximum(t + 1.0, np.nan_to_num(rc, nan=0.0)))
    name = np.concatenate([r[1] for r in rows]).astype(str)
    value = np.where(name == "watch_end", ev["watch"][idx] * duration[idx], 1.0)
    o = np.argsort(arrive, kind="stable")
    return {"event_id": first_event_id + np.arange(len(idx)), "event": name[o],
            "impression_id": (request_id[idx] * 10 + slot[idx] - 1)[o], "request_id": request_id[idx][o],
            "session_id": session_id[idx][o], "user_id": user_id[idx][o], "item_id": item[idx][o],
            "value": value[o], "event_time": t[o], "server_time": arrive[o], "consent_at_time": consent[idx][o]}


def load(surface):
    f = CACHE / f"bandit_logs_{surface}.npz"
    if not f.exists():
        raise SystemExit(f"{f} is missing: run chapter_24/fresh_content.py and chapter_24/slate_bandits.py first")
    z = np.load(f, allow_pickle=False)
    L = {k: z[k] for k in z.files}
    assert L["cand_ptr"][-1] == len(L["cand_item"]), "the candidate table is incomplete"
    return L


def candidates(L, d):
    a, b = L["cand_ptr"][d], L["cand_ptr"][d + 1]
    out = {k: L[k][a:b] for k in ("cand_item", "cand_prob", "cand_prob_se", "cand_mean", "cand_sd", "_cand_p",
                                  "cand_x")}
    if "slate_marginal" in L:
        out["slate_marginal"] = L["slate_marginal"][a:b].astype(np.float64)
    return out


def check(L, floor, bins=(0, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0001)):
    """The checks of section 24.7. Returns a dict of results and prints them."""
    ptr = L["cand_ptr"]
    d_of = np.repeat(np.arange(len(ptr) - 1), np.diff(ptr))
    sums = np.bincount(d_of, weights=L["cand_prob"])
    n = np.diff(ptr)
    out = {"decisions": len(n), "candidates": int(n.sum()),
           "max |sum - 1|": float(np.abs(sums - 1).max()),
           "smallest chance / (floor / n)": float((L["cand_prob"] / (floor / n[d_of])).min()) if floor else np.nan}
    if "slate" in L:
        sm = L["slate_marginal"].astype(np.float64)
        out["max |slot sums - 1|"] = float(np.abs(np.stack([np.bincount(d_of, weights=sm[:, k])
                                                          for k in range(sm.shape[1])], 1) - 1).max())
        shown = np.zeros((len(d_of), 10))
        pos_in = {}
        for d in range(len(n)):
            for k, it in enumerate(L["slate"][d]):
                pos_in[(d, int(it))] = k
        for j, (d, it) in enumerate(zip(d_of, L["cand_item"])):
            k = pos_in.get((int(d), int(it)))
            if k is not None:
                shown[j, k] = 1.0
        p, y = sm.ravel(), shown.ravel()
    else:
        chosen = L["cand_item"] == L["chosen_item"][d_of]
        first = np.zeros(len(d_of), dtype=bool)                  # the chosen item appears once per decision
        idx = np.flatnonzero(chosen)
        _, f = np.unique(d_of[idx], return_index=True)
        first[idx[f]] = True
        out["chosen chance matches its candidate"] = bool(np.allclose(L["cand_prob"][first], L["chosen_prob"]))
        p, y = L["cand_prob"], first.astype(float)
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum():
            rows.append((lo, hi, int(m.sum()), float(p[m].mean()), float(y[m].mean()),
                         float(np.sqrt(max(p[m].mean() * (1 - p[m].mean()), 1e-12) / m.sum()))))
    out["calibration"] = rows
    return out


def recompute(L, decisions, draws=1_000, seed=0):
    """Rows only: redraw each decision's slot marginals from the logged policy snapshot, and
    compare them with the logged marginals. Returns the mean absolute difference and the
    mean absolute difference expected from Monte Carlo noise alone (two runs of draws)."""
    sys.path.insert(0, str(HERE))
    from policies import HybridLinear                                   # noqa: E402
    floor = float(str(L["settings"]).split("floor=")[1].split(";")[0])
    rng = np.random.default_rng(seed)
    k = L["cand_x"].shape[1]
    pol = HybridLinear(k, L["snap_a"].shape[1], mode="ts", alpha=float(L["snap_alpha"]), kappa=float(L["snap_kappa"]))
    diffs, noise = [], []
    for d in decisions:
        j = int(np.searchsorted(L["snap_time"], L["model_time"][d], side="right") - 1)
        pol.A0, pol.b0, pol.a, pol.B, pol.bv = (L[f"snap_{x}"][j].astype(np.float64) for x in ("A0", "b0", "a", "B", "bv"))
        pol.refresh()
        c = candidates(L, d)
        n, slots = len(c["cand_item"]), L["slate"].shape[1]
        marg = []
        for _ in range(2):
            top = np.argsort(-pol.scores_many(c["cand_x"].astype(np.float64), c["cand_item"], rng, draws),
                             axis=1, kind="stable")[:, :slots]
            m = np.zeros((n, slots))
            np.add.at(m, (top, np.broadcast_to(np.arange(slots), top.shape)), 1.0 / draws)
            marg.append((1 - floor) * m + floor / n)
        diffs.append(np.abs(marg[0] - c["slate_marginal"]).mean())
        noise.append(np.abs(marg[0] - marg[1]).mean())
    return float(np.mean(diffs)), float(np.mean(noise))


def report(name, res):
    print(f"  {name}: {res['decisions']:,} decisions, {res['candidates']:,} candidates; largest |sum of chances - 1| "
          f"{res['max |sum - 1|']:.2e}" + (f"; largest |sum over candidates per slot - 1| {res['max |slot sums - 1|']:.2e}"
                                          if "max |slot sums - 1|" in res else "")
          + (f"; smallest chance / (floor / candidates) {res['smallest chance / (floor / n)']:.3f}"
             if res["smallest chance / (floor / n)"] == res["smallest chance / (floor / n)"] else "")
          + (f"; chosen chance matches: {res['chosen chance matches its candidate']}"
             if "chosen chance matches its candidate" in res else ""))
    print("    logged chance against how often the policy chose (bins of the logged chance):")
    for lo, hi, m, pm, ym, se in res["calibration"]:
        print(f"      [{lo:g}, {hi:g}): {m:>10,} pairs, mean logged chance {pm:.5f}, share chosen {ym:.5f} "
              f"(expected spread {se:.5f})")


if __name__ == "__main__":
    for s in SURFACES:
        L = load(s)
        report(s, check(L, float(str(L["settings"]).split("floor=")[1].split(";")[0])))
        if "snap_time" in L:
            pick = np.random.default_rng(1).choice(len(L["decision_id"]), 300, replace=False)
            got, noise = recompute(L, pick)
            print(f"    redrawn from the logged policy snapshots, 300 decisions: mean |redrawn - logged| slot chance "
                  f"{got:.2e}; between two redraws {noise:.2e}")
    sys.exit()
