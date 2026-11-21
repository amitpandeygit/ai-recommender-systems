"""Reasons, headlines, and a generator with an evaluator (sections 17.6 and 17.7).

Two writing tasks, each in the reader's own language:

  reasons    - the one-line reason under a "Because you watched" video (Chapter 3's
               row). For 300 test requests of days 23 to 30, 50 in each app language,
               drawn at random (seed cfg.seed + 1740), the row's video is the allowed
               video closest to the request's history by Chapter 3's content similarity,
               and the anchor is the history video closest to it. The writer is told the
               two videos' concepts and the concepts they share, and nothing else. A
               title's generic words ("new", "best", "cheap") are a seller's claims, not
               facts, so they are left out of every fact list.
  headlines  - an ad headline for a sponsored product: 50 of the launched products,
               drawn at random, each in all six languages (300 headlines). The writer is
               told the product's category and its title's concepts.

Every text is checked by rules that the simulation can apply exactly (RULES):

  length      - at most 70 characters for a reason, 40 for a headline
  script      - Devanagari for Hindi, Arabic letters for Arabic, Latin letters otherwise
  faithful    - every catalogue concept it names (in its language or in English) is one
                of the concepts it was given
  no claim    - none of the words of a claim the platform cannot back: popular,
                trending, free, best, a discount, a number (only "2026" is allowed, and
                only when the title has it), and the sales phrases of listings.py

The rules check words, not the truth of sentences: "Clinically proven sunscreen" or
"Made in Italy" passes them when sunscreen is a given concept. They find the problems
they look for, and section 17.6 reports exactly that.

A template that fills a fixed sentence with the given concepts is the baseline. The
generator and evaluator loop (Listing 17.7) writes, checks, and on failure writes again
with the reasons for the failure, at most three times. Its evaluator is the rules, an
LLM critic, or both; the critic is judged against the rules on the cases the rules can
decide, which is how far it can be trusted on the cases they cannot.

Needs llm_cache (or CH17_RUN=1 and the models). From the repository root:
    python chapter_17/explain.py
"""
import os
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "2"
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from listings import CATEGORIES, GENERIC_WORDS, LANGUAGES, SALES  # noqa: E402
from title_words import LANGUAGE_NAMES, WORDS  # noqa: E402
import llm  # noqa: E402

GENERATOR = os.environ.get("CH17_WRITER", "Qwen3.5-4B")
CRITIC_MODEL = "Qwen3.5-4B"
MAX_ROUNDS = 3
LIMIT = {"reasons": 70, "headlines": 40}
N_PER_LANGUAGE, N_PRODUCTS = 50, 50
CLAIMS = {
    "en": ["popular", "trending", "viral", "everyone", "friends", "free", "best", "number one", "discount", "original",
           "guaranteed"],
    "hi": ["लोकप्रिय", "ट्रेंडिंग", "वायरल", "सब लोग", "दोस्त", "मुफ्त", "फ्री", "सबसे अच्छा", "नंबर वन", "छूट",
           "ओरिजिनल", "गारंटी"],
    "pt": ["popular", "em alta", "viral", "todo mundo", "amigos", "grátis", "melhor", "número um", "desconto",
           "original", "garantido"],
    "es": ["popular", "tendencia", "viral", "todo el mundo", "amigos", "gratis", "mejor", "número uno", "descuento",
           "original", "garantizado"],
    "id": ["populer", "trending", "viral", "semua orang", "teman", "gratis", "terbaik", "nomor satu", "diskon",
           "original", "dijamin"],
    "ar": ["شائع", "رائج", "الجميع", "أصدقاء", "مجاني", "مجانا", "أفضل", "رقم واحد", "خصم", "أصلي", "مضمون"],
}
TEMPLATE = {"reasons": {"en": "Because you watched: {}", "hi": "क्योंकि आपने देखा: {}", "pt": "Porque você assistiu: {}",
                        "es": "Porque viste: {}", "id": "Karena kamu menonton: {}", "ar": "لأنك شاهدت: {}"},
            "headlines": {"en": "{}", "hi": "{}", "pt": "{}", "es": "{}", "id": "{}", "ar": "{}"}}


# ---------------------------------------------------------------- the rules
def _norm(text):
    return " " + " ".join(re.sub(r"[^\wऀ-ॿ؀-ۿ]", " ", unicodedata.normalize("NFC", text.lower()))
                          .split()) + " "


def concepts_named(text, language):
    """The catalogue concepts that the text names, in its language or in English (forms of
    3 or more characters, matched as whole words, longest first, so that "كرة القدم",
    football, is not also read as "كرة", ball). Returns one set per word found: the
    concepts that word can mean ("bateria" is a battery or a drum kit)."""
    t = _norm(text)
    meanings = {}
    for tok, ws in WORDS.items():
        for f in {ws[language], ws[0]}:
            meanings.setdefault(_norm(f).strip(), set()).add(tok)
    found = []
    for f in sorted(meanings, key=len, reverse=True):
        if len(f) >= 3 and f" {f} " in t:
            found.append(meanings[f])
            t = t.replace(f" {f} ", " | ")
    return found


def script_ok(text, language):
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    lang = LANGUAGES[language]
    block = {"hi": lambda c: "ऀ" <= c <= "ॿ", "ar": lambda c: "؀" <= c <= "ۿ"}.get(
        lang, lambda c: c.isascii() or "À" <= c <= "ɏ")
    return np.mean([block(c) for c in letters]) >= 0.8


def claims_made(text, language, allowed):
    lang = LANGUAGES[language]
    t = _norm(text)
    bad = [c for c in CLAIMS[lang] + CLAIMS["en"] + SALES[lang] + SALES["en"] if f" {_norm(c).strip()} " in t]
    rest = text.replace("2026", "") if "2026" in allowed else text   # the year only when the title has it
    if re.search(r"\d", rest):
        bad.append("a number")
    return bad


def rules(task, text, language, allowed):
    """The problems the rules find: an empty list when the text passes."""
    problems = []
    if len(text) > LIMIT[task]:
        problems.append(f"longer than {LIMIT[task]} characters")
    if not script_ok(text, language):
        problems.append(f"not written in {LANGUAGE_NAMES[language]}")
    extra = [m for m in concepts_named(text, language) if not m & set(allowed)]
    if extra:
        problems.append("names something not given: " + ", ".join(sorted(WORDS[min(m)][0] for m in extra)))
    claims = claims_made(text, language, allowed)
    if claims:
        problems.append("makes a claim: " + ", ".join(claims))
    return problems


# ---------------------------------------------------------------- the writing tasks
def reason_tasks(W):
    """300 requests of the test week (50 per app language): the row's video, the anchor
    from the history, and the concepts each has."""
    from graph_data import phase
    from similarity_recs import content_scores, item_features
    S = W["snaps"][23]
    R, H, _, bad = phase(W, 23, S)
    users, items = W["users"], W["items"]
    n_i = len(items["vec"])
    feats = item_features(items, n_i)
    rng = np.random.default_rng(W["cfg"].seed + 1740)
    lang = users["language"][R["user"]]
    has = np.array([(h >= 0).any() for h in H])
    pick = [int(k) for g in range(len(LANGUAGES))
            for k in sorted(rng.choice(np.flatnonzero((lang == g) & has), N_PER_LANGUAGE, replace=False))]
    out = []
    for k in pick:
        h = np.unique(H[k][H[k] >= 0])
        weight = np.zeros((1, n_i))
        weight[0, h] = 1.0
        s = content_scores(weight, feats)[0][0]
        s[~W["video"] | bad[k]] = -np.inf
        s[h] = -np.inf
        rec = int(np.argmax(s))
        anchor = int(h[np.argmax(feats[h] @ feats[rec])])
        a, r = set(topical(str(t) for t in items["title"][anchor])), set(topical(str(t) for t in items["title"][rec]))
        out.append({"request": k, "language": int(lang[k]), "anchor": anchor, "item": rec,
                    "anchor_concepts": sorted(a), "item_concepts": sorted(r), "shared": sorted(a & r),
                    "anchor_category": int(items["category"][anchor]), "category": int(items["category"][rec])})
    return out


def headline_tasks(W):
    from cold_items import launches
    items = W["items"]
    rng = np.random.default_rng(W["cfg"].seed + 1741)
    prods = sorted(rng.choice(np.flatnonzero(launches(W)), N_PRODUCTS, replace=False))
    return [{"item": int(i), "language": g, "item_concepts": topical({str(t) for t in items["title"][i]}),
             "category": int(items["category"][i])} for i in prods for g in range(len(LANGUAGES))]


def allowed(task, t):
    return set(t["item_concepts"]) | (set(t["anchor_concepts"]) if task == "reasons" else set())


def topical(tokens):
    """A title's concepts without its generic words ("new", "best", "cheap", ...), which
    are a seller's claims rather than facts about the item."""
    return sorted(t for t in tokens if t not in GENERIC_WORDS)


def names(tokens, language):
    return ", ".join(WORDS[x][language] for x in tokens)


# ---------------------------------------------------------------- Listing 17.7: write, check, write again
WRITER = {
    "reasons": """You write the one-line reason shown under a recommended video in Tesselo's "Because you watched" row. Write it in the language you are given, in that language's usual script, in at most 70 characters. Use only the facts you are given: the video the user watched and the topics the two videos share. Never say that a video is popular, trending, new, free, the best, or liked by friends, and never add a topic that is not in the facts. Answer with the reason only.""",
    "headlines": """You write a short headline for a sponsored product in Tesselo's shop. Write it in the language you are given, in that language's usual script, in at most 40 characters. Use only the product's own concepts. Never claim a price, a discount, free delivery, that it is original, guaranteed or the best, and never use numbers. Answer with the headline only.""",
}
CRITIC = """You check short texts before Tesselo shows them to users. You are given the facts the writer was given, the rules, and the text. Answer pass true only if the text follows every rule: it is in the required language and script, within the length limit, names nothing that is not in the facts, and makes none of the forbidden claims. If it fails, give the main problem in a few words."""
TEXT_SCHEMA = {"reasons": {"type": "object", "properties": {"text": {"type": "string", "maxLength": 100}},
                           "required": ["text"]},
               "headlines": {"type": "object", "properties": {"text": {"type": "string", "maxLength": 60}},
                             "required": ["text"]}}
CRITIC_SCHEMA = {"type": "object", "properties": {"pass": {"type": "boolean"},
                                                  "problem": {"type": "string", "maxLength": 60}},
                 "required": ["pass", "problem"]}


def facts(task, t):
    g = t["language"]
    if task == "reasons":
        shared = names(t["shared"], 0) if t["shared"] else "none"
        return (f"Language: {LANGUAGE_NAMES[g]}\nThe user watched a {CATEGORIES[t['anchor_category']]} video about: "
                f"{names(t['anchor_concepts'], g)} (in English: {names(t['anchor_concepts'], 0)})\n"
                f"The recommended video is a {CATEGORIES[t['category']]} video about: {names(t['item_concepts'], g)} "
                f"(in English: {names(t['item_concepts'], 0)})\nTopics they share, in English: {shared}")
    return (f"Language: {LANGUAGE_NAMES[g]}\nThe product is in {CATEGORIES[t['category']]}. Its concepts: "
            f"{names(t['item_concepts'], g)} (in English: {names(t['item_concepts'], 0)})")


def template(task, t):
    g = t["language"]
    words = t["anchor_concepts"] if task == "reasons" else t["item_concepts"]
    return TEMPLATE[task][LANGUAGES[g]].format(names(words[:2], g))


def write(task, tasks, feedback=None):
    """One round of writing for every task; feedback[k] holds the problems of task k's
    previous text, or None for a first attempt."""
    reqs = []
    for k, t in enumerate(tasks):
        user = facts(task, t)
        if feedback is not None and feedback[k] is not None:
            prev, problems = feedback[k]
            user += f"\nYour previous text was: {prev}\nIt was rejected because: {'; '.join(problems)}\nWrite a new one."
        reqs.append((WRITER[task], user, TEXT_SCHEMA[task]))
    out = llm.ask_many(GENERATOR, f"write_{task}", reqs, max_tokens=64)
    return [(r["answer"] or {"text": ""})["text"].strip() for r in out], out


def criticise(task, tasks, texts, critic):
    rule_text = (f"Rules: in {{lang}}, at most {LIMIT[task]} characters, names only the concepts in the facts, and "
                 "makes no claim of popularity, trend, price, discount, free delivery, originality, guarantee or "
                 "being the best, and has no numbers.")
    reqs = [(CRITIC, facts(task, t) + "\n" + rule_text.format(lang=LANGUAGE_NAMES[t["language"]]) + f"\nText: {x}",
             CRITIC_SCHEMA) for t, x in zip(tasks, texts)]
    out = llm.ask_many(critic, f"critic_{task}", reqs, max_tokens=48)
    return [(bool(r["answer"]["pass"]), r["answer"]["problem"]) if r["answer"] else (False, "no answer")
            for r in out], out


def loop(task, tasks, evaluator, critic=None):
    """Listing 17.7. Write; check with the evaluator ("rules", "critic" or "both"); write
    again with the problems, at most MAX_ROUNDS times. Returns each task's final text,
    the round it was accepted in (None if never), and every request made."""
    texts, recs = write(task, tasks)
    final, accepted, made = list(texts), [None] * len(tasks), list(recs)
    pending = list(range(len(tasks)))
    for rnd in range(1, MAX_ROUNDS + 1):
        sub = [tasks[k] for k in pending]
        problems = {}
        if evaluator in ("rules", "both"):
            for k in pending:
                p = rules(task, final[k], tasks[k]["language"], allowed(task, tasks[k]))
                if p:
                    problems[k] = p
        if evaluator in ("critic", "both"):
            verdicts, crecs = criticise(task, sub, [final[k] for k in pending], critic)
            made += crecs
            for k, (ok, why) in zip(pending, verdicts):
                if not ok:
                    problems.setdefault(k, []).append(why)
        for k in pending:
            if k not in problems:
                accepted[k] = rnd
        pending = [k for k in pending if k in problems]
        if not pending or rnd == MAX_ROUNDS:
            break
        fb = [None] * len(tasks)
        for k in pending:
            fb[k] = (final[k], problems[k])
        new, recs = write(task, [tasks[k] for k in pending], [fb[k] for k in pending])
        made += recs
        for k, x in zip(pending, new):
            final[k] = x
    return final, accepted, made


def rule_summary(task, tasks, texts):
    p = [rules(task, x, t["language"], allowed(task, t)) for t, x in zip(tasks, texts)]
    has = lambda key: np.mean([any(q.startswith(key) for q in ps) for ps in p])   # noqa: E731
    return {"pass": np.mean([not ps for ps in p]), "length": has("longer"), "script": has("not written"),
            "unfaithful": has("names"), "claim": has("makes")}


def critic_check(task, tasks, texts, critic):
    """The critic's verdicts against the rules' on the same texts: how often it passes a
    text the rules fail (a missed problem) and fails a text the rules pass (a false alarm)."""
    verdicts, recs = criticise(task, tasks, texts, critic)
    bad = np.array([bool(rules(task, x, t["language"], allowed(task, t))) for t, x in zip(tasks, texts)])
    passed = np.array([ok for ok, _ in verdicts])
    return {"texts": len(texts), "rules fail": int(bad.sum()), "missed": int((passed & bad).sum()),
            "false alarm": int((~passed & ~bad).sum()), "agree": float(np.mean(passed == ~bad))}, recs


def cost(records):
    """Requests, tokens and processor seconds of a list of answer records."""
    return {"requests": len(records), "in": sum(r["prompt_tokens"] for r in records),
            "out": sum(r["completion_tokens"] for r in records), "seconds": sum(r["seconds"] for r in records)}


def subset(tasks, per_language):
    """The first per_language tasks of each language, in the tasks' order."""
    seen, out = {}, []
    for k, t in enumerate(tasks):
        if seen.get(t["language"], 0) < per_language:
            seen[t["language"]] = seen.get(t["language"], 0) + 1
            out.append(k)
    return out


if __name__ == "__main__":
    from content_data import load
    t0 = time.time()
    W = load()
    all_tasks = (("reasons", reason_tasks(W)), ("headlines", headline_tasks(W)))
    del W
    for task, tasks in all_tasks:
        print(f"\n{task}: {len(tasks)} texts, {len(tasks) // len(LANGUAGES)} per language. The rules' verdict: share "
              f"that pass every rule, and the share with each problem  [{time.time() - t0:.0f} s]")
        print(f"  {'':<40}{'pass':>7}{'length':>8}{'script':>8}{'unfaithful':>12}{'claim':>7}")
        rows = {"template": [template(task, t) for t in tasks]}
        first, recs = write(task, tasks)
        print(f"  {GENERATOR}'s first attempts: {llm.failures(recs, [TEXT_SCHEMA[task]] * len(recs))}")
        rows[f"{GENERATOR}, one attempt"] = first
        loops = {}
        for evaluator, critic in (("rules", None), ("critic", CRITIC_MODEL), ("both", CRITIC_MODEL)):
            label = f"loop, {evaluator}" + (f" ({critic})" if critic else "")
            loops[label] = loop(task, tasks, evaluator, critic)
            rows[label] = loops[label][0]
        for name, texts in rows.items():
            s = rule_summary(task, tasks, texts)
            print(f"  {name:<40}{s['pass']:>7.1%}{s['length']:>8.1%}{s['script']:>8.1%}{s['unfaithful']:>12.1%}"
                  f"{s['claim']:>7.1%}")
        print(f"  by language, share passing every rule:")
        print(f"  {'':<40}" + "".join(f"{LANGUAGE_NAMES[g]:>12}" for g in range(len(LANGUAGES))))
        for name, texts in rows.items():
            cells = []
            for g in range(len(LANGUAGES)):
                ks = [k for k, t in enumerate(tasks) if t["language"] == g]
                cells.append(np.mean([not rules(task, texts[k], g, allowed(task, tasks[k])) for k in ks]))
            print(f"  {name:<40}" + "".join(f"{c:>12.1%}" for c in cells))
        print(f"  the loops: accepted by their evaluator, by round; requests per 1,000 texts; processor seconds per "
              f"text on two cores")
        for label, (texts, accepted, made) in loops.items():
            by = [np.mean([a == r for a in accepted]) for r in range(1, MAX_ROUNDS + 1)]
            c = cost(made)
            print(f"  {label:<40} round 1 {by[0]:.1%}, round 2 {by[1]:.1%}, round 3 {by[2]:.1%}, never "
                  f"{np.mean([a is None for a in accepted]):.1%}; {1000 * c['requests'] / len(tasks):,.0f} requests, "
                  f"{c['seconds'] / len(tasks):.2f} s")
        for k in subset(tasks, 1):
            fixed = [x for label, (x, _, _) in loops.items() if label.startswith("loop, rules")][0][k]
            print(f"    e.g. ({LANGUAGE_NAMES[tasks[k]['language']]}) one attempt {first[k]!r} "
                  f"{rules(task, first[k], tasks[k]['language'], allowed(task, tasks[k]))}; after the rules loop "
                  f"{fixed!r}")
        print(f"  the critics against the rules, on the one-attempt texts")
        critics = [(CRITIC_MODEL, list(range(len(tasks))))]
        if task == "headlines":                  # item data only: the hosted model never sees a user's history
            critics.append(("Claude Sonnet 5.5", list(range(len(tasks)))))
        for critic, ks in critics:
            s, crecs = critic_check(task, [tasks[k] for k in ks], [first[k] for k in ks], critic)
            c = cost(crecs)
            print(f"  {critic:<22} {s['texts']:>4} texts, the rules fail {s['rules fail']:>3}; the critic missed "
                  f"{s['missed']:>3} of them and raised {s['false alarm']:>3} false alarms; agreement {s['agree']:.1%}; "
                  f"{c['seconds'] / max(len(ks), 1):.2f} s per text"
                  + (f", USD {llm.usage(crecs, critic)['usd'] * 1000 / len(ks):.2f} per 1,000"
                     if critic.startswith("Claude") else ""))
    print(f"\n[{time.time() - t0:.0f} s]")
