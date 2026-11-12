"""Shared content data for Chapter 11: titles in six languages, simulated pictures, video
and sound, and simulated vision-language tags (sections 11.2 to 11.5).

Every test in this chapter runs in Chapter 2's world, with Chapter 9's test contract,
exactly as in Chapter 10 (chapter_10/graph_data.py). This module adds what an item
looks and sounds like.

  titles(W, language=None)   - every item's title written in its own language (or in one
                               given language), from Chapter 3's tokens (title_words.py)
  vlm_tags(W, quality)       - the words a vision-language model would write for each item
                               after looking at it, at quality "good" or "weak" (section 11.5)
  modalities(W)              - simulated outputs of pretrained image, video and audio
                               encoders, and which items have each (section 11.4)
  embedding_texts(W)         - every text the text encoders must embed, numbered once
  text_embeddings(W, model)  - the embeddings that embed_texts.py computed, by item

Run from the repository root to write texts_to_embed.tsv for embed_texts.py:
    python chapter_11/content_data.py
"""
import os
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "2"
import hashlib
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "chapter_10"))
sys.path.insert(0, str(HERE.parent / "chapter_03"))
from graph_data import load  # noqa: E402,F401
from tesselo_shop import CATEGORIES, GENERIC_WORDS, LANGUAGES, VIDEO_CATEGORIES, WORDS_PER_CATEGORY  # noqa: E402
from title_words import WORDS, render  # noqa: E402

TEXTS_FILE = HERE / "texts_to_embed.tsv"
EMBEDDINGS_FILE = Path(os.environ.get("CH11_EMBEDDINGS", HERE / "text_embeddings.npz"))
MODELS = ("all-MiniLM-L6-v2", "paraphrase-multilingual-MiniLM-L12-v2", "Qwen3-Embedding-0.6B")

# Simulated pretrained encoders (section 11.4). Each sees part of an item's hidden taste
# (a random subspace of `signal` of its 16 directions) mixed with things that do not
# matter for taste: the creator's style, and the item's own lighting, framing or
# recording. Products have a picture only; 10% of videos have no usable sound.
MODALITY_SETTINGS = {"image": {"signal": 8, "dim": 64},
                     "video": {"signal": 10, "dim": 64},
                     "audio": {"signal": 6, "dim": 32}}
NO_AUDIO = 0.10
VLM_QUALITY = {"good": (0.95, 0.80), "weak": (0.70, 0.40)}   # (category right, each word right)


def titles(W, language=None):
    """Every item's title as text: in the item's own language, or in `language` (0 to 5)."""
    items = W["items"]
    return [render(t, items["language"][i] if language is None else language) for i, t in enumerate(items["title"])]


def category_words(c):
    return [f"{CATEGORIES[c].replace(' ', '_')}_{k}" for k in range(WORDS_PER_CATEGORY)]


def vlm_tags(W, quality):
    """The 3 topic words a vision-language model writes for each item after looking at it
    (no generic word). With probability p_cat it recognises the item's category; then
    each of the 3 category words of the real title is kept with probability p_word and
    otherwise replaced by another word of that category. A wrong category gives 3 words
    of the wrong category (of the same kind: video or product)."""
    p_cat, p_word = VLM_QUALITY[quality]
    items = W["items"]
    rng = np.random.default_rng(W["cfg"].seed + 1110 + list(VLM_QUALITY).index(quality))
    n_vid = len(VIDEO_CATEGORIES)
    out = []
    for i, t in enumerate(items["title"]):
        c = int(items["category"][i])
        own = category_words(c)
        if rng.random() < p_cat:
            words = [str(w) for w in t if str(w) in own][:3]
            for k, w in enumerate(words):
                if rng.random() >= p_word:
                    words[k] = str(rng.choice([x for x in own if x not in words]))
        else:
            kind = range(n_vid) if c < n_vid else range(n_vid, len(CATEGORIES))
            wrong = int(rng.choice([x for x in kind if x != c]))
            words = [str(w) for w in rng.choice(category_words(wrong), 3, replace=False)]
        out.append(words)
    return out


def modalities(W):
    """Listing 11.1. Simulated outputs of pretrained image, video and audio encoders, unit
    length, with a mask of the items that have each. Returns {name: (vectors, has)}."""
    cfg, items = W["cfg"], W["items"]
    v, n = items["vec"], len(items["vec"])
    video = ~items["is_product"]
    out = {}
    for k, (name, s) in enumerate(MODALITY_SETTINGS.items()):
        rng = np.random.default_rng(cfg.seed + 1100 + k)
        basis = np.linalg.qr(rng.normal(size=(v.shape[1], s["signal"])))[0]     # what this encoder can see
        style = rng.normal(0.0, 0.5, size=(cfg.num_creators, 8))[items["creator"]]   # creator's style
        own = rng.normal(0.0, 0.5, size=(n, 8))                                  # lighting, framing, recording
        x = np.hstack([v @ basis, style, own])
        mix = rng.normal(size=(x.shape[1], s["dim"])) / np.sqrt(x.shape[1])
        e = x @ mix + rng.normal(0.0, 0.1, size=(n, s["dim"]))
        e /= np.linalg.norm(e, axis=1, keepdims=True)
        has = np.ones(n, dtype=bool) if name == "image" else video.copy()
        if name == "audio":
            has &= rng.random(n) >= NO_AUDIO
        e[~has] = 0.0
        out[name] = (e.astype(np.float32), has)
    return out


def embedding_texts(W):
    """Every text the encoders must embed, each once. Returns the list of texts and the
    index of each: titles[item, language], tags[quality][item], generic[word, language]
    and words[token, language]."""
    items = W["items"]
    texts, index = [], {}

    def num(t):
        if t not in index:
            index[t] = len(texts)
            texts.append(t)
        return index[t]
    n = len(items["vec"])
    title = np.array([[num(render(items["title"][i], lang)) for lang in range(len(LANGUAGES))] for i in range(n)])
    tags = {q: np.array([num(render(t, items["language"][i])) for i, t in enumerate(vlm_tags(W, q))])
            for q in VLM_QUALITY}
    generic = np.array([[num(render([w], lang)) for lang in range(len(LANGUAGES))] for w in GENERIC_WORDS])
    tokens = list(WORDS)
    words = np.array([[num(render([w], lang)) for lang in range(len(LANGUAGES))] for w in tokens])
    return texts, {"titles": title, "tags": tags, "generic": generic, "words": words, "tokens": tokens}


def texts_digest(texts):
    return hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest()[:16]


def text_embeddings(W, model):
    """The embeddings of every text for one model (rows in embedding_texts order), and the
    index from embedding_texts. Fails clearly if embed_texts.py has not been run."""
    texts, idx = embedding_texts(W)
    if not EMBEDDINGS_FILE.exists():
        raise SystemExit(f"{EMBEDDINGS_FILE} is missing: run chapter_11/embed_texts.py first (section 11.3)")
    z = np.load(EMBEDDINGS_FILE)
    if str(z["digest"]) != texts_digest(texts):
        raise SystemExit("text_embeddings.npz was computed from different texts: run embed_texts.py again")
    E = z[model].astype(np.float32)
    E /= np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
    return E, idx


if __name__ == "__main__":
    W = load()
    texts, idx = embedding_texts(W)
    with open(TEXTS_FILE, "w", encoding="utf-8") as f:
        f.write(f"# {texts_digest(texts)}\n")
        for t in texts:
            f.write(t.replace("\t", " ") + "\n")
    items = W["items"]
    print(f"{len(texts):,} distinct texts written to {TEXTS_FILE.name} (digest {texts_digest(texts)})")
    for i in range(4):
        print(f"  item {i} ({LANGUAGES[items['language'][i]]}): {texts[idx['titles'][i, items['language'][i]]]}"
              f"  |  English: {texts[idx['titles'][i, 0]]}")
