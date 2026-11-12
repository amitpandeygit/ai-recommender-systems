"""Embed every Chapter 11 text with three pretrained text encoders (section 11.3).

Reads texts_to_embed.tsv (written by content_data.py) and writes text_embeddings.npz,
with one array of unit-length vectors per model, in the file's order:

  all-MiniLM-L6-v2                       - a small English-only sentence encoder (2021)
  paraphrase-multilingual-MiniLM-L12-v2  - a multilingual encoder distilled from an
                                           English one with parallel data (2020)
  Qwen3-Embedding-0.6B                   - a multilingual encoder built on the Qwen3
                                           language model (2025)

The models are downloaded from Hugging Face the first time (about 1.5 GB in all). Item
titles are embedded as documents, with no instruction, so title-to-title similarity is
symmetric.

Needs Python 3.10 or later and sentence-transformers 5 (it installs PyTorch). On a
laptop, from the chapter_11 folder (about 10 to 30 minutes, most of it Qwen3):
    python -m venv .venv
    .venv/bin/pip install "sentence-transformers>=5,<6"
    .venv/bin/python embed_texts.py
"""
import sys
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

HERE = Path(__file__).resolve().parent
MODELS = {"all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
          "paraphrase-multilingual-MiniLM-L12-v2": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
          "Qwen3-Embedding-0.6B": "Qwen/Qwen3-Embedding-0.6B"}

if __name__ == "__main__":
    lines = (HERE / "texts_to_embed.tsv").read_text(encoding="utf-8").split("\n")
    digest, texts = lines[0][2:], [t for t in lines[1:] if t]
    out = {"digest": np.array(digest)}
    for name, repo in MODELS.items():
        t0 = time.time()
        model = SentenceTransformer(repo)
        E = model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True)
        out[name] = E.astype(np.float16)
        print(f"{name}: {E.shape[0]:,} texts x {E.shape[1]} numbers  [{time.time() - t0:.0f} s]", flush=True)
    np.savez_compressed(HERE / "text_embeddings.npz", **out)
    print(f"Wrote {HERE / 'text_embeddings.npz'} (digest {digest}); Python {sys.version.split()[0]}")
