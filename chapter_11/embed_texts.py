"""Embed every Chapter 11 text with three pretrained text encoders (section 11.3).

Reads texts_to_embed.tsv (written by content_data.py) and writes text_embeddings.npz,
with one array of unit-length vectors per model, in the file's order:

  all-MiniLM-L6-v2                       - a small English-only sentence encoder (2021)
  paraphrase-multilingual-MiniLM-L12-v2  - a multilingual encoder distilled from an
                                           English one with parallel data (2020)
  Qwen3-Embedding-0.6B                   - a multilingual encoder built on the Qwen3
                                           language model (2025)

The models are downloaded from Hugging Face the first time (about 1.5 GB in all), each
at the exact revision in REVISIONS, and the file records those revisions, the encoding
options and the package versions; the other scripts check the revisions when they load
it. Item titles are embedded as documents, with no instruction, so title-to-title
similarity is symmetric. The book's vectors were made with Python 3.13.13,
sentence-transformers 5.7.0, transformers 5.22.2, torch 2.15.0 and NumPy 2.5.4.

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

HERE = Path(__file__).resolve().parent
MODELS = {"all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
          "paraphrase-multilingual-MiniLM-L12-v2": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
          "Qwen3-Embedding-0.6B": "Qwen/Qwen3-Embedding-0.6B"}
# The exact model revisions (Hugging Face commits) used for the book's vectors
REVISIONS = {"all-MiniLM-L6-v2": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
             "paraphrase-multilingual-MiniLM-L12-v2": "e8f8c211226b894fcb81acc59f3b34ba3efd5f42",
             "Qwen3-Embedding-0.6B": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"}
OPTIONS = "normalize_embeddings=True, batch_size=64, stored as float16"

if __name__ == "__main__":
    import sentence_transformers
    import torch
    import transformers
    from sentence_transformers import SentenceTransformer
    lines = (HERE / "texts_to_embed.tsv").read_text(encoding="utf-8").split("\n")
    digest, texts = lines[0][2:], [t for t in lines[1:] if t]
    out = {"digest": np.array(digest), "options": np.array(OPTIONS),
           "versions": np.array(f"Python {sys.version.split()[0]}, sentence-transformers "
                                f"{sentence_transformers.__version__}, transformers {transformers.__version__}, "
                                f"torch {torch.__version__}, NumPy {np.__version__}")}
    for name, repo in MODELS.items():
        t0 = time.time()
        model = SentenceTransformer(repo, revision=REVISIONS[name])
        E = model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True)
        out[name] = E.astype(np.float16)
        out[name + " revision"] = np.array(REVISIONS[name])
        print(f"{name}: {E.shape[0]:,} texts x {E.shape[1]} numbers  [{time.time() - t0:.0f} s]", flush=True)
    np.savez_compressed(HERE / "text_embeddings.npz", **out)
    print(f"Wrote {HERE / 'text_embeddings.npz'} (digest {digest}); {out['versions']}")
