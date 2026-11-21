"""Qwen3-Embedding vectors for Chapter 17's new texts (sections 17.3 and 17.4).

Chapter 11 embedded every clean title with Qwen3-Embedding-0.6B (text_embeddings.npz).
Chapter 17 has new texts: listings as written, and records rendered from a cleaner's
concepts. vectors(texts) returns their unit vectors from text_vectors.npz, and embeds
any text not yet in it with the same model, at the revision Chapter 11 pinned, with the
same options (normalised, batch size 64, stored as float16). The file is in the
repository, so readers need the model only to embed texts of their own.
"""
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "chapter_11"))
from embed_texts import REVISIONS  # noqa: E402

FILE = HERE / "text_vectors.npz"
MODEL_DIR = Path(os.environ.get("CH17_MODELS", HERE / "models")) / "Qwen3-Embedding-0.6B"
NAME = "Qwen3-Embedding-0.6B"


def _load():
    if not FILE.exists():
        return [], np.zeros((0, 1024), dtype=np.float16)
    z = np.load(FILE, allow_pickle=False)
    assert str(z["revision"]) == REVISIONS[NAME], "text_vectors.npz was made with another revision"
    return list(z["texts"]), z["vectors"]


def vectors(texts):
    """Unit vectors (float32) for the texts, in order."""
    known, V = _load()
    index = {t: k for k, t in enumerate(known)}
    todo = sorted({t for t in texts if t not in index})
    if todo and os.environ.get("CH17_COLLECT"):                 # staged: embed in a process of its own
        import json
        with open(os.environ["CH17_COLLECT"], "w", encoding="utf-8") as f:
            json.dump({"embed": todo}, f, ensure_ascii=False)
        raise SystemExit(f"collected {len(todo)} texts to embed")
    if todo:
        import sentence_transformers
        import torch
        import transformers
        from sentence_transformers import SentenceTransformer
        torch.set_num_threads(2)
        model = SentenceTransformer(str(MODEL_DIR), device="cpu")
        E = model.encode(todo, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
        V = np.vstack([V, E.astype(np.float16)])
        for t in todo:
            index[t] = len(index)
        known = known + todo
        np.savez_compressed(FILE, texts=np.array(known), vectors=V, revision=np.array(REVISIONS[NAME]),
                            versions=np.array(f"sentence-transformers {sentence_transformers.__version__}, "
                                              f"transformers {transformers.__version__}, torch {torch.__version__}, "
                                              f"NumPy {np.__version__}"))
    X = V[[index[t] for t in texts]].astype(np.float32)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


if __name__ == "__main__":                                       # python textvec.py texts.json
    import json
    vectors(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["embed"])
