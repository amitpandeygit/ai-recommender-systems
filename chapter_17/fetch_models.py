"""Download the open-weight models of Chapter 17 (section 17.2), once.

  Qwen3.5 small models, 0.8B, 2B, 4B and 9B, as unsloth's 4-bit GGUF files (Q4_K_M) for
  llama.cpp, at the commits pinned in llm.FILES
  Qwen3-Embedding-0.6B, at the revision Chapter 11 used, to embed new texts

Each GGUF file is saved in chapter_17/models and checked against the SHA-256 in
llm.FILES, so llm.py runs exactly the files the book used. Readers who only want to
reproduce the numbers do not need this: every answer is in llm_cache.

Needs huggingface_hub. From the repository root (about 10 GB):
    pip install "huggingface_hub>=0.30"
    python chapter_17/fetch_models.py
"""
import hashlib
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from llm import FILES, MODEL_DIR, MODELS  # noqa: E402

EMBEDDER = ("Qwen/Qwen3-Embedding-0.6B", "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


if __name__ == "__main__":
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    bad = 0
    for name, (repo, commit, digest) in FILES.items():
        file = MODELS[name][1]
        print(f"{name}: {repo} {file} at commit {commit[:12]} ...", flush=True)
        path = Path(hf_hub_download(repo, file, revision=commit, local_dir=MODEL_DIR))
        ok = sha256(path) == digest
        bad += not ok
        print(f"   {path.stat().st_size / 1e9:.2f} GB, checksum {'matches' if ok else 'DOES NOT MATCH'}", flush=True)
    repo, rev = EMBEDDER
    print(f"{repo} at commit {rev[:12]} ...", flush=True)
    snapshot_download(repo, revision=rev, local_dir=MODEL_DIR / "Qwen3-Embedding-0.6B")
    sys.exit(1 if bad else 0)
