"""Run a Chapter 17 script whose model requests must not share memory with its data.

The scripts load Chapter 2's world (about 3 GB) and call language models (up to 6 GB for
the 9B), which do not fit together in the book's 7 GB workspace. staged.py runs the script
with CH17_COLLECT set: at the first request not yet in llm_cache, the script writes the
missing requests to a file and stops. staged.py then answers them with llm.py's worker,
which loads only the model, and runs the script again, until it finishes. Every answer
lands in llm_cache, so a reader with the cache never needs this.

    CH17_RUN=1 python chapter_17/staged.py chapter_17/cosmo_kg.py
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

if __name__ == "__main__":
    script, args = sys.argv[1], sys.argv[2:]
    path = Path(tempfile.gettempdir()) / "ch17_collect.json"
    for stage in range(1, 100):
        path.unlink(missing_ok=True)
        env = dict(os.environ, CH17_COLLECT=str(path))
        done = subprocess.run([sys.executable, script, *args], env=env)
        if not path.exists():
            sys.exit(done.returncode)
        job = json.loads(path.read_text(encoding="utf-8"))
        if "embed" in job:
            print(f"[staged] stage {stage}: {len(job['embed'])} texts to embed", flush=True)
            subprocess.run([sys.executable, str(HERE / "textvec.py"), str(path)], check=True)
            continue
        print(f"[staged] stage {stage}: {len(job['requests'])} requests for {job['name']} ({job['task']})", flush=True)
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
            for r in job["requests"]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        subprocess.run([sys.executable, str(HERE / "llm.py"), "worker", job["name"], job["task"], f.name,
                        str(job["max_tokens"]), "50"], check=True, env=dict(os.environ, CH17_RUN="1"))
        os.unlink(f.name)
