"""Tesselo's LLM toolkit for Part 4 (section 17.2; Listing 17.1 is _Local.run).

One small module runs every language model in Chapters 17 to 23, so that every answer
printed in the book can be traced to the exact model, prompt and settings that made it.

  MODELS            - the models, each with the exact file (and its SHA-256) or API name
  ask(model, task, system, user, schema)
                    - one request: the answer as a Python object that matches the JSON
                      schema, with its token counts and time. Every answer is stored in
                      llm_cache/<task>.jsonl under a key made from the model, the file's
                      checksum, the messages, the schema and the settings, so a second
                      run reads the answer instead of computing it
  usage(task)       - the tokens, seconds and cost of every cached answer of a task

Open-weight models run on Tesselo's own servers with llama.cpp (llama-cpp-python),
on the CPU, with greedy decoding: the most likely token at every step, so the same
file, prompt and build give the same answer. Their output is constrained by the JSON
schema: llama.cpp turns the schema into a grammar and allows only tokens that keep the
answer valid, so an answer cannot name a category or a concept that the schema does
not list. The hosted reference model (Claude Sonnet 5.5) is reached through
Anthropic's API, only for data that describes items, never users; its key is read from
the ANTHROPIC_API_KEY environment variable and is never stored.

Readers reproduce every number from the cache without running a model (replay). To run
the models, put the GGUF files from fetch_models.py in chapter_17/models; each file is
checked against its SHA-256 before it is loaded. CH17_RUN=1 computes the answers that are
missing, and also those whose cached answer came from another engine or build (the
engine is stored with every answer); CH17_FRESH=1 with CH17_RUN=1 computes every answer
again, best with CH17_CACHE pointing to another folder so that the archive stays as it
is. ANTHROPIC_API_KEY must be set for the hosted model. Every answer is checked against
its schema (valid), because a grammar binds only an answer that finishes, and the
hosted model does not enforce counts or lengths.
"""
import functools
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL_DIR = Path(os.environ.get("CH17_MODELS", HERE / "models"))
CACHE_DIR = Path(os.environ.get("CH17_CACHE", HERE / "llm_cache"))
THREADS = 2
CONTEXT = 8_192
SETTINGS = {"temperature": 0.0, "seed": 0, "top_k": 1}

# name: (engine, file or API model, size in billions of parameters, price in USD per million
# input and output tokens; open-weight models have no per-token price)
MODELS = {
    "Qwen3.5-0.8B": ("llama.cpp", "Qwen3.5-0.8B-Q4_K_M.gguf", 0.8, None),
    "Qwen3.5-2B": ("llama.cpp", "Qwen3.5-2B-Q4_K_M.gguf", 2.0, None),
    "Qwen3.5-4B": ("llama.cpp", "Qwen3.5-4B-Q4_K_M.gguf", 4.0, None),
    "Qwen3.5-9B": ("llama.cpp", "Qwen3.5-9B-Q4_K_M.gguf", 9.0, None),
    "Claude Sonnet 5.5": ("api", "claude-sonnet-5-5", None, (2.00, 10.00, 2.50, 0.10)),
}
# A hosted model's prices: input, output, cache write (5 minutes) and cache read, in USD per
# million tokens (Anthropic's list prices, November 2026).
# The exact files: unsloth's 4-bit GGUF conversions on Hugging Face, at these commits
FILES = {
    "Qwen3.5-0.8B": ("unsloth/Qwen3.5-0.8B-GGUF", "6ab461498e2023f6e3c1baea90a8f0fe38ab64d0",
                     "bd258782e35f7f458f8aced1adc053e6e92e89bc735ba3be89d38a06121dc517"),
    "Qwen3.5-2B": ("unsloth/Qwen3.5-2B-GGUF", "f6d5376be1edb4d416d56da11e5397a961aca8ae",
                   "aaf42c8b7c3cab2bf3d69c355048d4a0ee9973d48f16c731c0520ee914699223"),
    "Qwen3.5-4B": ("unsloth/Qwen3.5-4B-GGUF", "e87f176479d0855a907a41277aca2f8ee7a09523",
                   "00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4"),
    "Qwen3.5-9B": ("unsloth/Qwen3.5-9B-GGUF", "3885219b6810b007914f3a7950a8d1b469d598a5",
                   "03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8"),
}
_loaded = {}


def model_id(name):
    """The model's name with what pins it: the file's SHA-256 for an open-weight model,
    the API model name for a hosted one."""
    engine, ref, _, _ = MODELS[name]
    return f"{name} ({ref})" if engine == "api" else f"{name} ({ref}, sha256 {FILES[name][2][:12]})"


class _Local:
    """An open-weight model in llama.cpp. The prompt is the model's own chat template
    applied to a system message and a user message. The system part is evaluated once
    and its state saved; every request restores that state and evaluates only its own
    user part, so a long shared instruction costs its tokens once, and every answer is
    computed in exactly the same way whatever came before it."""

    def __init__(self, name):
        import llama_cpp
        import llama_cpp.llama
        from llama_cpp import Llama
        from llama_cpp.llama_chat_format import Jinja2ChatFormatter
        path = MODEL_DIR / MODELS[name][1]
        if not path.exists():
            raise SystemExit(f"{path} is missing: run fetch_models.py (see llm.py)")
        if _sha256(path) != FILES[name][2]:                      # the exact file, or nothing
            raise SystemExit(f"{path} is not the file llm.FILES pins (SHA-256 differs): run fetch_models.py")
        default = llama_cpp.llama_model_default_params

        def params():                                            # weights read in place from the file, not
            p = default()                                        # repacked into a second copy in memory
            p.use_extra_bufts = False
            return p
        llama_cpp.llama.llama_cpp.llama_model_default_params = params
        self.m = Llama(model_path=str(path), n_ctx=CONTEXT, n_threads=THREADS, n_threads_batch=THREADS,
                       n_batch=512, seed=SETTINGS["seed"], verbose=False)
        self.fmt = Jinja2ChatFormatter(template=self.m.metadata["tokenizer.chat_template"],
                                       eos_token="<|im_end|>", bos_token="")
        self.states, self.grammars = {}, {}

    def tokens(self, system, user):
        text = self.fmt(messages=[{"role": "system", "content": system}, {"role": "user", "content": user}]).prompt
        cut = text.index(user, text.index(system) + len(system))
        prefix, rest = text[:cut], text[cut:]
        tok = lambda s: self.m.tokenize(s.encode("utf-8"), add_bos=False, special=True)   # noqa: E731
        return prefix, tok(prefix), tok(rest)

    def run(self, system, user, schema, max_tokens):
        """Listing 17.1. Restore the state after the system part, evaluate the user part,
        and decode greedily with the schema's grammar."""
        from llama_cpp import LlamaGrammar
        prefix, pre, rest = self.tokens(system, user)
        if prefix not in self.states:
            if len(self.states) >= 4:
                self.states.clear()
            self.m.reset()
            self.m.eval(pre)
            self.states[prefix] = self.m.save_state()
        g = json.dumps(schema, sort_keys=True)
        if g not in self.grammars:
            self.grammars[g] = LlamaGrammar.from_json_schema(g, verbose=False)
        self.m.load_state(self.states[prefix])
        out = self.m.create_completion(prompt=pre + rest, max_tokens=max_tokens, temperature=SETTINGS["temperature"],
                                       top_k=SETTINGS["top_k"], seed=SETTINGS["seed"], grammar=self.grammars[g])
        return out["choices"][0]["text"], (out["usage"]["prompt_tokens"], out["usage"]["completion_tokens"])


def _strict(schema):
    """The schema as the API's structured outputs accept it: no properties beyond those
    listed, and no limits on lengths or counts (the caller applies those to the answer)."""
    if isinstance(schema, dict):
        out = {k: _strict(v) for k, v in schema.items() if k not in ("maxItems", "minItems", "maxLength")}
        if out.get("type") == "object":
            out["additionalProperties"] = False
        return out
    return [_strict(v) for v in schema] if isinstance(schema, list) else schema


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def _local(name):
    if name not in _loaded:
        _loaded.clear()                                          # one model in memory at a time
        _loaded[name] = _Local(name)
    return _loaded[name]


def _key(name, system, user, schema, max_tokens):
    blob = json.dumps([model_id(name), system, user, schema, SETTINGS, max_tokens], ensure_ascii=False,
                      sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _cache(task):
    path = CACHE_DIR / f"{task}.jsonl"
    out = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                out[r["key"]] = r
    return out


_caches = {}


def _task_cache(task):
    """The task's cached answers, read from disk once per process. The cache is meant for
    one writing process at a time (the chapter's runs); several writers would need a
    shared store."""
    if task not in _caches:
        _caches[task] = _cache(task)
    return _caches[task]


def cached(name, task, system, user, schema, max_tokens=256):
    """The cached answer to this request, or None."""
    return _task_cache(task).get(_key(name, system, user, schema, max_tokens))


def _usable(rec, name):
    """Whether a cached answer may be returned. Replaying (CH17_RUN unset) returns the
    archived answer. With CH17_RUN=1, an answer from a local model is reused only if it was
    computed by the same engine and build as now; with CH17_FRESH=1, never (set CH17_CACHE
    to another folder to keep fresh answers apart from the archive)."""
    if rec is None or os.environ.get("CH17_FRESH") == "1":
        return False
    if os.environ.get("CH17_RUN") == "1" and MODELS[name][0] == "llama.cpp":
        return rec.get("engine") == engine_version(name)
    return True


def valid(answer, schema):
    """Whether a parsed answer matches the request's schema: types, enums, required keys,
    and the limits on counts and lengths. An answer cut off at max_tokens is None. The
    hosted model's structured output does not enforce counts or lengths (see _strict), and
    a grammar applies only to an answer that finishes, so every answer is checked here."""
    if answer is None:
        return False
    t = schema.get("type")
    if t == "object":
        if not isinstance(answer, dict) or any(k not in answer for k in schema.get("required", [])):
            return False
        return all(valid(answer[k], sub) for k, sub in schema.get("properties", {}).items() if k in answer)
    if t == "array":
        if not isinstance(answer, list) or len(answer) > schema.get("maxItems", len(answer)) \
                or len(answer) < schema.get("minItems", 0):
            return False
        return all(valid(x, schema["items"]) for x in answer) if "items" in schema else True
    if t == "string":
        return isinstance(answer, str) and len(answer) <= schema.get("maxLength", len(answer)) \
            and ("enum" not in schema or answer in schema["enum"])
    if t == "boolean":
        return isinstance(answer, bool)
    return True


def failures(records, schemas):
    """How many answers were cut off (no complete JSON) and how many do not match their
    schema, in words for the scripts' reports."""
    cut = sum(r["answer"] is None for r in records)
    bad = sum(r["answer"] is not None and not valid(r["answer"], sc) for r, sc in zip(records, schemas))
    return f"{cut} answers cut off, {bad} not matching the schema"


def ask(name, task, system, user, schema, max_tokens=256):
    """One request to a model: a system and a user message in, an object
    that matches the JSON schema out, read from the cache when the same request was
    answered before. Returns the cached record: the answer, the tokens and the seconds."""
    key = _key(name, system, user, schema, max_tokens)
    cache = _task_cache(task)
    if _usable(cache.get(key), name):
        return cache[key]
    if os.environ.get("CH17_RUN") != "1":
        raise SystemExit(f"{task}: an answer is not in llm_cache; set CH17_RUN=1 to run {name} (see llm.py)")
    t0 = time.time()
    engine, ref, _, _ = MODELS[name]
    cached_tokens = (0, 0)
    if engine == "llama.cpp":
        text, tokens = _local(name).run(system, user, schema, max_tokens)
    else:
        import anthropic
        client = anthropic.Anthropic(base_url="https://api.anthropic.com")    # the key: ANTHROPIC_API_KEY
        msg = client.messages.create(                            # the long system part is cached by the API
            model=ref, max_tokens=max_tokens + 1_024, system=[
                {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": _strict(schema)}})
        text = next(b.text for b in msg.content if b.type == "text")
        tokens = (msg.usage.input_tokens, msg.usage.output_tokens)
        cached_tokens = (msg.usage.cache_creation_input_tokens or 0, msg.usage.cache_read_input_tokens or 0)
    try:
        answer = json.loads(text)
    except json.JSONDecodeError:                                 # cut off at max_tokens
        answer = None
    rec = {"key": key, "model": model_id(name), "user": user, "answer": answer, "text": text,
           "prompt_tokens": tokens[0], "completion_tokens": tokens[1], "cache_write_tokens": cached_tokens[0],
           "cache_read_tokens": cached_tokens[1], "seconds": round(time.time() - t0, 3),
           "engine": engine_version(name)}
    CACHE_DIR.mkdir(exist_ok=True)
    with open(CACHE_DIR / f"{task}.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    cache[key] = rec
    return rec


def ask_many(name, task, requests, max_tokens=256, every=50):
    """Many requests to one model: [(system, user, schema), ...] in, the cached records out,
    in the same order. Requests not yet in the cache are answered in a separate process
    that loads only the model, so that the model and the caller's data never share memory
    (two processor cores and 7 GB on the book's machine)."""
    import subprocess
    import tempfile
    cache = _task_cache(task)
    keys = [_key(name, s, u, sc, max_tokens) for s, u, sc in requests]
    todo = [r for r, k in zip(requests, keys) if not _usable(cache.get(k), name)]
    if todo and os.environ.get("CH17_COLLECT"):                 # staged: hand the requests to staged.py
        with open(os.environ["CH17_COLLECT"], "w", encoding="utf-8") as f:
            json.dump({"name": name, "task": task, "max_tokens": max_tokens, "requests": todo}, f, ensure_ascii=False)
        raise SystemExit(f"collected {len(todo)} requests for {name} ({task})")
    if todo:
        if os.environ.get("CH17_RUN") != "1":
            raise SystemExit(f"{task}: {len(todo)} answers are not in llm_cache; set CH17_RUN=1 to run {name}")
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
            for s, u, sc in todo:
                f.write(json.dumps([s, u, sc], ensure_ascii=False) + "\n")
        subprocess.run([sys.executable, __file__, "worker", name, task, f.name, str(max_tokens), str(every)],
                       check=True)
        os.unlink(f.name)
        cache = _caches[task] = _cache(task)
    return [cache[k] for k in keys]


@functools.lru_cache(maxsize=None)
def engine_version(name):
    if MODELS[name][0] == "api":
        import anthropic
        return f"anthropic {anthropic.__version__}"
    import llama_cpp
    return f"llama-cpp-python {llama_cpp.__version__}, {THREADS} threads, {platform.machine()}"


def usage(records, name):
    """Totals over a list of ask() results: requests, input and output tokens, seconds and,
    for a hosted model, the cost in USD at its list price (cached input billed at the
    cache prices)."""
    n = len(records)
    pin = sum(r["prompt_tokens"] for r in records)
    pout = sum(r["completion_tokens"] for r in records)
    cw = sum(r.get("cache_write_tokens", 0) for r in records)
    cr = sum(r.get("cache_read_tokens", 0) for r in records)
    sec = sum(r["seconds"] for r in records)
    price = MODELS[name][3]
    cost = (pin * price[0] + pout * price[1] + cw * price[2] + cr * price[3]) / 1e6 if price else None
    return {"requests": n, "input": pin + cw + cr, "output": pout, "seconds": sec, "usd": cost}


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "worker":
    _, _, name, task, path, max_tokens, every = sys.argv
    t0 = time.time()
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    for k, line in enumerate(lines):
        system, user, sc = json.loads(line)
        ask(name, task, system, user, sc, int(max_tokens))
        if (k + 1) % int(every) == 0 or k + 1 == len(lines):
            print(f"    {name} {task}: {k + 1:,}/{len(lines):,}  [{time.time() - t0:.0f} s]", flush=True)
