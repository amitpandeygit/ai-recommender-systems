"""Fill llm_cache for section 17.3's cleaning requests, one model at a time (used to run
the long jobs; the chapter's scripts then read the cache). It reads listings.json and
listings_heldout.json, so it does not load the world, and the model has the machine's
memory to itself.
    CH17_RUN=1 python chapter_17/fill_cache.py <clean|hybrid> <model> <size|small|all>
    CH17_RUN=1 python chapter_17/fill_cache.py <hybrid_rom|heldout_rom> <model>
hybrid_rom: the romanised dictionary with the model behind it, on all listings;
heldout_rom: the same on the held-out spelling test."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from enrich import NINE_B_LANGUAGES, NINE_B_PER_LANGUAGE, SAMPLE_PER_LANGUAGE, clean, hybrid, sample  # noqa: E402
from listings import load_heldout, load_listings, romanised_forms  # noqa: E402

if __name__ == "__main__":
    task, name = sys.argv[1:3]
    L, seed = load_listings()
    if task in ("hybrid_rom", "heldout_rom"):
        H = load_heldout() if task == "heldout_rom" else L
        hybrid(name, H, sorted(H), romanised_forms())
        print("done", task, name, len(H), flush=True)
        sys.exit(0)
    which = sys.argv[3]
    ids = sample(L, SAMPLE_PER_LANGUAGE, seed)
    if which == "small":
        ids = [i for lang in range(NINE_B_LANGUAGES) for i in [j for j in ids if L[j]["language"] == lang][
            :NINE_B_PER_LANGUAGE]]
    elif which == "all":
        ids = ids + [i for i in sorted(L) if i not in set(ids)]
    (hybrid if task == "hybrid" else clean)(name, L, ids)
    print("done", name, which, len(ids), flush=True)
