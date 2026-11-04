# AI Recommender Systems: companion code

Working code for the book *AI Recommender Systems*. Each chapter has its own folder, and every listing in the book is a short excerpt of a full, runnable file here.

The running example throughout the book is **Tesselo**, a fictional global platform with a shop, a video feed, ads and an AI assistant.

## Setup

You need Python 3.12 or later, because NumPy 2.5.4 (the tested version) does not support older versions of Python.

```bash
python -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

To reproduce the book's printed output exactly, install the tested versions instead:

```bash
pip install -r requirements.txt -c constraints.txt
```

## Chapters

| Folder | Chapter | Run |
| --- | --- | --- |
| `chapter_01` | The Anatomy of a Modern Recommender: the five-stage funnel in miniature | `python chapter_01/listing_1_1_funnel.py` |
| `chapter_02` | Data, Feedback and User Signals: the Tesselo data generator with planted biases (Listing 2.3), and from raw logs to implicit scores (Listings 2.1 and 2.2) | `python chapter_02/tesselo_data.py`<br>`python chapter_02/implicit_scores.py` |
| `chapter_03` | Implementing a Simple Recommender: shop orders and item attributes for Tesselo, charts, trending and "bought together" rules (Listings 3.1 and 3.2), similarity, content-based filtering, item-based collaborative filtering, k-means segments and a time-split test (Listings 3.3 to 3.7) | `python chapter_03/tesselo_shop.py`<br>`python chapter_03/popular_and_together.py`<br>`python chapter_03/similarity_recs.py` |
| `chapter_04` | Matrix Factorisation and Embedding-Based Recommenders: star ratings for Tesselo's shop and biased matrix factorisation by SGD (Listing 4.1), iALS and BPR for implicit feedback, neighbours, fold-in, item vectors from content and the time-split test (Listings 4.2 to 4.6) | `python chapter_04/tesselo_ratings.py`<br>`python chapter_04/mf_ratings.py`<br>`python chapter_04/implicit_mf.py` (about five minutes) |
| `chapter_05` | Evaluating What Makes a Good Recommendation: business metrics, ranking metrics (Listing 5.1), calibration with Platt scaling, diversity, novelty, serendipity and exposure (Listing 5.2), baselines, splits and the sampled-metrics trap, a simulated online test (Listing 5.3) and an A/B test with sample sizes | `python chapter_05/metrics.py`<br>`python chapter_05/evaluate.py` (about four minutes) |
| `chapter_06` | Retrieval with Encoders: Two-Tower Models and Vector Search: a two-tower model in PyTorch with nine ways to choose negatives, the logQ correction, hard and impression negatives, and the honest test against Chapter 5's baselines (Listings 6.1 and 6.3); late interaction (MaxSim) against one pooled vector (Listing 6.2); a one-million-item Tesselo catalogue, MIPS as nearest-neighbour search, teaching versions of IVF, PQ and HNSW and the Faiss indexes, with filtered search (Listings 6.4 to 6.8); merging candidate sources (Listing 6.9) | `python chapter_06/two_tower.py` (about 30 minutes)<br>`python chapter_06/late_interaction.py` (under a minute)<br>`python chapter_06/tesselo_catalogue.py` (under a minute)<br>`python chapter_06/vector_search.py` (about 15 minutes; it runs `chapter_06/embed_catalogue.py` first, in its own process)<br>`python chapter_06/retrieval_channels.py` (about five minutes) |
| `chapter_07` | Pre-ranking, Heavy Ranking and Feature Interactions: training rows and features from impressions (Listing 7.1); eight heavy rankers on the same inputs, including DLRM, DCN v2 and Wukong (Listings 7.2, 7.3 and 7.7), with a shallow position tower and the honest test against Chapter 5's baselines; a planted experiment with real feature interactions; multi-task ranking with a shared bottom, MMoE and PLE (Listing 7.4); a distilled pre-ranker, the truncation trade-off and each complete pipeline with its guardrails (Listing 7.5); and video-id tables: hashing, the quotient-remainder trick and shared rows for rare ids (Listing 7.6) | `python chapter_07/heavy_ranking.py` (about 10 minutes)<br>`python chapter_07/planted_crosses.py` (about three minutes)<br>`python chapter_07/multitask.py` (about six minutes)<br>`python chapter_07/pre_ranking.py` (about four minutes)<br>`python chapter_07/embedding_tables.py` (about four minutes) |
| `chapter_08` | Filtering, Learning to Rank, Hybrids and Reranking: hard filters as set operations with Roaring Bitmaps (Listing 8.1); pointwise, pairwise, listwise and LambdaRank losses on one network, BPR and LambdaMART (Listing 8.2); stacking and feature-weighted linear stacking (Listing 8.3); a small cross-encoder against a bi-encoder (Listing 8.4); MMR and a whole-row builder, with a planted boredom test (Listing 8.5); creator limits, a new-video slot and ad allocation (Listing 8.6); and the honest test of complete rows | `python chapter_08/filtering.py` (about two minutes)<br>`python chapter_08/learning_to_rank.py` (about three minutes)<br>`python chapter_08/hybrids.py` (about two minutes)<br>`python chapter_08/cross_encoder.py` (about 13 minutes)<br>`python chapter_08/slates.py` (about three minutes)<br>`python chapter_08/honest_test.py` (about three minutes) |

Tested with Python 3.13 and 3.14 and NumPy 2.5.4 (pinned in `constraints.txt`); Chapters 6 to 8 with Python 3.13, PyTorch 2.15.0 and faiss-cpu 1.15.1 (Chapter 7 needs PyTorch only), and Chapter 8 with pyroaring 1.2.0 and LightGBM 4.7.0. Other versions may print a different final slate. Chapter 4's star-rating results come from SGD, which makes many small floating-point updates, so on a different processor their last decimal can differ slightly.

Chapter 6 needs PyTorch and Faiss; Chapter 7 needs PyTorch only (it imports Chapter 6's two-tower code, not Faiss); Chapter 8 needs PyTorch, pyroaring and LightGBM, and imports Chapters 6 and 7's code. The tested versions of both publish wheels on PyPI for Linux and for macOS on Apple Silicon (macOS 14 or later). On Linux, `pip install torch` downloads the CUDA build, several gigabytes; the scripts use only the CPU, so you can install the smaller CPU build first:

```bash
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt -c constraints.txt
```

Chapter 6's to Chapter 8's run times were measured on a two-core cloud machine (Intel Xeon at 2.10 GHz), and the numbers printed in the book are one reference run on that machine. On one machine the scripts repeat exactly, apart from timings: every random draw is seeded, and the scripts fix their thread counts (two for PyTorch and for NumPy's linear algebra, one for Faiss). Across machines, results can move, because processors and library builds add up floating-point numbers in different orders: PyTorch does not promise identical training on different platforms, and Faiss documents that its matrix operations are implemented differently on different processors, so they need not match exactly even on one thread. The results affected are those that depend on a trained two-tower model or, in Chapters 7 and 8, a trained ranker, including the order of methods within one language, and the index recall of `vector_search.py`. Two external runs with the pinned versions (the first on an Apple Silicon Mac) gave, for example, hit@10 0.189 instead of 0.192, HNSW recall@10 0.883 instead of 0.900, and IVF-PQ recall@10 0.568 instead of 0.604 at nprobe 16. These are observed differences, not limits; another machine could differ by more. In the runs checked, none of the chapter's conclusions changed. The chapter gives standard errors for its close comparisons; read those before reading a small difference.

On macOS, the pip wheels of PyTorch and Faiss each bring their own copy of the OpenMP library, and a Python process that loads both stops with `OMP: Error #15`. No Chapter 6 script loads both: `vector_search.py` runs the PyTorch part (`embed_catalogue.py`) in a separate process.
