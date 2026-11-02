# AI Recommender Systems: companion code

Working code for the book *AI Recommender Systems*. Each chapter has its own folder, and every listing in the book is a short excerpt of a full, runnable file here.

The running example throughout the book is **Tesselo**, a fictional global platform with a shop, a video feed, ads and an AI assistant.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Chapters

| Folder | Chapter | Run |
| --- | --- | --- |
| `chapter_01` | The Anatomy of a Modern Recommender: the five-stage funnel in miniature | `python chapter_01/listing_1_1_funnel.py` |

Tested with Python 3.13 and 3.14 and NumPy 2.5. Other versions may print a different final slate.
