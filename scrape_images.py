"""Scrape face images from Google + Bing for Malay / Chinese / Indian.

Each query is run on BOTH engines with split budget, so no query (and no
gender) is starved when one engine rate-limits. Progress and per-engine
counts are printed so you can see exactly what succeeded.

Default target: ~1000 images per class (10 queries * 100 each, split 50/50
across Google + Bing).

Usage:
    python scrape_images.py                       # scrape all three classes
    python scrape_images.py --classes malay       # one class
    python scrape_images.py --per-query 120       # override images per query
    python scrape_images.py --shuffle             # randomize query order
"""

import argparse
import random
from pathlib import Path

from icrawler.builtin import BingImageCrawler, GoogleImageCrawler

# Each entry: (query, gender) so we can report balance after scraping.
# Gender interleaved M/F/M/F so early termination still gives balance.
# More male queries than female to counter residual search-engine bias.
QUERIES = {
    "malay": [
        ("malay man songkok",                   "M"),
        ("malay woman face",                    "F"),
        ("melayu pakcik lelaki",                "M"),  # pakcik = uncle (male only)
        ("melayu perempuan wajah",              "F"),
        ("malay groom pengantin lelaki",        "M"),  # groom (male only)
        ("malay bride pengantin perempuan",     "F"),
        ("malay imam",                          "M"),  # religious role (male only)
        ("malay woman hijab face",              "F"),
        ("melayu lelaki tua wajah",             "M"),  # old man
        ("malay man malaysia portrait",         "M"),
    ],
    "chinese": [
        ("chinese man face",                    "M"),
        ("chinese woman face",                  "F"),
        ("chinese old man elderly",             "M"),
        ("chinese old woman elderly",           "F"),
        ("chinese groom wedding man",           "M"),
        ("chinese bride wedding woman",         "F"),
        ("chinese businessman suit headshot",   "M"),
        ("chinese businesswoman suit headshot", "F"),
        ("chinese grandfather face",            "M"),
        ("chinese father man face",             "M"),
    ],
    "indian": [
        ("indian man face",                     "M"),
        ("indian woman face",                   "F"),
        ("indian groom wedding man",            "M"),
        ("indian bride wedding woman",          "F"),
        ("indian old man elderly",              "M"),
        ("indian old woman elderly",            "F"),
        ("indian businessman suit",             "M"),
        ("tamil woman face",                    "F"),
        ("tamil man face",                      "M"),
        ("indian father man face",              "M"),
    ],
}

DATA_ROOT = Path(__file__).parent / "data"


def run_engine(engine_cls, engine_name, query, out_dir, budget):
    """Run one engine for one query. Return number of images actually downloaded."""
    before = len(list(out_dir.glob("*")))
    crawler = engine_cls(
        storage={"root_dir": str(out_dir)},
        feeder_threads=1,
        parser_threads=2,
        downloader_threads=4,
    )
    try:
        crawler.crawl(
            keyword=query,
            max_num=budget,
            min_size=(200, 200),
            file_idx_offset="auto",
        )
    except Exception as e:
        print(f"      ! {engine_name} error: {e}")
    after = len(list(out_dir.glob("*")))
    return after - before


def scrape_class(cls: str, per_query: int, shuffle: bool):
    out_dir = DATA_ROOT / cls
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = len(list(out_dir.glob("*")))
    print(f"\n=== {cls.upper()}  (existing files: {existing}) ===")

    queries = list(QUERIES[cls])
    if shuffle:
        random.shuffle(queries)

    # Split budget 50/50 between Google and Bing for EVERY query.
    # This is the fix for the previous bug where one gender got starved
    # because engine alternated by position and all male queries were on one engine.
    google_budget = per_query // 2
    bing_budget = per_query - google_budget

    totals = {"M": 0, "F": 0}
    for i, (query, gender) in enumerate(queries, 1):
        print(f"  [{i}/{len(queries)}] ({gender}) {query!r}")
        g_count = run_engine(GoogleImageCrawler, "google", query, out_dir, google_budget)
        b_count = run_engine(BingImageCrawler,   "bing",   query, out_dir, bing_budget)
        got = g_count + b_count
        print(f"      google={g_count}  bing={b_count}  (+{got})")
        totals[gender] += got

    final = len(list(out_dir.glob("*")))
    print(f"  -> {cls}: {final} files total  |  "
          f"approx M contribution={totals['M']}, F contribution={totals['F']}")
    if totals["M"] == 0:
        print("      WARNING: 0 male images downloaded for this class.")
    if totals["F"] == 0:
        print("      WARNING: 0 female images downloaded for this class.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", nargs="+", default=list(QUERIES.keys()),
                    choices=list(QUERIES.keys()))
    ap.add_argument("--per-query", type=int, default=100,
                    help="Images per query (split 50/50 Google+Bing). "
                         "10 queries/class -> 100 gives ~1000/class (default 100).")
    ap.add_argument("--shuffle", action="store_true",
                    help="Randomize query order each run.")
    args = ap.parse_args()

    DATA_ROOT.mkdir(exist_ok=True)
    for cls in args.classes:
        scrape_class(cls, args.per_query, args.shuffle)

    print("\nDone. Note the per-query google/bing counts above: if Google shows "
          "mostly 0, it was rate-limited but Bing filled in (that's the whole point).")
    print("\nNext: python clean_dataset.py  (verifies final gender balance)")


if __name__ == "__main__":
    main()
