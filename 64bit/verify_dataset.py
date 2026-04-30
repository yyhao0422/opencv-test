"""Automatically verify data_clean/ using DeepFace's 6-class race classifier.

For each class folder, checks that DeepFace's dominant_race prediction is
consistent with the folder label. Images that disagree are moved to
data_rejected/<class>/ for review.

Mapping (DeepFace 6-class -> our 3-class):
    malay   <- latino hispanic, middle eastern, (sometimes indian/asian borderline)
    chinese <- asian
    indian  <- indian

The DeepFace race model is noisy on Malay (no direct class), so we treat
indian as indian, asian as chinese, and accept latino/middle-eastern/white
predictions as plausible malay. Anything else for malay is rejected.

Usage:
    python verify_dataset.py                  # auto-filter all classes
    python verify_dataset.py --classes malay  # one class
    python verify_dataset.py --dry-run        # don't move anything

The TF-free version in 32bit/ is an interactive manual reviewer instead.
"""

import argparse
import shutil
from collections import Counter
from pathlib import Path

import cv2
from deepface import DeepFace
from tqdm import tqdm

SRC = Path(__file__).parent / "data_clean"
REJECTED = Path(__file__).parent / "data_rejected"

ACCEPTABLE = {
    "chinese": {"asian"},
    "indian":  {"indian"},
    "malay":   {"latino hispanic", "middle eastern", "white", "indian", "asian"},
}


def predict_race(img_path: Path) -> str | None:
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    try:
        result = DeepFace.analyze(
            img_path=img,
            actions=["race"],
            detector_backend="skip",
            enforce_detection=False,
            silent=True,
        )
        result = result[0] if isinstance(result, list) else result
        return result.get("dominant_race")
    except Exception:
        return None


def review_class(cls: str, dry_run: bool) -> tuple[int, int, Counter]:
    cls_dir = SRC / cls
    rej_dir = REJECTED / cls
    if not dry_run:
        rej_dir.mkdir(parents=True, exist_ok=True)

    acceptable = ACCEPTABLE.get(cls, set())
    kept = rejected = 0
    race_counts = Counter()

    for f in tqdm(sorted(cls_dir.glob("*.jpg")), desc=f"verify {cls}"):
        race = predict_race(f)
        race_counts[race or "unknown"] += 1
        if race is None or race not in acceptable:
            rejected += 1
            if not dry_run:
                shutil.move(str(f), str(rej_dir / f.name))
        else:
            kept += 1

    return kept, rejected, race_counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", nargs="+",
                    help="Subset of classes to review (default: all).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Don't move rejects, just count.")
    args = ap.parse_args()

    if not SRC.exists():
        raise SystemExit(f"{SRC} not found. Run clean_dataset.py first.")

    classes = args.classes or sorted(p.name for p in SRC.iterdir() if p.is_dir())
    summary = {}

    for cls in classes:
        if not (SRC / cls).exists():
            print(f"  skip {cls}: no folder")
            continue
        print(f"\nReviewing {cls}... (acceptable races: {ACCEPTABLE.get(cls, set())})")
        kept, rejected, races = review_class(cls, args.dry_run)
        summary[cls] = (kept, rejected, races)

    print("\nSummary:")
    print(f"  {'class':<10} {'kept':>6} {'rejected':>9}")
    for cls, (k, r, _) in summary.items():
        print(f"  {cls:<10} {k:>6} {r:>9}")

    print("\nPer-class race distribution (DeepFace predictions):")
    for cls, (_, _, races) in summary.items():
        print(f"  {cls}: {dict(races)}")

    if args.dry_run:
        print("\nDRY RUN - no files moved.")
    else:
        print(f"\nRejected images -> {REJECTED}/")
        print("Then retrain: python train_ethnicity.py")


if __name__ == "__main__":
    main()
