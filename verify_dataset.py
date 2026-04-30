"""Verify cleaned dataset by running DeepFace's built-in race classifier and
filtering out obvious mismatches (e.g., a white face accidentally scraped
into data_clean/malay/).

LIMITATION: DeepFace's race classifier only has 6 broad classes -
  asian, indian, black, white, middle eastern, latino hispanic
So it CANNOT distinguish Malay vs Chinese (both labeled 'asian'). This
filter only catches cross-category label noise like westerners, etc.

Usage:
    python verify_dataset.py                # apply, move rejects to data_rejected/
    python verify_dataset.py --dry-run      # report only, don't move files
    python verify_dataset.py --threshold 60 # require >60% confidence to reject

Strategy:
  - malay, chinese folders -> require DeepFace race == "asian" (else reject)
  - indian folder         -> require DeepFace race == "indian" (else reject)
  - if DeepFace is <threshold% confident in ANY race, keep (benefit of doubt)
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

# Which DeepFace race labels are acceptable for each of our classes.
EXPECTED = {
    "malay": {"asian"},
    "chinese": {"asian"},
    "indian": {"indian"},
}


def verify_image(img_path: Path, expected: set, threshold: float) -> tuple[str, str | None]:
    img = cv2.imread(str(img_path))
    if img is None:
        return "error", None
    try:
        res = DeepFace.analyze(
            img_path=img,
            actions=["race"],
            detector_backend="skip",
            enforce_detection=False,
            silent=True,
        )
        res = res[0] if isinstance(res, list) else res
        race = res.get("dominant_race", "")
        conf = res.get("race", {}).get(race, 0)

        if race in expected:
            return "keep", race
        if conf < threshold:
            # DeepFace isn't confident enough to override our label
            return "keep_uncertain", race
        return f"reject_{race}", race
    except Exception:
        return "error", None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="Only report; don't move files.")
    ap.add_argument("--threshold", type=float, default=50.0,
                    help="Min DeepFace confidence to reject an image "
                         "(default 50 — lower = stricter).")
    args = ap.parse_args()

    if not SRC.exists():
        raise SystemExit(f"{SRC} not found. Run clean_dataset.py first.")

    summary = {}
    for cls, expected in EXPECTED.items():
        cls_dir = SRC / cls
        if not cls_dir.exists():
            print(f"  {cls}: no folder, skipping")
            continue

        rej_dir = REJECTED / cls
        if not args.dry_run:
            rej_dir.mkdir(parents=True, exist_ok=True)

        files = list(cls_dir.glob("*.jpg"))
        stats = Counter()
        race_breakdown = Counter()

        for f in tqdm(files, desc=f"verify {cls}"):
            result, race = verify_image(f, expected, args.threshold)
            stats[result] += 1
            if race:
                race_breakdown[race] += 1
            if result.startswith("reject_") and not args.dry_run:
                shutil.move(str(f), str(rej_dir / f.name))

        total = len(files)
        kept = stats["keep"] + stats["keep_uncertain"]
        rejected = sum(v for k, v in stats.items() if k.startswith("reject_"))
        pct = 100 * rejected / max(total, 1)

        print(f"\n  {cls}: {total} -> kept {kept}, rejected {rejected} ({pct:.1f}%)")
        print(f"    DeepFace race breakdown: {dict(race_breakdown)}")
        print(f"    actions: {dict(stats)}")
        summary[cls] = (total, kept, rejected)

    print("\nSummary:")
    print(f"  {'class':<10} {'before':>7} {'kept':>6} {'rejected':>9}")
    for cls, (b, k, r) in summary.items():
        print(f"  {cls:<10} {b:>7} {k:>6} {r:>9}")

    if args.dry_run:
        print("\nDRY RUN - no files moved. Rerun without --dry-run to apply.")
    else:
        print(f"\nRejected images -> {REJECTED}/. Eyeball them; if the rejects "
              f"look legitimately mislabeled, good. If many are actually correct "
              f"(DeepFace being wrong), rerun with a stricter --threshold.")
        print("\nThen retrain: python train_ethnicity.py")


if __name__ == "__main__":
    main()
