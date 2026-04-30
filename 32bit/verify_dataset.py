"""Interactive manual reviewer for data_clean/.

The old version used DeepFace's 6-class race classifier as an automatic filter.
There's no lightweight equivalent without TensorFlow, so this version shows
each image and lets you decide. Eyeballing a few hundred cropped faces is
faster than chasing down another ML dep for a 3-class POC.

Usage:
    python verify_dataset.py                # review all classes
    python verify_dataset.py --classes malay # one class

Controls (inside the window):
    y / SPACE  -> keep
    n / d      -> reject (move to data_rejected/<class>/)
    q          -> stop reviewing this class
    ESC        -> exit entirely
"""

import argparse
import shutil
from pathlib import Path

import cv2

SRC = Path(__file__).parent / "data_clean"
REJECTED = Path(__file__).parent / "data_rejected"


def review_class(cls: str, dry_run: bool) -> tuple[int, int, bool]:
    cls_dir = SRC / cls
    rej_dir = REJECTED / cls
    if not dry_run:
        rej_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(cls_dir.glob("*.jpg"))
    kept = rejected = 0
    abort_all = False

    for i, f in enumerate(files, 1):
        img = cv2.imread(str(f))
        if img is None:
            continue
        display = cv2.resize(img, (400, 400))
        title = f"{cls} {i}/{len(files)}  (y=keep  n=reject  q=next-class  ESC=exit)"
        cv2.imshow(title, display)

        while True:
            k = cv2.waitKey(0) & 0xFF
            if k in (ord('y'), ord(' ')):
                kept += 1
                break
            if k in (ord('n'), ord('d')):
                rejected += 1
                if not dry_run:
                    shutil.move(str(f), str(rej_dir / f.name))
                break
            if k == ord('q'):
                cv2.destroyWindow(title)
                return kept, rejected, False
            if k == 27:  # ESC
                cv2.destroyWindow(title)
                return kept, rejected, True
        cv2.destroyWindow(title)

    return kept, rejected, abort_all


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
        print(f"\nReviewing {cls}...")
        kept, rejected, abort = review_class(cls, args.dry_run)
        summary[cls] = (kept, rejected)
        if abort:
            print("ESC pressed, stopping.")
            break

    cv2.destroyAllWindows()

    print("\nSummary:")
    print(f"  {'class':<10} {'kept':>6} {'rejected':>9}")
    for cls, (k, r) in summary.items():
        print(f"  {cls:<10} {k:>6} {r:>9}")

    if args.dry_run:
        print("\nDRY RUN - no files moved.")
    else:
        print(f"\nRejected images -> {REJECTED}/")
        print("Then retrain: python train_ethnicity.py")


if __name__ == "__main__":
    main()
