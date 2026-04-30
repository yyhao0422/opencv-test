"""Clean scraped dataset: detect + crop the primary face in each image.

Rejects images with 0 or >1 detected faces. Reports gender distribution per
class so you can verify male/female balance.

Usage:
    python clean_dataset.py

Input:  data/<class>/*.{jpg,jpeg,png,webp,bmp}
Output: data_clean/<class>/*.jpg   (224x224 face crops)
"""

import shutil
from collections import Counter
from pathlib import Path

import cv2
from tqdm import tqdm

import models

SRC = Path(__file__).parent.parent / "data"
DST = Path(__file__).parent / "data_clean"
CROP_SIZE = 224
MIN_FACE_CONF = 0.8


def process_image(img_path: Path, out_dir: Path):
    """Return (status, gender_or_None)."""
    img = cv2.imread(str(img_path))
    if img is None:
        return "unreadable", None

    faces = models.detect_faces(img)
    valid = [f for f in faces if f["confidence"] >= MIN_FACE_CONF]
    if len(valid) != 1:
        return f"bad_count_{len(valid)}", None

    crop = models.crop_face(img, valid[0], pad=0.15)
    if crop.size == 0:
        return "empty_crop", None
    crop = cv2.resize(crop, (CROP_SIZE, CROP_SIZE))

    gender = None
    try:
        gender, _ = models.predict_gender(crop)
    except Exception:
        pass

    out_path = out_dir / (img_path.stem + ".jpg")
    cv2.imwrite(str(out_path), crop)
    return "ok", gender


def main():
    if DST.exists():
        shutil.rmtree(DST)

    overall = {}
    for class_dir in sorted(p for p in SRC.iterdir() if p.is_dir()):
        out_dir = DST / class_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)

        images = [p for p in class_dir.iterdir()
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}]
        stats = Counter()
        genders = Counter()

        for img_path in tqdm(images, desc=f"clean {class_dir.name}"):
            status, gender = process_image(img_path, out_dir)
            stats[status] += 1
            if status == "ok" and gender:
                genders[gender] += 1

        total_ok = stats["ok"]
        print(f"  {class_dir.name}: {dict(stats)}")
        if total_ok:
            m = genders.get("Male", 0)
            w = genders.get("Female", 0)
            print(f"    gender split: Male={m} ({m / total_ok:.0%})  "
                  f"Female={w} ({w / total_ok:.0%})")
        overall[class_dir.name] = (total_ok, genders.get("Male", 0), genders.get("Female", 0))

    print(f"\nClean dataset written to {DST}/")
    print("\nSummary:")
    print(f"  {'class':<10} {'total':>6} {'men':>6} {'women':>6}")
    for cls, (total, m, w) in overall.items():
        print(f"  {cls:<10} {total:>6} {m:>6} {w:>6}")
    print("\nIf any class is badly imbalanced (<25% of either gender), "
          "rerun scrape with additional queries for the missing gender.")


if __name__ == "__main__":
    main()
