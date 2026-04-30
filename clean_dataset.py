"""Clean a scraped dataset by detecting + cropping the primary face in each image.

Reject images with 0 faces or >1 faces (group photos are noisy labels).
Also runs DeepFace gender detection and reports gender distribution per class
so you can verify male/female balance after scraping.

Usage:
    python clean_dataset.py

Input:  data/<class>/*.jpg
Output: data_clean/<class>/*.jpg   (square face crops, 224x224)
"""

import shutil
from collections import Counter
from pathlib import Path

import cv2
from deepface import DeepFace
from tqdm import tqdm

SRC = Path(__file__).parent / "data"
DST = Path(__file__).parent / "data_clean"
CROP_SIZE = 224
DETECTOR = "retinaface"  # slower but much fewer false positives than opencv/ssd


def process_image(img_path: Path, out_dir: Path):
    """Return (status, gender_or_None)."""
    img = cv2.imread(str(img_path))
    if img is None:
        return "unreadable", None

    try:
        faces = DeepFace.extract_faces(
            img_path=img,
            detector_backend=DETECTOR,
            enforce_detection=True,
            align=True,
        )
    except Exception:
        return "no_face", None

    valid = [f for f in faces if f.get("confidence", 0) > 0.9]
    if len(valid) != 1:
        return f"bad_count_{len(valid)}", None

    face = valid[0]["face"]  # float RGB 0-1
    face = (face * 255).astype("uint8")
    face_bgr = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
    face_bgr = cv2.resize(face_bgr, (CROP_SIZE, CROP_SIZE))

    # Quick gender pass (cheap) so we can track dataset balance
    gender = None
    try:
        res = DeepFace.analyze(
            img_path=face_bgr,
            actions=["gender"],
            detector_backend="skip",
            enforce_detection=False,
            silent=True,
        )
        res = res[0] if isinstance(res, list) else res
        gender = res.get("dominant_gender")
    except Exception:
        pass

    out_path = out_dir / (img_path.stem + ".jpg")
    cv2.imwrite(str(out_path), face_bgr)
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
            m = genders.get("Man", 0)
            w = genders.get("Woman", 0)
            print(f"    gender split: Man={m} ({m/total_ok:.0%})  "
                  f"Woman={w} ({w/total_ok:.0%})")
        overall[class_dir.name] = (total_ok, genders.get("Man", 0), genders.get("Woman", 0))

    print(f"\nClean dataset written to {DST}/")
    print("\nSummary:")
    print(f"  {'class':<10} {'total':>6} {'men':>6} {'women':>6}")
    for cls, (total, m, w) in overall.items():
        print(f"  {cls:<10} {total:>6} {m:>6} {w:>6}")
    print("\nIf any class is badly imbalanced (<25% of either gender), "
          "rerun scrape with additional male/female queries.")


if __name__ == "__main__":
    main()
