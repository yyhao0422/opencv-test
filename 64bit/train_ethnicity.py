"""Train a 3-class ethnicity classifier (malay / chinese / indian) on
DeepFace Facenet512 embeddings.

Pipeline:
  1. For each image in data_clean/<class>/, run Facenet512 -> 512-d embedding.
  2. Train logistic regression with class-balanced weights.
  3. Report held-out accuracy + confusion matrix.
  4. Save classifier to ethnicity_clf.pkl.

The pkl is portable: face_analyzer.py auto-loads it on start.

Usage:
    python train_ethnicity.py
    python train_ethnicity.py --c 0.5        # stronger regularization
    python train_ethnicity.py --embed ArcFace
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
from deepface import DeepFace
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

import cv2

DATA = Path(__file__).parent / "data_clean"
MODEL_OUT = Path(__file__).parent / "ethnicity_clf.pkl"


def embed(img_bgr, model_name):
    rep = DeepFace.represent(
        img_path=img_bgr,
        model_name=model_name,
        detector_backend="skip",
        enforce_detection=False,
        align=False,
    )
    vec = rep[0]["embedding"] if isinstance(rep, list) else rep["embedding"]
    vec = np.asarray(vec, dtype=np.float32)
    n = np.linalg.norm(vec) + 1e-9
    return vec / n


def extract_embeddings(model_name):
    X, y = [], []
    image_files = []
    for class_dir in sorted(p for p in DATA.iterdir() if p.is_dir()):
        files = list(class_dir.glob("*.jpg"))
        print(f"  {class_dir.name}: {len(files)} images")
        for f in files:
            image_files.append((f, class_dir.name))

    n_err = 0
    for img_path, label in tqdm(image_files, desc=f"embed ({model_name})"):
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            vec = embed(img, model_name)
            X.append(vec)
            y.append(label)
        except Exception:
            n_err += 1

    if n_err:
        print(f"  skipped {n_err} images with embedding errors")
    return np.asarray(X, dtype=np.float32), np.asarray(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c", type=float, default=1.0,
                    help="LogReg inverse regularization strength (try 0.3-3.0).")
    ap.add_argument("--embed", default="Facenet512",
                    help="DeepFace embedding model (Facenet512, ArcFace, VGG-Face, ...)")
    args = ap.parse_args()

    if not DATA.exists():
        raise SystemExit(f"{DATA} does not exist. Run clean_dataset.py first.")

    print(f"Embedding model: {args.embed}")
    X, y = extract_embeddings(args.embed)
    print(f"Total samples: {len(X)}  |  feature dim: {X.shape[1]}")

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    print(f"Classes: {list(le.classes_)}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_enc, test_size=0.2, stratify=y_enc, random_state=42
    )

    clf = LogisticRegression(
        max_iter=2000,
        C=args.c,
        class_weight="balanced",
        multi_class="multinomial",
    )
    clf.fit(X_tr, y_tr)

    y_pred = clf.predict(X_te)
    print("\nHeld-out performance:")
    print(classification_report(y_te, y_pred, target_names=le.classes_))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_te, y_pred))

    joblib.dump(
        {"clf": clf, "labels": le.classes_.tolist(), "embed_model": args.embed},
        MODEL_OUT,
    )
    print(f"\nSaved classifier -> {MODEL_OUT} (embed_model={args.embed})")


if __name__ == "__main__":
    main()
