"""Train a 3-class ethnicity classifier (malay / chinese / indian) on face embeddings.

Pipeline:
  1. For each image in data_clean/<class>/, run an embedding model -> vector.
  2. Train logistic regression with class-balanced weights.
  3. Report held-out accuracy + confusion matrix.
  4. Save classifier + label encoder to ethnicity_clf.pkl (auto-detected by
     face_analyzer.py at runtime).

Usage:
    python train_ethnicity.py                        # default Facenet512
    python train_ethnicity.py --model ArcFace        # usually +3-5% accuracy
    python train_ethnicity.py --model VGG-Face       # slower, sometimes better
    python train_ethnicity.py --c 0.5                # stronger regularization
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
from deepface import DeepFace
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, normalize
from tqdm import tqdm

DATA = Path(__file__).parent / "data_clean"
MODEL_OUT = Path(__file__).parent / "ethnicity_clf.pkl"

SUPPORTED_MODELS = ["Facenet512", "ArcFace", "VGG-Face", "Facenet", "OpenFace", "SFace"]


def extract_embeddings(embed_model: str):
    X, y = [], []
    image_files = []
    for class_dir in sorted(p for p in DATA.iterdir() if p.is_dir()):
        files = list(class_dir.glob("*.jpg"))
        print(f"  {class_dir.name}: {len(files)} images")
        for f in files:
            image_files.append((f, class_dir.name))

    for img_path, label in tqdm(image_files, desc=f"embed ({embed_model})"):
        try:
            rep = DeepFace.represent(
                img_path=str(img_path),
                model_name=embed_model,
                detector_backend="skip",  # images are already cropped
                enforce_detection=False,
                align=False,
            )
            vec = rep[0]["embedding"] if isinstance(rep, list) else rep["embedding"]
            X.append(vec)
            y.append(label)
        except Exception as e:
            print(f"    skip {img_path.name}: {e}")

    return np.asarray(X, dtype=np.float32), np.asarray(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Facenet512", choices=SUPPORTED_MODELS,
                    help="Embedding model (default Facenet512). "
                         "ArcFace usually +3-5%% accuracy but slower.")
    ap.add_argument("--c", type=float, default=1.0,
                    help="Logistic regression inverse regularization strength "
                         "(default 1.0; try 0.3-3.0).")
    args = ap.parse_args()

    if not DATA.exists():
        raise SystemExit(f"{DATA} does not exist. Run clean_dataset.py first.")

    print(f"Embedding model: {args.model}")
    print("Extracting embeddings...")
    X, y = extract_embeddings(args.model)
    print(f"Total samples: {len(X)}  |  feature dim: {X.shape[1]}")

    # L2-normalize embeddings so logistic regression operates on direction, not magnitude
    X = normalize(X)

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

    # NB: embed_model is saved in the pkl so face_analyzer.py automatically
    # uses the matching embedding model at inference time.
    joblib.dump({"clf": clf, "labels": le.classes_.tolist(), "embed_model": args.model},
                MODEL_OUT)
    print(f"\nSaved classifier -> {MODEL_OUT} (embed_model={args.model})")


if __name__ == "__main__":
    main()
