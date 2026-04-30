"""Simple webcam face analyzer using OpenCV + DeepFace.

If ethnicity_clf.pkl exists (trained via train_ethnicity.py), the "Race" field
is replaced with a custom Malay / Chinese / Indian prediction driven by face
embeddings. Otherwise falls back to DeepFace's built-in 6-class race model.

Press 'q' to quit.
"""

from pathlib import Path

import cv2
import numpy as np
from deepface import DeepFace

ACTIONS = ["age", "gender", "race", "emotion"]
ANALYZE_EVERY_N_FRAMES = 15  # DeepFace is slow; don't run on every frame
DETECTOR_BACKEND = "opencv"   # fast; swap to "retinaface" for higher accuracy
ETHNICITY_MODEL_PATH = Path(__file__).parent / "ethnicity_clf.pkl"
MIN_ETHNICITY_CONF = 55.0     # below this, show "uncertain"


def load_ethnicity_model():
    if not ETHNICITY_MODEL_PATH.exists():
        return None
    import joblib
    bundle = joblib.load(ETHNICITY_MODEL_PATH)
    print(f"Loaded custom ethnicity classifier: {bundle['labels']} "
          f"(embed={bundle['embed_model']})")
    return bundle


def predict_ethnicity(bundle, face_bgr):
    """Return (label, confidence_percent) or (None, 0)."""
    try:
        rep = DeepFace.represent(
            img_path=face_bgr,
            model_name=bundle["embed_model"],
            detector_backend="skip",
            enforce_detection=False,
            align=False,
        )
        vec = rep[0]["embedding"] if isinstance(rep, list) else rep["embedding"]
        vec = np.asarray(vec, dtype=np.float32).reshape(1, -1)
        vec /= (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-9)
        proba = bundle["clf"].predict_proba(vec)[0]
        idx = int(np.argmax(proba))
        return bundle["labels"][idx], float(proba[idx]) * 100
    except Exception as e:
        print(f"ethnicity predict error: {e}")
        return None, 0.0


def draw_results(frame, results, ethnicity_bundle):
    for r in results:
        region = r.get("region", {})
        x, y, w, h = region.get("x", 0), region.get("y", 0), region.get("w", 0), region.get("h", 0)
        if w == 0 or h == 0:
            continue

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        age = r.get("age")
        gender = r.get("dominant_gender")
        emotion = r.get("dominant_emotion")
        face_conf = r.get("face_confidence", 0) * 100
        gender_conf = r.get("gender", {}).get(gender, 0)
        emotion_conf = r.get("emotion", {}).get(emotion, 0)

        # Ethnicity: custom classifier if available, else DeepFace race
        if ethnicity_bundle is not None:
            face_crop = frame[max(y, 0):y + h, max(x, 0):x + w]
            if face_crop.size > 0:
                label, conf = predict_ethnicity(ethnicity_bundle, face_crop)
                if label is None:
                    ethnicity_line = "Ethnicity: ?"
                elif conf < MIN_ETHNICITY_CONF:
                    ethnicity_line = f"Ethnicity: uncertain ({label} {conf:.0f}%)"
                else:
                    ethnicity_line = f"Ethnicity: {label} ({conf:.1f}%)"
            else:
                ethnicity_line = "Ethnicity: ?"
        else:
            race = r.get("dominant_race")
            race_conf = r.get("race", {}).get(race, 0)
            ethnicity_line = f"Race: {race} ({race_conf:.1f}%)"

        lines = [
            f"Face: {face_conf:.1f}%",
            f"Age: {age}",
            f"Gender: {gender} ({gender_conf:.1f}%)",
            ethnicity_line,
            f"Emotion: {emotion} ({emotion_conf:.1f}%)",
        ]

        y_text = max(y - 10, 15)
        for i, line in enumerate(lines):
            cv2.putText(
                frame,
                line,
                (x, y_text + i * 18 - (len(lines) - 1) * 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )


def main():
    ethnicity_bundle = load_ethnicity_model()
    # Drop "race" from DeepFace actions when we have our own model — saves ~500 MB RAM + speed
    actions = [a for a in ACTIONS if a != "race"] if ethnicity_bundle else ACTIONS

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check camera permissions for your terminal.")

    frame_idx = 0
    last_results = []

    print("Starting camera. Press 'q' in the video window to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % ANALYZE_EVERY_N_FRAMES == 0:
            try:
                analysis = DeepFace.analyze(
                    img_path=frame,
                    actions=actions,
                    detector_backend=DETECTOR_BACKEND,
                    enforce_detection=False,
                    silent=True,
                )
                last_results = analysis if isinstance(analysis, list) else [analysis]
            except Exception as e:
                print(f"Analyze error: {e}")

        draw_results(frame, last_results, ethnicity_bundle)
        cv2.imshow("Face Analyzer (press q to quit)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
