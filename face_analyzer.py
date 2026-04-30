"""Webcam face analyzer using OpenCV DNN (no TensorFlow, no deepface).

If ethnicity_clf.pkl exists (trained via train_ethnicity.py), a Malay / Chinese /
Indian prediction is shown based on SFace embeddings. Otherwise that line is
skipped.

Press 'q' to quit.
"""

from pathlib import Path

import cv2
import numpy as np

import models

ANALYZE_EVERY_N_FRAMES = 15  # age/gender/emotion are cheap-ish; detection every frame
ETHNICITY_MODEL_PATH = Path(__file__).parent / "ethnicity_clf.pkl"
MIN_ETHNICITY_CONF = 55.0


def load_ethnicity_model():
    if not ETHNICITY_MODEL_PATH.exists():
        return None
    import joblib
    bundle = joblib.load(ETHNICITY_MODEL_PATH)
    print(f"Loaded ethnicity classifier: {bundle['labels']} "
          f"(embed={bundle['embed_model']})")
    return bundle


def predict_ethnicity(bundle, frame, face):
    try:
        vec = models.embed_face(frame, face).reshape(1, -1)
        proba = bundle["clf"].predict_proba(vec)[0]
        idx = int(np.argmax(proba))
        return bundle["labels"][idx], float(proba[idx]) * 100
    except Exception as e:
        print(f"ethnicity predict error: {e}")
        return None, 0.0


def analyze_face(frame, face, ethnicity_bundle):
    crop = models.crop_face(frame, face, pad=0.1)
    if crop.size == 0:
        return None

    age, age_conf = models.predict_age(crop)
    gender, gender_conf = models.predict_gender(crop)
    emotion, emotion_conf = models.predict_emotion(crop)

    result = {
        "region": {"x": face["x"], "y": face["y"], "w": face["w"], "h": face["h"]},
        "face_conf": face["confidence"] * 100,
        "age": age, "age_conf": age_conf,
        "gender": gender, "gender_conf": gender_conf,
        "emotion": emotion, "emotion_conf": emotion_conf,
    }

    if ethnicity_bundle is not None:
        label, conf = predict_ethnicity(ethnicity_bundle, frame, face)
        result["ethnicity"] = (label, conf)

    return result


def draw_result(frame, r):
    reg = r["region"]
    x, y, w, h = reg["x"], reg["y"], reg["w"], reg["h"]
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    lines = [
        f"Face: {r['face_conf']:.1f}%",
        f"Age: {r['age']}",
        f"Gender: {r['gender']} ({r['gender_conf']:.1f}%)",
    ]
    if "ethnicity" in r:
        label, conf = r["ethnicity"]
        if label is None:
            lines.append("Ethnicity: ?")
        elif conf < MIN_ETHNICITY_CONF:
            lines.append(f"Ethnicity: uncertain ({label} {conf:.0f}%)")
        else:
            lines.append(f"Ethnicity: {label} ({conf:.1f}%)")
    lines.append(f"Emotion: {r['emotion']} ({r['emotion_conf']:.1f}%)")

    y_text = max(y - 10, 15)
    for i, line in enumerate(lines):
        cv2.putText(
            frame, line,
            (x, y_text + i * 18 - (len(lines) - 1) * 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA,
        )


def main():
    ethnicity_bundle = load_ethnicity_model()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    frame_idx = 0
    last_results = []

    print("Starting camera. Press 'q' in the video window to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Run the heavy pipeline every N frames; reuse last_results in between
        if frame_idx % ANALYZE_EVERY_N_FRAMES == 0:
            try:
                faces = models.detect_faces(frame)
                last_results = [r for r in (analyze_face(frame, f, ethnicity_bundle)
                                             for f in faces) if r is not None]
            except Exception as e:
                print(f"Analyze error: {e}")

        for r in last_results:
            draw_result(frame, r)
        cv2.imshow("Face Analyzer (press q to quit)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
