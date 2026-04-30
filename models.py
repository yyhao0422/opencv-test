"""OpenCV-DNN wrappers. No TensorFlow, no deepface.

Shared by face_analyzer.py (Pi), clean_dataset.py, train_ethnicity.py.
Call get_*() lazily; models load once and are cached in-process.
"""

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

MODELS_DIR = Path(__file__).parent / "models"

# Levi-Hassner mean + buckets
AGE_MEAN = (78.4263377603, 87.7689143744, 114.895847746)
AGE_BUCKETS = ["0-2", "4-6", "8-12", "15-20", "25-32", "38-43", "48-53", "60-100"]
GENDER_LABELS = ["Male", "Female"]

# OpenCV Zoo FER output order
FER_LABELS = ["angry", "disgust", "fearful", "happy", "neutral", "sad", "surprised"]

_cache: dict = {}


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"Model file missing: {path}\n"
            f"Run ./download_models.sh first."
        )
    return path


def get_face_detector():
    if "yunet" not in _cache:
        model = _require(MODELS_DIR / "face_detection_yunet_2023mar.onnx")
        _cache["yunet"] = cv2.FaceDetectorYN.create(
            str(model), "", (320, 320),
            score_threshold=0.6, nms_threshold=0.3, top_k=50,
        )
    return _cache["yunet"]


def get_sface():
    """FaceRecognizerSF handles 5-landmark alignment + embedding internally."""
    if "sface" not in _cache:
        model = _require(MODELS_DIR / "face_recognition_sface_2021dec.onnx")
        _cache["sface"] = cv2.FaceRecognizerSF.create(str(model), "")
    return _cache["sface"]


def get_age_net():
    if "age" not in _cache:
        proto = _require(MODELS_DIR / "age_deploy.prototxt")
        weights = _require(MODELS_DIR / "age_net.caffemodel")
        _cache["age"] = cv2.dnn.readNet(str(weights), str(proto))
    return _cache["age"]


def get_gender_net():
    if "gender" not in _cache:
        proto = _require(MODELS_DIR / "gender_deploy.prototxt")
        weights = _require(MODELS_DIR / "gender_net.caffemodel")
        _cache["gender"] = cv2.dnn.readNet(str(weights), str(proto))
    return _cache["gender"]


def get_fer_net():
    if "fer" not in _cache:
        model = _require(MODELS_DIR / "facial_expression_recognition_mobilefacenet_2022july.onnx")
        _cache["fer"] = cv2.dnn.readNetFromONNX(str(model))
    return _cache["fer"]


def detect_faces(frame_bgr: np.ndarray) -> List[dict]:
    """Return list of {x, y, w, h, confidence, raw}. 'raw' is the 15-element
    YuNet row (bbox + 5 landmarks + conf) needed by FaceRecognizerSF."""
    h, w = frame_bgr.shape[:2]
    det = get_face_detector()
    det.setInputSize((w, h))
    _, faces = det.detect(frame_bgr)
    if faces is None:
        return []
    out = []
    for f in faces:
        x, y, fw, fh = [int(v) for v in f[:4]]
        conf = float(f[14])
        out.append({
            "x": max(x, 0), "y": max(y, 0),
            "w": max(fw, 1), "h": max(fh, 1),
            "confidence": conf,
            "raw": f,
        })
    return out


def crop_face(frame_bgr: np.ndarray, face: dict, pad: float = 0.0) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    x, y, fw, fh = face["x"], face["y"], face["w"], face["h"]
    if pad > 0:
        px, py = int(fw * pad), int(fh * pad)
        x = max(0, x - px); y = max(0, y - py)
        fw = min(w - x, fw + 2 * px); fh = min(h - y, fh + 2 * py)
    return frame_bgr[y:y + fh, x:x + fw]


def embed_face(frame_bgr: np.ndarray, face: dict) -> np.ndarray:
    """SFace embedding (128-d, L2-normalized) with 5-point alignment.
    Pass the full image + one face dict from detect_faces (needs face['raw'])."""
    fr = get_sface()
    aligned = fr.alignCrop(frame_bgr, face["raw"])
    vec = fr.feature(aligned).flatten().astype(np.float32)
    n = np.linalg.norm(vec) + 1e-9
    return vec / n


def embed_from_image(frame_bgr: np.ndarray) -> np.ndarray | None:
    """Detect the largest face in the image and return its embedding, or None."""
    faces = detect_faces(frame_bgr)
    if not faces:
        return None
    biggest = max(faces, key=lambda f: f["w"] * f["h"])
    return embed_face(frame_bgr, biggest)


def predict_age(face_bgr: np.ndarray) -> Tuple[str, float]:
    net = get_age_net()
    blob = cv2.dnn.blobFromImage(face_bgr, scalefactor=1.0, size=(227, 227),
                                 mean=AGE_MEAN, swapRB=False, crop=False)
    net.setInput(blob)
    probs = net.forward().flatten()
    idx = int(np.argmax(probs))
    return AGE_BUCKETS[idx], float(probs[idx]) * 100


def predict_gender(face_bgr: np.ndarray) -> Tuple[str, float]:
    net = get_gender_net()
    blob = cv2.dnn.blobFromImage(face_bgr, scalefactor=1.0, size=(227, 227),
                                 mean=AGE_MEAN, swapRB=False, crop=False)
    net.setInput(blob)
    probs = net.forward().flatten()
    idx = int(np.argmax(probs))
    return GENDER_LABELS[idx], float(probs[idx]) * 100


def predict_emotion(face_bgr: np.ndarray) -> Tuple[str, float]:
    net = get_fer_net()
    aligned = cv2.resize(face_bgr, (112, 112))
    blob = cv2.dnn.blobFromImage(aligned, scalefactor=1.0 / 255, size=(112, 112),
                                 mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    probs = net.forward().flatten()
    # Softmax (OpenCV FER outputs logits)
    probs = np.exp(probs - probs.max())
    probs = probs / probs.sum()
    idx = int(np.argmax(probs))
    return FER_LABELS[idx], float(probs[idx]) * 100
