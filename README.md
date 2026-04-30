# OpenCV Face Analyzer

Two parallel implementations of the same feature set (face detection, age,
gender, emotion, 3-class ethnicity: Malay / Chinese / Indian). Pick the
folder that matches your runtime:

| Folder | Stack | Runs on |
|---|---|---|
| `32bit/` | OpenCV DNN only (ONNX/Caffe) | Raspberry Pi 4 with 32-bit Raspberry Pi OS (armv7l). No TensorFlow. |
| `64bit/` | DeepFace + TensorFlow | Mac, Linux x86_64, Pi with 64-bit OS. Higher ethnicity accuracy. |

Both folders share:
- `../data/` — raw scraped images (shared source of truth)
- `../scrape_images.py` — Google + Bing scraper (icrawler, no ML deps)

Each folder owns its own `data_clean/`, `ethnicity_clf.pkl`,
`requirements.txt`, and training scripts, so they don't cross-contaminate.

## Quick start

```bash
# One-time: scrape ~1000 imgs/class (shared between both stacks)
python3 scrape_images.py

# Then pick a stack:
cd 64bit && python3 -m pip install -r requirements.txt   # on Mac / 64-bit Pi
# or
cd 32bit && ./download_models.sh && python3 -m pip install -r requirements.txt
```

See each folder's README for the rest.

## Why two folders?

`deepface` pulls in `tensorflow>=1.9`, and Google has never published a
TensorFlow wheel for armv7l (32-bit ARM). The 32-bit Pi path replaces the
entire ML stack with `cv2.dnn` + OpenCV Zoo ONNX models, giving a working
POC on the Pi at the cost of ~5-10 pp accuracy on the ethnicity classifier
(SFace 128-d embeddings vs Facenet512 512-d).

Keep the 64-bit folder around: if you ever reflash the Pi to 64-bit OS, or
deploy this anywhere else, use that one.
