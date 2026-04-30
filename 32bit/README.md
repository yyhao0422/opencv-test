# OpenCV Face Analyzer (Pi-friendly, no TensorFlow)

Pure OpenCV-DNN face analysis: detection, age, gender, emotion, and a 3-class
ethnicity classifier (Malay / Chinese / Indian). Runs on Raspberry Pi 4 with
32-bit Raspberry Pi OS — no deepface, no TensorFlow, no TFLite.

## First-time setup (both Mac and Pi)

```bash
./download_models.sh              # ~130 MB of ONNX/Caffe weights into models/
python3 -m pip install -r requirements.txt
```

## On the Pi — just run the analyzer

```bash
python3 face_analyzer.py          # q to quit
```

If `ethnicity_clf.pkl` is present it'll show the Malay/Chinese/Indian line;
otherwise it's skipped.

## On the Mac — full pipeline (once)

```bash
python3 scrape_images.py          # grabs ~1000 imgs/class into data/
python3 clean_dataset.py          # crops faces -> data_clean/
python3 verify_dataset.py         # interactive y/n reviewer (optional)
python3 train_ethnicity.py        # -> ethnicity_clf.pkl
```

Then copy the trained classifier to the Pi:

```bash
scp ethnicity_clf.pkl pi@<pi-ip>:~/Desktop/opencv-test/
```

## Models used

| Task | Model | Size |
|---|---|---|
| Face detect + 5 landmarks | YuNet (OpenCV Zoo) | 337 KB |
| Face embedding (128-d) | SFace (OpenCV Zoo) | 38 MB |
| Age (8 buckets) | Levi-Hassner Caffe | 43 MB |
| Gender (M/F) | Levi-Hassner Caffe | 43 MB |
| Emotion (7 classes) | MobileFaceNet FER (OpenCV Zoo) | 1 MB |

All loaded via `cv2.dnn`. No ML framework dependency — only `opencv-python`.

## Notes

- **Retraining is required** after this migration: the old
  `ethnicity_clf.pkl` was trained on DeepFace's Facenet512 (512-d) embeddings.
  The new pipeline uses SFace (128-d), so the old pkl is incompatible.
- Age output is coarse (buckets, not a number) — that's Levi-Hassner's limitation.
- On a Pi 4 expect ~2-3 fps with the analysis step running every frame. The
  `ANALYZE_EVERY_N_FRAMES` throttle in `face_analyzer.py` trades freshness for
  smoother video.
