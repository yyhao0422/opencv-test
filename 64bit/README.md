# OpenCV Face Analyzer — 64-bit / DeepFace build

DeepFace + TensorFlow version. Runs on Mac, Linux x86_64, and Raspberry Pi
with 64-bit OS. For 32-bit Pi OS use `../32bit/` instead.

## Setup

```bash
python3 -m pip install -r requirements.txt
```

First run will auto-download DeepFace's model weights (~500 MB into
`~/.deepface/weights/`).

## Run the analyzer

```bash
python3 face_analyzer.py          # q to quit
```

If `ethnicity_clf.pkl` is present, the "Race" line is replaced with the
Malay / Chinese / Indian prediction. Otherwise DeepFace's built-in 6-class
race model is shown.

## Full pipeline (from scratch)

```bash
python3 ../scrape_images.py       # shared data/ folder (parent)
python3 clean_dataset.py          # crops faces -> data_clean/
python3 verify_dataset.py         # auto-filter via DeepFace race model
python3 train_ethnicity.py        # -> ethnicity_clf.pkl (Facenet512, 512-d)
```

## Models used

| Task | Source |
|---|---|
| Face detection | OpenCV Haar (DeepFace `detector_backend="opencv"`) |
| Age / gender / emotion / race | DeepFace built-in (VGG-Face / Facenet etc.) |
| Ethnicity embedding | Facenet512 (default, overrideable via `--embed ArcFace`) |

## Notes

- `verify_dataset.py` here does an auto-filter using DeepFace's 6-class race
  output, which is not available in the 32-bit build. The 32-bit build uses
  an interactive manual reviewer instead.
- Ethnicity accuracy with Facenet512 is typically ~5-10 pp higher than the
  SFace-based 32-bit build.
- The `ethnicity_clf.pkl` that ships here is the legacy one from `../backup/`.
  Rerun `train_ethnicity.py` whenever you refresh `data_clean/`.
