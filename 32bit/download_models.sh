#!/usr/bin/env bash
# Download all DNN model files into ./models/.
# Run once on every machine (Pi + Mac) before first use.

set -euo pipefail
cd "$(dirname "$0")"
mkdir -p models
cd models

dl() {
  local url="$1"
  local out="$2"
  if [[ -f "$out" ]]; then
    echo "  [skip] $out"
    return
  fi
  echo "  [get]  $out"
  curl -fL --retry 3 -o "$out" "$url"
}

echo "Face detection (YuNet, ~337 KB)"
dl "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx" \
   "face_detection_yunet_2023mar.onnx"

echo "Face embedding (SFace, ~38 MB)"
dl "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx" \
   "face_recognition_sface_2021dec.onnx"

echo "Emotion (MobileFaceNet FER, ~1 MB)"
dl "https://github.com/opencv/opencv_zoo/raw/main/models/facial_expression_recognition/facial_expression_recognition_mobilefacenet_2022july.onnx" \
   "facial_expression_recognition_mobilefacenet_2022july.onnx"

echo "Age (Levi-Hassner Caffe, ~43 MB)"
dl "https://github.com/smahesh29/Gender-and-Age-Detection/raw/master/age_deploy.prototxt" \
   "age_deploy.prototxt"
dl "https://github.com/smahesh29/Gender-and-Age-Detection/raw/master/age_net.caffemodel" \
   "age_net.caffemodel"

echo "Gender (Levi-Hassner Caffe, ~13 MB)"
dl "https://github.com/smahesh29/Gender-and-Age-Detection/raw/master/gender_deploy.prototxt" \
   "gender_deploy.prototxt"
dl "https://github.com/smahesh29/Gender-and-Age-Detection/raw/master/gender_net.caffemodel" \
   "gender_net.caffemodel"

echo
echo "Done. Model files in ./models/"
ls -lh .
