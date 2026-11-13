"""Exposure classification using the Random Forest trained in Google Colab.

Training features (in order): mean, std, dark fraction (<64), bright
fraction (>192), p10, p50, p90 of OpenCV BGR2GRAY after resize to 128x128.
"""
from functools import lru_cache
from pathlib import Path
import cv2
import joblib
import numpy as np

MODEL_PATH = Path(__file__).resolve().parent / 'random_forest_3class.joblib'
LABELS = {0: 'Low-light', 1: 'Normal', 2: 'Overexposed'}


def extract_features(rgb_image):
    """Reproduce the Colab feature extraction, including uint8 conversion."""
    rgb_image = np.asarray(rgb_image, dtype=np.uint8)
    bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    bgr = cv2.resize(bgr, (128, 128))
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return [
        float(np.mean(gray)),
        float(np.std(gray)),
        float(np.mean(gray < 64)),
        float(np.mean(gray > 192)),
        float(np.percentile(gray, 10)),
        float(np.percentile(gray, 50)),
        float(np.percentile(gray, 90)),
    ]


@lru_cache(maxsize=1)
def get_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f'Trained AI model not found: {MODEL_PATH}')
    model = joblib.load(MODEL_PATH)  # Only load your own trusted model file.
    if model.n_features_in_ != 7 or set(model.classes_) != {0, 1, 2}:
        raise ValueError('Model does not match the expected seven features and three classes.')
    return model


def predict(rgb_image):
    model = get_model()
    x = [extract_features(rgb_image)]
    class_id = int(model.predict(x)[0])
    probabilities = model.predict_proba(x)[0]
    confidence = float(probabilities[list(model.classes_).index(class_id)])
    by_class = {LABELS[int(c)]: float(p) for c, p in zip(model.classes_, probabilities)}
    return LABELS[class_id], confidence, by_class
