"""
Spot the Fake Photo -- predict.py

Usage:
    python predict.py some_image.jpg
Prints ONE number from 0 to 1:
    0 = real photo,  1 = photo of a screen (recapture)

Approach: 6 hand-designed image-forensics features (moire/aliasing from the
FFT, local grid periodicity via autocorrelation, glare/specular highlights,
color cast, and high-frequency detail rolloff -- see features.py for the
full reasoning) combined with a tiny logistic regression.

If model.json (produced by `python train.py` on your own real/ and screen/
folders) is present, it is used. Otherwise this falls back to fixed,
hand-set weights so the script always works out of the box, just less
accurately -- run train.py on your own photos for the real, calibrated
version.
"""

import json
import os
import sys

import numpy as np

from features import extract_features

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.json")

# Fallback weights, hand-set from the feature reasoning in features.py, used
# only if no calibrated model.json exists yet. Rough directionality:
# higher fft_peakiness, hf_ratio, autocorr_peak, glare -> more "screen-like".
# higher color_cast (blue/green skew) -> more "screen-like".
# hf_rolloff is roughly a wash on its own so it gets a small weight.
_FALLBACK_MU = np.array([0.9, 0.35, 0.25, 0.01, 0.0, 1.0])
_FALLBACK_SIGMA = np.array([0.5, 0.15, 0.15, 0.03, 0.05, 0.5])
_FALLBACK_W = np.array([1.4, 1.1, 1.3, 1.0, 0.8, 0.2])
_FALLBACK_B = -1.2


def _load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH) as f:
            m = json.load(f)
        return np.array(m["mu"]), np.array(m["sigma"]), np.array(m["w"]), float(m["b"])
    return _FALLBACK_MU, _FALLBACK_SIGMA, _FALLBACK_W, _FALLBACK_B


def predict(image_path: str) -> float:
    x = extract_features(image_path)
    mu, sigma, w, b = _load_model()
    xn = (x - mu) / sigma
    z = float(xn @ w + b)
    score = 1.0 / (1.0 + np.exp(-z))
    return score


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py some_image.jpg", file=sys.stderr)
        sys.exit(1)
    print(round(predict(sys.argv[1]), 4))
