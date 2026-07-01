"""
Calibrate the recapture detector on your own data.

Usage:
    python train.py            # expects ./real/*.jpg and ./screen/*.jpg
    python train.py --real path/to/real --screen path/to/screen

What it does:
1. Extracts the 6 hand-designed features (features.py) for every image.
2. Standardizes them (zero mean / unit variance) and fits a tiny logistic
   regression by gradient descent (no sklearn dependency, so predict.py
   stays lightweight at inference time).
3. Saves weights + standardization stats to model.json.
4. Reports train accuracy (and, if you pass --holdout, a proper held-out
   accuracy) so you have an honest number for your note.

Kept deliberately simple: 6 features -> 1 logistic unit. That's the whole
"model" -- a handful of floats in a JSON file, <1ms to apply.
"""

import argparse
import glob
import json
import os
import time

import numpy as np

from features import extract_features, FEATURE_NAMES

IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG", "*.heic", "*.HEIC")


def list_images(folder):
    files = []
    for ext in IMG_EXTS:
        files.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(set(files))


def build_dataset(real_dir, screen_dir):
    X, y, paths = [], [], []
    for p in list_images(real_dir):
        try:
            X.append(extract_features(p)); y.append(0); paths.append(p)
        except Exception as e:
            print(f"  skip {p}: {e}")
    for p in list_images(screen_dir):
        try:
            X.append(extract_features(p)); y.append(1); paths.append(p)
        except Exception as e:
            print(f"  skip {p}: {e}")
    return np.array(X), np.array(y), paths


def train_logreg(X, y, epochs=3000, lr=0.2, l2=1e-3):
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    for _ in range(epochs):
        z = X @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        grad_w = X.T @ (p - y) / n + l2 * w
        grad_b = np.mean(p - y)
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default="real")
    ap.add_argument("--screen", default="screen")
    ap.add_argument("--holdout", type=float, default=0.2,
                     help="fraction of data held out for an honest accuracy estimate")
    ap.add_argument("--out", default="model.json")
    args = ap.parse_args()

    print(f"Scanning {args.real}/ and {args.screen}/ ...")
    X, y, paths = build_dataset(args.real, args.screen)
    n_real = int(np.sum(y == 0))
    n_screen = int(np.sum(y == 1))
    print(f"Found {n_real} real photos, {n_screen} screen photos.")
    if n_real < 5 or n_screen < 5:
        print("Not enough data to calibrate (need at least ~5 of each, ideally ~50). "
              "predict.py will fall back to default heuristic weights.")
        return

    rng = np.random.default_rng(0)
    idx = rng.permutation(len(y))
    X, y, paths = X[idx], y[idx], [paths[i] for i in idx]

    n_hold = max(1, int(len(y) * args.holdout)) if args.holdout > 0 else 0
    X_hold, y_hold = X[:n_hold], y[:n_hold]
    X_train, y_train = X[n_hold:], y[n_hold:]

    mu, sigma = X_train.mean(axis=0), X_train.std(axis=0) + 1e-6
    Xn_train = (X_train - mu) / sigma

    t0 = time.time()
    w, b = train_logreg(Xn_train, y_train)
    print(f"Trained in {time.time() - t0:.2f}s on {len(y_train)} images.")

    def accuracy(Xd, yd):
        if len(yd) == 0:
            return None
        Xn = (Xd - mu) / sigma
        p = 1.0 / (1.0 + np.exp(-(Xn @ w + b)))
        pred = (p >= 0.5).astype(int)
        return float(np.mean(pred == yd))

    train_acc = accuracy(X_train, y_train)
    hold_acc = accuracy(X_hold, y_hold)
    print(f"Train accuracy: {train_acc:.3f}")
    if hold_acc is not None:
        print(f"Held-out accuracy ({n_hold} images): {hold_acc:.3f}  <-- use this in your note")

    model = {
        "feature_names": FEATURE_NAMES,
        "mu": mu.tolist(),
        "sigma": sigma.tolist(),
        "w": w.tolist(),
        "b": float(b),
        "train_accuracy": train_acc,
        "holdout_accuracy": hold_acc,
        "n_real": n_real,
        "n_screen": n_screen,
    }
    with open(args.out, "w") as f:
        json.dump(model, f, indent=2)
    print(f"Saved calibrated model to {args.out}")


if __name__ == "__main__":
    main()
