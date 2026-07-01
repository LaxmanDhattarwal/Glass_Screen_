"""
Feature extraction for "photo vs photo-of-a-screen" (recapture) detection.

Why these features (intuition):

Photographing a screen (phone/laptop/monitor/printout) instead of the real
thing introduces physical artifacts that a real-world photo essentially never
has, because you are photographing a *pixel grid emitting/reflecting light*
through a second lens, rather than a continuous real-world scene:

1. MOIRE / ALIASING (fft_moire_score, radial_peakiness)
   A screen is itself a regular grid of pixels/subpixels. Photographing that
   grid with a camera (which has its own pixel grid + demosaicing) causes
   aliasing: beat-frequency interference patterns (moire) that show up as
   sharp, localized, non-radially-smooth peaks in the 2D Fourier spectrum.
   A real photo's spectrum falls off smoothly with frequency (natural image
   statistics: ~1/f^2 power spectrum). A recapture's spectrum has "bumps"
   riding on top of that smooth falloff.

2. LOCAL TEXTURE PERIODICITY (autocorr_periodicity_score)
   The same grid regularity shows up as a small but real peak in the
   autocorrelation of the high-pass-filtered image at a short, non-zero lag
   (the subpixel/pixel pitch, as imaged). Natural textures don't
   autocorrelate this way at short lags.

3. SPECULAR GLARE / HOTSPOTS (glare_score)
   Screens are self-lit or reflective and are very often photographed at a
   slight angle, producing tell-tale blown-out specular streaks/patches
   (very bright, low-saturation regions) and sometimes a soft vignetted glow
   that real matte objects rarely produce indoors.

4. COLOR CAST / GAMUT (color_cast_score)
   Consumer displays render color via R/G/B subpixels with a color gamut and
   white point that differ subtly from real-world illuminants; recaptures
   often skew slightly blue/green and have compressed color range compared
   to real scenes lit by ambient light.

5. HIGH-FREQUENCY DETAIL ROLLOFF (hf_rolloff_score)
   A screen has finite resolution. Once you zoom the camera in on it, detail
   beyond the screen's own pixel pitch does not exist -- so the image looks
   *sharp up to a point and then unnaturally flat*, unlike a real object
   which usually has continuous fine detail (texture, noise, grain) all the
   way down.

None of these features alone is bulletproof (that's why the brief calls the
clue "subtle") -- so we combine them with a small logistic regression that
gets calibrated on YOUR real/ and screen/ photos (see train.py). If no
calibrated model is found, predict.py falls back to sensible hand-set
weights so the script never crashes / always returns *something* useful.
"""

import numpy as np
import cv2

# Fixed working resolution: big enough to see moire, small enough to be fast.
WORK_SIZE = 512


def _load_gray_and_color(image_path: str):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        # Fall back to PIL for formats/paths OpenCV chokes on (e.g. some
        # iPhone HEIC-derived JPEGs with unusual color profiles).
        from PIL import Image
        pil = Image.open(image_path).convert("RGB")
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    h, w = img.shape[:2]
    scale = WORK_SIZE / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    return img, gray


def _radial_profile(mag: np.ndarray, n_bins: int = 64):
    """Average FFT magnitude over rings of constant radius (radial average)."""
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    r_max = min(cy, cx)
    r_bin = (r / r_max * (n_bins - 1)).astype(np.int32)
    r_bin = np.clip(r_bin, 0, n_bins - 1)

    sums = np.bincount(r_bin.ravel(), weights=mag.ravel(), minlength=n_bins)
    counts = np.bincount(r_bin.ravel(), minlength=n_bins)
    counts[counts == 0] = 1
    return sums / counts


def fft_moire_features(gray: np.ndarray):
    """Returns (peakiness_score, hf_energy_ratio).

    peakiness_score: how "bumpy" the radial FFT profile is relative to a
    smoothed version of itself. Natural photos -> smooth monotonic falloff
    -> low peakiness. Screen recaptures -> extra narrow-band energy from
    the pixel grid -> local bumps -> high peakiness.
    """
    win = np.hanning(gray.shape[0])[:, None] * np.hanning(gray.shape[1])[None, :]
    f = np.fft.fftshift(np.fft.fft2(gray * win))
    mag = np.log1p(np.abs(f))

    profile = _radial_profile(mag)
    # ignore DC / very low freq (bins 0-2): that's overall brightness/contrast,
    # not texture.
    profile = profile[3:]

    # Smooth version of the same profile (expected natural falloff).
    kernel = np.ones(5) / 5
    smooth = np.convolve(profile, kernel, mode="same")
    residual = profile - smooth
    peakiness = float(np.std(residual) / (np.mean(profile) + 1e-6))

    # Energy ratio: how much spectral energy sits in the mid-high band
    # (screen pixel-grid aliasing tends to land here) vs total.
    n = len(profile)
    mid_hi = profile[n // 4:]
    hf_ratio = float(np.sum(mid_hi) / (np.sum(profile) + 1e-6))

    return peakiness, hf_ratio


def autocorr_periodicity_score(gray: np.ndarray):
    """Peak in the autocorrelation of a high-pass version, away from lag 0,
    normalised. High for regular grid-like textures (screens), low for
    natural, non-periodic textures."""
    hp = gray - cv2.GaussianBlur(gray, (0, 0), sigmaX=2.0)
    hp = hp[:256, :256]  # crop for speed; local pattern is what matters
    f = np.fft.fft2(hp)
    power = np.abs(f) ** 2
    ac = np.fft.ifft2(power).real
    ac = np.fft.fftshift(ac)
    ac = ac / (ac.max() + 1e-9)

    h, w = ac.shape
    cy, cx = h // 2, w // 2
    # Look in an annulus a few pixels out from the center (lag 0 is
    # trivially 1.0 and uninformative).
    y, x = np.indices((h, w))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    ring = (r > 3) & (r < 20)
    return float(ac[ring].max())


def glare_score(img_bgr: np.ndarray):
    """Fraction of very bright, low-saturation "blown highlight" pixels,
    typical of screen glare/reflection."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    v = hsv[:, :, 2] / 255.0
    s = hsv[:, :, 1] / 255.0
    hot = (v > 0.92) & (s < 0.25)
    return float(np.mean(hot))


def color_cast_score(img_bgr: np.ndarray):
    """Blue/green skew relative to red, common in emissive-display recaptures."""
    b, g, r = cv2.split(img_bgr.astype(np.float32))
    r_mean, g_mean, b_mean = r.mean(), g.mean(), b.mean()
    denom = (r_mean + g_mean + b_mean) / 3.0 + 1e-6
    return float(((g_mean + b_mean) / 2.0 - r_mean) / denom)


def hf_rolloff_score(gray: np.ndarray):
    """Ratio comparing fine-detail energy (Laplacian at small scale) to
    medium-detail energy (Laplacian at a coarser scale). Screens show a
    sharper "cliff": lots of medium detail, disproportionately little of
    the very finest detail once past the screen's own pixel pitch."""
    lap_fine = cv2.Laplacian(gray, cv2.CV_32F, ksize=1)
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.5)
    lap_med = cv2.Laplacian(blurred, cv2.CV_32F, ksize=1)
    fine_e = float(np.mean(lap_fine ** 2))
    med_e = float(np.mean(lap_med ** 2)) + 1e-6
    return fine_e / med_e


FEATURE_NAMES = [
    "fft_peakiness",
    "fft_hf_ratio",
    "autocorr_peak",
    "glare",
    "color_cast",
    "hf_rolloff",
]


def extract_features(image_path: str) -> np.ndarray:
    img_bgr, gray = _load_gray_and_color(image_path)
    peakiness, hf_ratio = fft_moire_features(gray)
    ac = autocorr_periodicity_score(gray)
    glare = glare_score(img_bgr)
    cast = color_cast_score(img_bgr)
    rolloff = hf_rolloff_score(gray)
    return np.array([peakiness, hf_ratio, ac, glare, cast, rolloff], dtype=np.float64)
