"""backend/python/spoof.py

Anti-spoofing check for attendance.

This version uses an ONNX anti-spoof model (`anti_spoof_model.onnx`) if present.
If the model isn't available or inference fails, it falls back to a lightweight
heuristic check (lenient) so the pipeline doesn't hard-fail.

CLI:
  python spoof.py <image_path>

JSON output (single line):
  {
    "success": true,
    "result": "real"|"fake",
    "confidence": 0.87,
    "details": {...}
  }

Notes:
  - The Node backend expects JSON on stdout.
  - We try hard to never print anything else.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


def _find_model_path() -> Optional[str]:
    """Find model path regardless of current working directory."""
    here = os.path.abspath(os.path.dirname(__file__))
    # repo root is two levels up from backend/python
    repo_root = os.path.abspath(os.path.join(here, "..", ".."))
    candidates = [
        os.path.join(repo_root, "anti_spoof_model.onnx"),
        os.path.join(repo_root, "models", "anti_spoof_model.onnx"),
        os.path.join(here, "anti_spoof_model.onnx"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    exp = np.exp(x)
    return exp / (np.sum(exp) + 1e-12)


def _center_crop_square(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    s = min(h, w)
    y0 = (h - s) // 2
    x0 = (w - s) // 2
    return img[y0 : y0 + s, x0 : x0 + s]


def _preprocess(img_bgr: np.ndarray, size: int) -> np.ndarray:
    """Generic preprocessing: BGR -> RGB, resize, normalize, NCHW float32."""
    img = _center_crop_square(img_bgr)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = img.astype(np.float32) / 255.0
    # common normalization used by many anti-spoof classifiers
    mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    x = (x - mean) / std
    x = np.transpose(x, (2, 0, 1))  # HWC -> CHW
    x = np.expand_dims(x, 0)  # NCHW
    return x


def _infer_onnx(img_bgr: np.ndarray, model_path: str) -> Dict[str, Any]:
    """Run ONNX model and return {label, confidence, raw}.

    We handle a few common output shapes:
      - [1,2] logits (real/fake)
      - [1,1] probability of real or fake (we detect via heuristic)
      - multi-class where max prob is used, and we map index 1->real if possible
    """
    try:
        import onnxruntime as ort  # type: ignore
    except Exception as e:
        raise RuntimeError(f"onnxruntime not available: {e}")

    sess = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"],
    )

    input_meta = sess.get_inputs()[0]
    input_name = input_meta.name
    input_shape = input_meta.shape

    # pick input size from shape if present, fallback to 224
    size = 224
    try:
        # shape like [1,3,224,224] or [None,3,80,80]
        if len(input_shape) == 4 and isinstance(input_shape[2], int):
            size = int(input_shape[2])
    except Exception:
        pass

    x = _preprocess(img_bgr, size)
    outputs = sess.run(None, {input_name: x})
    if not outputs:
        raise RuntimeError("Model returned no outputs")

    y = outputs[0]
    y = np.array(y)
    y_flat = y.reshape(-1).astype(np.float32)

    details: Dict[str, Any] = {
        "model_path": model_path,
        "input_size": size,
        "output_shape": list(y.shape),
    }

    # Case A: logits/probs for 2 classes
    if y_flat.size == 2:
        probs = _softmax(y_flat)
        # assume index 1 = real (common), index 0 = fake
        real_prob = float(probs[1])
        fake_prob = float(probs[0])
        label = "real" if real_prob >= fake_prob else "fake"
        conf = max(real_prob, fake_prob)
        details.update({"real_prob": real_prob, "fake_prob": fake_prob})
        return {"label": label, "confidence": float(conf), "details": details}

    # Case B: single scalar output
    if y_flat.size == 1:
        p = float(y_flat[0])
        # if output already looks like probability
        if 0.0 <= p <= 1.0:
            # assume p = real probability by default
            label = "real" if p >= 0.5 else "fake"
            return {"label": label, "confidence": float(p if label == "real" else 1.0 - p), "details": {**details, "p": p}}
        # otherwise treat as logit
        p_real = float(1.0 / (1.0 + np.exp(-p)))
        label = "real" if p_real >= 0.5 else "fake"
        return {"label": label, "confidence": float(p_real if label == "real" else 1.0 - p_real), "details": {**details, "logit": p}}

    # Case C: multi-class
    probs = _softmax(y_flat)
    idx = int(np.argmax(probs))
    conf = float(probs[idx])
    # best-effort mapping: if 2+ classes, treat class 1 as real if conf high, else unknown
    label = "real" if idx == 1 else "fake"
    details.update({"top_index": idx})
    return {"label": label, "confidence": conf, "details": details}


def _heuristic_fallback(img_bgr: np.ndarray) -> Dict[str, Any]:
    """Lenient fallback: only flags obvious replays (big rectangular border).

    This is intentionally conservative to reduce false positives.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape[:2]
    big_rects = 0
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            area = cv2.contourArea(c)
            if area > (h * w) * 0.15:
                big_rects += 1

    if big_rects >= 1:
        return {"label": "fake", "confidence": 0.65, "details": {"fallback": True, "big_rects": big_rects}}
    return {"label": "real", "confidence": 0.75, "details": {"fallback": True, "big_rects": big_rects}}


def detect_spoof(image_path: str) -> Dict[str, Any]:
    if not os.path.exists(image_path):
        return {"success": False, "error": f"Image not found: {image_path}"}

    img = cv2.imread(image_path)
    if img is None:
        return {"success": False, "error": f"Failed to load image: {image_path}"}

    model_path = _find_model_path()

    try:
        if model_path:
            out = _infer_onnx(img, model_path)
        else:
            out = _heuristic_fallback(img)

        return {
            "success": True,
            "result": out["label"],
            "confidence": round(float(out["confidence"]), 4),
            "details": out.get("details", {}),
        }
    except Exception as e:
        # fall back rather than hard failing the pipeline
        out = _heuristic_fallback(img)
        return {
            "success": True,
            "result": out["label"],
            "confidence": round(float(out["confidence"]), 4),
            "details": {**out.get("details", {}), "onnx_error": str(e)},
        }


def _main() -> int:
    if len(sys.argv) < 2:
        sys.stdout.write(json.dumps({"success": False, "error": "Usage: python spoof.py <image_path>"}))
        sys.stdout.write("\n")
        return 1

    image_path = sys.argv[1]
    result = detect_spoof(image_path)
    sys.stdout.write(json.dumps(result))
    sys.stdout.write("\n")
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(_main())
"""
spoof.py - Enhanced Face Anti-Spoofing Detection

Detects whether a face image is real (live person) or fake
(photo on phone screen, printed photo, screen replay, mask).

Detection methods:
1. Moiré pattern detection (FFT periodic spike analysis)
2. Screen/phone reflection & glare detection
3. Color channel analysis (blue shift from screens)
4. Noise pattern analysis (sensor vs screen noise)
5. Gradient uniformity (flat backgrounds = screen border)
6. Screen border / bezel detection
7. Texture micro-pattern analysis (LBP)
8. Specular highlight analysis
9. Edge sharpness double-capture analysis

Usage:
    python spoof.py <image_path>

Output (JSON):
    {
        "success": true,
        "result": "real" | "fake",
        "confidence": 0.85,
        "details": {...}
    }
"""

import sys
import os
import json
import numpy as np
import cv2


# ───────────────────────────────────────────────────────
# 1. Moiré Pattern Detection (FFT periodic spikes)
# ───────────────────────────────────────────────────────
def detect_moire(gray):
    """
    Phone/monitor screens produce periodic moiré interference
    patterns that show up as distinct spikes in the FFT spectrum.
    """
    f = np.fft.fft2(gray.astype(np.float64))
    fshift = np.fft.fftshift(f)
    mag = np.log1p(np.abs(fshift))

    h, w = mag.shape
    cy, cx = h // 2, w // 2

    # Zero out the DC component and very-low freq
    r_dc = max(min(h, w) // 16, 4)
    cv2.circle(mag, (cx, cy), r_dc, 0, -1)

    # Look for abnormally bright peaks (spikes)
    mean_val = np.mean(mag)
    std_val = np.std(mag)
    threshold = mean_val + 3.0 * std_val
    spike_count = int(np.sum(mag > threshold))

    # Radial energy distribution — screens have energy at specific radii
    radii = np.sqrt((np.arange(h)[:, None] - cy) ** 2 +
                    (np.arange(w)[None, :] - cx) ** 2)
    max_r = int(min(h, w) // 2)
    radial_profile = np.zeros(max_r)
    for r in range(max_r):
        mask = (radii >= r) & (radii < r + 1)
        if mask.any():
            radial_profile[r] = np.mean(mag[mask])

    # Peaks in radial profile → periodic pattern
    if len(radial_profile) > 10:
        rp = radial_profile[5:]  # skip DC neighbourhood
        rp_diff = np.diff(rp)
        sign_changes = np.sum(np.diff(np.sign(rp_diff)) != 0)
        periodicity_score = sign_changes / len(rp_diff)
    else:
        periodicity_score = 0.0

    return {
        "spike_count": spike_count,
        "periodicity": round(float(periodicity_score), 4),
        "mean_mag": round(float(mean_val), 4),
        "std_mag": round(float(std_val), 4),
    }


# ───────────────────────────────────────────────────────
# 2. Screen Reflection / Glare Detection
# ───────────────────────────────────────────────────────
def detect_reflection(gray):
    """
    Phone screens and monitors create specular highlights /
    glare patches that are over-exposed (near 255).
    """
    # Count very bright pixels
    bright_thresh = 240
    very_bright = np.sum(gray >= bright_thresh)
    total = gray.size
    bright_ratio = very_bright / total

    # Look for concentrated bright blobs (reflections)
    _, binary = cv2.threshold(gray, bright_thresh, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    large_blobs = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area > total * 0.002:  # blob > 0.2% of image
            large_blobs += 1

    return {
        "bright_ratio": round(float(bright_ratio), 6),
        "large_bright_blobs": large_blobs,
    }


# ───────────────────────────────────────────────────────
# 3. Color Channel Analysis (blue shift from screens)
# ───────────────────────────────────────────────────────
def analyze_color_channels(image):
    """
    Screens emit more blue light. Captured-from-screen images
    have a measurable blue channel shift compared to real faces.
    Also check saturation — screens often over-saturate or under-saturate.
    """
    b, g, r = cv2.split(image.astype(np.float64))
    total = b + g + r + 1e-7

    blue_ratio = float(np.mean(b / total))
    green_ratio = float(np.mean(g / total))
    red_ratio = float(np.mean(r / total))

    # Blue dominance score
    blue_dominance = blue_ratio - (red_ratio + green_ratio) / 2.0

    # Saturation analysis
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float64)
    sat_mean = float(np.mean(sat))
    sat_std = float(np.std(sat))

    # Value (brightness) uniformity — screens have more uniform brightness
    val = hsv[:, :, 2].astype(np.float64)
    val_std = float(np.std(val))

    return {
        "blue_ratio": round(blue_ratio, 5),
        "green_ratio": round(green_ratio, 5),
        "red_ratio": round(red_ratio, 5),
        "blue_dominance": round(blue_dominance, 5),
        "sat_mean": round(sat_mean, 2),
        "sat_std": round(sat_std, 2),
        "val_std": round(val_std, 2),
    }


# ───────────────────────────────────────────────────────
# 4. Noise Pattern Analysis
# ───────────────────────────────────────────────────────
def analyze_noise(gray):
    """
    Real webcam images have natural sensor noise.
    Captured-from-screen images have different noise characteristics:
    quantization artifacts, compression artifacts, re-sampling noise.
    """
    # Denoise and compute residual
    denoised = cv2.GaussianBlur(gray, (5, 5), 0)
    noise = gray.astype(np.float64) - denoised.astype(np.float64)

    noise_std = float(np.std(noise))
    noise_mean = float(np.mean(np.abs(noise)))

    # Kurtosis of noise — real sensor noise is roughly Gaussian (kurtosis ≈ 3)
    # Screen re-capture noise is more uniform/peaked
    noise_flat = noise.flatten()
    n = len(noise_flat)
    mean_n = np.mean(noise_flat)
    std_n = np.std(noise_flat) + 1e-7
    kurtosis = float(np.mean(((noise_flat - mean_n) / std_n) ** 4))

    # Block-wise noise variance — JPEG artifacts create blocky noise patterns
    block_size = 8
    h, w = gray.shape
    block_vars = []
    for i in range(0, h - block_size, block_size):
        for j in range(0, w - block_size, block_size):
            block = noise[i:i + block_size, j:j + block_size]
            block_vars.append(np.var(block))
    
    block_var_std = float(np.std(block_vars)) if block_vars else 0.0
    block_var_mean = float(np.mean(block_vars)) if block_vars else 0.0

    return {
        "noise_std": round(noise_std, 4),
        "noise_mean": round(noise_mean, 4),
        "noise_kurtosis": round(kurtosis, 4),
        "block_var_std": round(block_var_std, 4),
        "block_var_mean": round(block_var_mean, 4),
    }


# ───────────────────────────────────────────────────────
# 5. Gradient Uniformity Analysis
# ───────────────────────────────────────────────────────
def analyze_gradients(gray):
    """
    Real faces have organic gradient distributions.
    Screen photos have more uniform gradient regions (phone bezels,
    flat backgrounds on screen).
    """
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(sobelx ** 2 + sobely ** 2)

    grad_mean = float(np.mean(gradient_mag))
    grad_std = float(np.std(gradient_mag))

    # Ratio of very-low-gradient pixels (flat areas)
    flat_thresh = 5.0
    flat_ratio = float(np.sum(gradient_mag < flat_thresh) / gradient_mag.size)

    # Edge density
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.sum(edges > 0) / edges.size)

    return {
        "grad_mean": round(grad_mean, 4),
        "grad_std": round(grad_std, 4),
        "flat_ratio": round(flat_ratio, 4),
        "edge_density": round(edge_density, 4),
    }


# ───────────────────────────────────────────────────────
# 6. Screen Border / Bezel Detection
# ───────────────────────────────────────────────────────
def detect_screen_border(gray):
    """
    When someone holds a phone up to the webcam, the phone's bezel
    or the edge of the screen often creates strong rectangular edges.
    """
    edges = cv2.Canny(gray, 50, 150)

    # Hough line detection for straight lines (screen edges)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                            minLineLength=60, maxLineGap=10)

    line_count = 0
    long_lines = 0
    h, w = gray.shape

    if lines is not None:
        line_count = len(lines)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if length > min(h, w) * 0.3:
                long_lines += 1

    # Check for rectangular contours (phone outline)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rect_score = 0
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            area = cv2.contourArea(c)
            if area > (h * w) * 0.1:
                rect_score += 1

    return {
        "line_count": line_count,
        "long_lines": long_lines,
        "rect_contours": rect_score,
    }


# ───────────────────────────────────────────────────────
# 7. Texture Micro-Pattern (LBP) Analysis
# ───────────────────────────────────────────────────────
def compute_lbp(image, radius=1, n_points=8):
    """
    Compute Local Binary Pattern for texture analysis.
    Real faces have richer micro-texture than screen replays.
    """
    h, w = image.shape
    lbp = np.zeros((h - 2 * radius, w - 2 * radius), dtype=np.uint8)

    for k in range(n_points):
        angle = 2.0 * np.pi * k / n_points
        dx = int(round(radius * np.cos(angle)))
        dy = int(round(-radius * np.sin(angle)))
        shifted = image[radius + dy:h - radius + dy,
                        radius + dx:w - radius + dx]
        center = image[radius:h - radius, radius:w - radius]
        lbp |= ((shifted >= center).astype(np.uint8) << k)

    hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
    hist = hist.astype(np.float64)
    hist /= (hist.sum() + 1e-7)
    return hist


def analyze_texture(gray):
    """Texture entropy from LBP at multiple scales."""
    results = {}
    total_entropy = 0.0
    for radius in [1, 2, 3]:
        hist = compute_lbp(gray, radius=radius, n_points=8)
        entropy = float(-np.sum(hist * np.log2(hist + 1e-7)))
        results[f"lbp_r{radius}_entropy"] = round(entropy, 4)
        total_entropy += entropy

    results["avg_entropy"] = round(total_entropy / 3.0, 4)

    # Uniform pattern ratio — real faces have more uniform LBP patterns
    hist = compute_lbp(gray, radius=1, n_points=8)
    # Uniform patterns have at most 2 transitions 0→1 or 1→0
    uniform_bins = []
    for val in range(256):
        bits = format(val, '08b')
        transitions = sum(1 for i in range(len(bits)) if bits[i] != bits[(i + 1) % len(bits)])
        if transitions <= 2:
            uniform_bins.append(val)
    uniform_ratio = float(np.sum(hist[uniform_bins]))
    results["uniform_ratio"] = round(uniform_ratio, 4)

    return results


# ───────────────────────────────────────────────────────
# 8. Specular Highlight / Skin Shine Analysis
# ───────────────────────────────────────────────────────
def analyze_specular(gray):
    """
    Real faces have small specular highlights (forehead, nose, cheeks).
    Screen captures have different highlight distributions (screen glare).
    """
    # Find highlights
    thresh = 220
    highlights = (gray >= thresh).astype(np.uint8) * 255
    contours, _ = cv2.findContours(highlights, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    small_highlights = 0
    medium_highlights = 0
    large_highlights = 0
    total = gray.size

    for c in contours:
        area = cv2.contourArea(c)
        ratio = area / total
        if ratio < 0.001:
            small_highlights += 1
        elif ratio < 0.01:
            medium_highlights += 1
        else:
            large_highlights += 1

    return {
        "small_highlights": small_highlights,
        "medium_highlights": medium_highlights,
        "large_highlights": large_highlights,
    }


# ───────────────────────────────────────────────────────
# 9. Double-Capture Sharpness Analysis
# ───────────────────────────────────────────────────────
def analyze_sharpness(gray):
    """
    Double-captured images (webcam → screen → webcam) lose sharpness
    in a characteristic way. Analyse at multiple scales.
    """
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Multi-scale sharpness
    scales = [1.0, 0.5, 0.25]
    sharpness_at_scale = []
    for s in scales:
        if s < 1.0:
            h, w = gray.shape
            resized = cv2.resize(gray, (int(w * s), int(h * s)))
        else:
            resized = gray
        lap = cv2.Laplacian(resized, cv2.CV_64F)
        sharpness_at_scale.append(float(lap.var()))

    # Sharpness drop ratio across scales
    if sharpness_at_scale[0] > 0:
        sharpness_drop = sharpness_at_scale[-1] / sharpness_at_scale[0]
    else:
        sharpness_drop = 0.0

    return {
        "laplacian_var": round(laplacian_var, 4),
        "sharpness_scales": [round(s, 4) for s in sharpness_at_scale],
        "sharpness_drop": round(sharpness_drop, 4),
    }


# ═══════════════════════════════════════════════════════
# MAIN SPOOF DETECTION — WEIGHTED SCORING
# ═══════════════════════════════════════════════════════
def detect_spoof(image_path):
    """
    Multi-factor anti-spoofing analysis.
    Higher fake_score = more likely to be fake.
    Threshold: fake_score >= 3.0 out of ~10 → FAKE
    """
    if not os.path.exists(image_path):
        return {"success": False, "error": f"Image not found: {image_path}"}

    image = cv2.imread(image_path)
    if image is None:
        return {"success": False, "error": f"Failed to load image: {image_path}"}

    # Resize for consistency
    image = cv2.resize(image, (256, 256))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Run all analyses
    moire = detect_moire(gray)
    reflection = detect_reflection(gray)
    color = analyze_color_channels(image)
    noise = analyze_noise(gray)
    gradients = analyze_gradients(gray)
    borders = detect_screen_border(gray)
    texture = analyze_texture(gray)
    specular = analyze_specular(gray)
    sharpness = analyze_sharpness(gray)

    # ──────── SCORING ────────
    # Calibrated for typical laptop webcams (720p-1080p, JPEG compression).
    # Webcam images naturally have: high noise kurtosis (~15-50),
    # low noise std (~1-2), moderate spike_count (~100-250),
    # low sharpness on aligned face crops (~5-30).
    # Thresholds are set so real webcam faces score < 3.5
    # and phone-photo spoofs score > 4.5.
    fake_score = 0.0
    checks = {}

    # 1. Moiré patterns (weight: 2.0) — strongest indicator of screen
    #    Real webcam: spike_count ~100-250; Phone photo of screen: 800+
    if moire["spike_count"] > 1000:
        fake_score += 2.0
        checks["moire"] = "strong_fail"
    elif moire["spike_count"] > 600:
        fake_score += 1.5
        checks["moire"] = "fail"
    elif moire["periodicity"] > 0.75:
        fake_score += 1.0
        checks["moire"] = "periodic_fail"
    else:
        checks["moire"] = "pass"

    # 2. Reflection / glare (weight: 1.5)
    if reflection["large_bright_blobs"] >= 3:
        fake_score += 1.5
        checks["reflection"] = "fail"
    elif reflection["bright_ratio"] > 0.08:
        fake_score += 1.0
        checks["reflection"] = "marginal_fail"
    elif reflection["bright_ratio"] > 0.04:
        fake_score += 0.5
        checks["reflection"] = "marginal"
    else:
        checks["reflection"] = "pass"

    # 3. Blue channel shift (weight: 1.5) — screens emit blue
    #    Real faces have negative blue_dominance (skin is warm).
    #    Screen photos push it towards 0 or positive.
    if color["blue_dominance"] > 0.03:
        fake_score += 1.5
        checks["blue_shift"] = "fail"
    elif color["blue_dominance"] > 0.01:
        fake_score += 0.75
        checks["blue_shift"] = "marginal"
    else:
        checks["blue_shift"] = "pass"

    # 4. Brightness uniformity — screens have unnaturally uniform brightness
    #    Real webcam val_std is typically 35-80. Screens: < 25.
    if color["val_std"] < 20:
        fake_score += 1.0
        checks["brightness_uniformity"] = "fail"
    elif color["val_std"] < 30:
        fake_score += 0.5
        checks["brightness_uniformity"] = "marginal"
    else:
        checks["brightness_uniformity"] = "pass"

    # 5. Noise characteristics (weight: 1.0)
    #    Webcam JPEG naturally produces high kurtosis (15-50) and low std (~1-2).
    #    Only flag when BOTH are extreme AND combined with other signals.
    #    Screen re-capture: kurtosis > 60 with noise_std < 0.8
    if noise["noise_kurtosis"] > 60.0 and noise["noise_std"] < 0.8:
        fake_score += 1.0
        checks["noise"] = "fail"
    elif noise["noise_kurtosis"] > 40.0 and noise["noise_std"] < 1.0:
        fake_score += 0.5
        checks["noise"] = "marginal"
    else:
        checks["noise"] = "pass"

    # 6. Gradient / flat regions (weight: 1.0)
    if gradients["flat_ratio"] > 0.75:
        fake_score += 1.0
        checks["gradients"] = "fail"
    elif gradients["flat_ratio"] > 0.65:
        fake_score += 0.5
        checks["gradients"] = "marginal"
    else:
        checks["gradients"] = "pass"

    # 7. Screen borders / bezels (weight: 2.0) — very strong signal
    #    Phone bezels create large rectangles and long straight lines.
    if borders["rect_contours"] >= 1 and borders["long_lines"] >= 2:
        fake_score += 2.0
        checks["borders"] = "rect_fail"
    elif borders["rect_contours"] >= 1:
        fake_score += 1.5
        checks["borders"] = "rect_marginal"
    elif borders["long_lines"] >= 4:
        fake_score += 1.0
        checks["borders"] = "line_fail"
    elif borders["long_lines"] >= 2:
        fake_score += 0.5
        checks["borders"] = "marginal"
    else:
        checks["borders"] = "pass"

    # 8. Texture entropy (weight: 1.0) — screen replays lose micro-texture
    if texture["avg_entropy"] < 3.5:
        fake_score += 1.0
        checks["texture"] = "fail"
    elif texture["avg_entropy"] < 4.5:
        fake_score += 0.5
        checks["texture"] = "marginal"
    else:
        checks["texture"] = "pass"

    # 9. Sharpness — only counts if extremely low (< 5)
    #    Webcam aligned crops are naturally 8-30, so don't penalize those.
    if sharpness["laplacian_var"] < 5:
        fake_score += 0.5
        checks["sharpness"] = "fail"
    else:
        checks["sharpness"] = "pass"

    # ──────── FINAL DECISION ────────
    # Threshold raised to 4.0 to avoid false positives on real webcam faces.
    # Real webcam faces typically score 0-2.5.
    # Phone photos of faces (spoof) typically score 4.5+.
    FAKE_THRESHOLD = 4.0
    is_fake = fake_score >= FAKE_THRESHOLD

    if is_fake:
        confidence = min(fake_score / 10.0, 1.0)
        result_label = "fake"
    else:
        confidence = 1.0 - (fake_score / FAKE_THRESHOLD)
        result_label = "real"

    result = {
        "success": True,
        "result": result_label,
        "confidence": round(float(confidence), 4),
        "details": {
            "fake_score": round(float(fake_score), 2),
            "threshold": FAKE_THRESHOLD,
            "checks": checks,
            "moire": moire,
            "reflection": reflection,
            "color": {k: color[k] for k in ["blue_dominance", "sat_mean", "val_std"]},
            "noise": {k: noise[k] for k in ["noise_std", "noise_kurtosis"]},
            "texture_entropy": texture["avg_entropy"],
            "sharpness": sharpness["laplacian_var"],
            "borders": borders,
        },
    }

    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"success": False, "error": "Usage: python spoof.py <image_path>"}))
        sys.exit(1)

    image_path = sys.argv[1]
    result = detect_spoof(image_path)
    print(json.dumps(result))

    if not result["success"]:
        sys.exit(1)
