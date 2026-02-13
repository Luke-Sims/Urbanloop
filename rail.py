import math

import cv2
import numpy as np


# ----------------------------
# Helpers
# ----------------------------
def ema(prev, new, alpha=0.2):
    if prev is None:
        return new
    return prev * (1 - alpha) + new * alpha


def draw_line_from_mb(img, m, b, y1, y2, color=(0, 255, 0), thickness=3):
    if m is None or abs(m) < 1e-6:
        return
    x1 = int((y1 - b) / m)
    x2 = int((y2 - b) / m)
    cv2.line(img, (x1, int(y1)), (x2, int(y2)), color, thickness)


def fit_weighted_line(segments):
    if not segments:
        return None

    xs, ys, ws = [], [], []
    for x1, y1, x2, y2, w in segments:
        xs += [x1, x2]
        ys += [y1, y2]
        ws += [w, w]

    X = np.array(xs, dtype=np.float32)
    Y = np.array(ys, dtype=np.float32)
    W = np.array(ws, dtype=np.float32)

    Sw = np.sum(W)
    Sx = np.sum(W * X)
    Sy = np.sum(W * Y)
    Sxx = np.sum(W * X * X)
    Sxy = np.sum(W * X * Y)

    denom = Sw * Sxx - Sx * Sx
    if abs(denom) < 1e-6:
        return None

    m = (Sw * Sxy - Sx * Sy) / denom
    b = (Sy - m * Sx) / Sw
    return float(m), float(b)


def x_at_y(x1, y1, x2, y2, y):
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0:
        return None
    m = dy / dx
    b = y1 - m * x1
    if abs(m) < 1e-6:
        return None
    x = (y - b) / m
    return x, m, b


# ----------------------------
# Main detection
# ----------------------------
def detect_rails(
    frame_bgr,
    debug=False,
    canny_low=60,
    canny_high=140,
    roi_top_ratio=0.65,
    min_len=60,
    max_gap=25,
    slope_min=0.35,
    split_x_ratio=0.5,
):
    h, w = frame_bgr.shape[:2]
    overlay = frame_bgr.copy()

    # ROI bas de l'image
    y0 = int(h * roi_top_ratio)
    roi = frame_bgr[y0:h, :]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, canny_low, canny_high)

    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)

    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=70, minLineLength=min_len, maxLineGap=max_gap
    )

    cx_center = w * 0.5
    y_top = y0
    y_bottom = h - 1

    # Point de fuite
    vp_min = w * 0.488
    vp_max = w * 0.503
    max_vp_dist = w * 0.22

    # Bande bas
    bottom_min = w * 0.45
    bottom_max = w * 0.54
    if bottom_min > bottom_max:
        bottom_min, bottom_max = bottom_max, bottom_min

    min_abs_slope = max(slope_min, 0.45)
    split_x = w * split_x_ratio

    candidates = []

    if lines is not None:
        for x1, y1r, x2, y2r in lines[:, 0]:
            y1 = y1r + y0
            y2 = y2r + y0

            length = math.hypot(x2 - x1, y2 - y1)
            if length < min_len:
                continue

            out_bot = x_at_y(x1, y1, x2, y2, y_bottom)
            if out_bot is None:
                continue
            x_bot, m, b = out_bot

            if abs(m) < min_abs_slope:
                continue

            out_top = x_at_y(x1, y1, x2, y2, y_top)
            if out_top is None:
                continue
            x_top, _, _ = out_top

            if not (vp_min <= x_top <= vp_max):
                continue
            if abs(x_top - cx_center) > max_vp_dist:
                continue

            if not (bottom_min <= x_bot <= bottom_max):
                continue

            if not (-0.2 * w <= x_bot <= 1.2 * w):
                continue

            candidates.append((x1, y1, x2, y2, length, x_bot, m))

    left_segments, right_segments = [], []
    for x1, y1, x2, y2, length, x_bot, m in candidates:
        if x_bot < split_x:
            left_segments.append((x1, y1, x2, y2, length))
        else:
            right_segments.append((x1, y1, x2, y2, length))

    # ✅ GARDER SEULEMENT LES 2 PLUS LONGS PAR CÔTÉ
    left_segments = sorted(left_segments, key=lambda s: s[4], reverse=True)[:2]
    right_segments = sorted(right_segments, key=lambda s: s[4], reverse=True)[:2]

    if debug:
        cv2.rectangle(overlay, (0, y0), (w - 1, h - 1), (255, 255, 255), 1)

        cv2.line(
            overlay,
            (int(bottom_min), y_bottom),
            (int(bottom_min), y_bottom - 40),
            (255, 255, 255),
            2,
        )
        cv2.line(
            overlay,
            (int(bottom_max), y_bottom),
            (int(bottom_max), y_bottom - 40),
            (255, 255, 255),
            2,
        )

        cv2.line(
            overlay, (int(vp_min), y_top), (int(vp_min), y_top + 25), (255, 255, 255), 2
        )
        cv2.line(
            overlay, (int(vp_max), y_top), (int(vp_max), y_top + 25), (255, 255, 255), 2
        )

        cv2.line(
            overlay,
            (int(split_x), y_bottom),
            (int(split_x), y_bottom - 60),
            (255, 255, 255),
            2,
        )

        for x1, y1, x2, y2, _ in left_segments:
            cv2.line(overlay, (x1, y1), (x2, y2), (255, 0, 0), 3)
        for x1, y1, x2, y2, _ in right_segments:
            cv2.line(overlay, (x1, y1), (x2, y2), (0, 0, 255), 3)

        # Debug counts pour comprendre le “rien détecté”
        nb_lines = 0 if lines is None else len(lines)
        cv2.putText(
            overlay,
            f"lines={nb_lines} cand={len(candidates)} L={len(left_segments)} R={len(right_segments)}",
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

    return overlay, edges


def main():
    source = "videoRgb.avi"
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la source vidéo : {source}")

    force_rotate_180 = True

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if force_rotate_180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)

        overlay, edges = detect_rails(frame, debug=True, split_x_ratio=0.494)

        cv2.imshow("Rails detection (segments)", overlay)
        cv2.imshow("Edges (ROI)", edges)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
