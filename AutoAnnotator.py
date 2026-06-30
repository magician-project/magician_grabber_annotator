#!/usr/bin/env python3
"""
AutoAnnotator.py — semi-automatic defect annotation helpers for wxAnnotator.

Why this exists
---------------
Direct defect detection by SAM3 / VLMs fails on MAGICIAN polarisation imagery
(D3.3 §2.4.2): those nets are trained on natural-image statistics and are blind to
the subtle polarimetric gradients that define a defect. BUT human operators draw a
prominent PEN MARKING (a circle / scribble) around every defect on the metal sheet.
That ink is high-contrast and in-distribution for SAM3, so we never ask the net to
find the defect — we ask it to find the *mark*, and infer the defect from it.

v1 task: PROPAGATE an annotation from frame N to frame N+1.
The pen mark is physical ink, so it is robust to the per-frame lighting/polarisation
changes that break naive optical flow. We segment the mark on both frames, match the
blob, and carry the annotated point (keeping its offset inside the mark) to N+1.

Standalone test:
    python3 AutoAnnotator.py prevFrame.png nextFrame.png --x 612 --y 512 \
            --ip 139.91.185.16 --port 7860 --prompt "pen mark" --debug ./dbg
"""

import os
import tempfile

import cv2
import numpy as np

from readData import readPolarPNMToRGBALive


# --- Server / behaviour defaults (edit here while experimenting) --------------
SAM3_IP      = "127.0.0.1"   # remote server: "139.91.185.16" (e.g. via SSH tunnel)
SAM3_PORT    = "7860"
DEFAULT_PROMPT = "drawn circle"  # what SAM3 actually grounds on; try also "circle"
REPRESENTATION = "clahe"        # "gray"  = averaged polarisation (natural-looking)
                                # "clahe" = CLAHE-enhanced gray (faint marks pop)
                                # "rgb"   = 0/45/90 deg -> B/G/R false colour
MIN_BLOB_AREA  = 30             # tracking: keep small blobs (propagation is seeded)
DETECT_MIN_AREA = 3000          # from-scratch detect: real pen rings are ~19k–96k px;
                                # spurious specks are <250 px, so this cleanly rejects them
ASSOC_MAX_DIST = 250            # render px: if no mark within this of the point, skip

# pointClicks in the dataset JSON are in FULL-MOSAIC coords (the annotator repacks the
# 4-channel PNG to a 2x mosaic before computing click ratios). Our renders are the
# debayered half-resolution image, so mosaic coords are MOSAIC_SCALE x render coords.
MOSAIC_SCALE = 2.0

# --- DoLP defect-refinement (nudge the point from the ring centroid onto the defect) ---
# The pen ring's centroid is offset from the true defect; inside the ring the defect shows
# as a Degree-of-Linear-Polarisation anomaly (negative dent -> LOW DoLP, positive -> HIGH).
# Params validated on ~56 hand-GT frames: 15/56 improved, 0 regressions (conservative).
REFINE_DEFECT     = True   # apply DoLP refinement after detecting the ring
REFINE_K          = 3.0    # anomaly threshold in robust-MAD units (higher = more cautious)
REFINE_MAX_MOVE   = 0.5    # max move from centroid as a fraction of the ring radius
REFINE_MIN_AREA   = 6      # min anomaly-cluster area (render px)
REFINE_RING_DILATE = 21    # dilation (px) used to exclude the ink ring from the interior
# -----------------------------------------------------------------------------


def dolp_from_raw(raw):
    """Degree of Linear Polarisation map (debayer res, [0,1]) from a raw 4-channel frame.
    Matches visualizeData.py: S0=p0+p90, S1=p0-p90, S2=p45-p135, DoLP=|S|/S0."""
    rgba = readPolarPNMToRGBALive(raw).astype(np.float32)
    p0, p45, p90, p135 = rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2], rgba[:, :, 3]
    S0 = p0 + p90
    S1 = p0 - p90
    S2 = p45 - p135
    return np.clip(np.sqrt(S1 * S1 + S2 * S2) / (S0 + 1e-6), 0.0, 1.0)


def polarity_for_class(defect_class):
    """Which DoLP direction marks the defect for a given class label.
    Positive dents bulge out -> HIGH DoLP; everything else (negative dents, …) -> LOW."""
    return "high" if "positive" in str(defect_class).lower() else "low"


def refine_defect_in_ring(dolp, ring_mask, cx, cy, polarity="low",
                          k=REFINE_K, max_move_frac=REFINE_MAX_MOVE,
                          min_area=REFINE_MIN_AREA, ring_dilate=REFINE_RING_DILATE):
    """Move (cx, cy) from the ring centroid onto the DoLP anomaly inside the ring.

    dolp       : full-frame DoLP map (render res).
    ring_mask  : binary mask of THIS ring's connected component (render res).
    Returns (rx, ry) in render coords — the centroid unchanged if no confident anomaly.
    """
    ys, xs = np.where(ring_mask > 0)
    if len(xs) < 3:
        return cx, cy
    # interior = inside the ring (convex hull) minus the ink itself
    hull = cv2.convexHull(np.column_stack((xs, ys)))
    disk = np.zeros_like(ring_mask)
    cv2.fillConvexPoly(disk, hull, 255)
    ring = cv2.dilate(ring_mask, np.ones((ring_dilate, ring_dilate), np.uint8))
    interior = cv2.bitwise_and(disk, cv2.bitwise_not(ring))
    if interior.sum() == 0:
        return cx, cy

    dv = dolp[interior > 0]
    med = np.median(dv)
    mad = np.median(np.abs(dv - med)) + 1e-6
    score = (med - dolp) if polarity == "low" else (dolp - med)

    cand = ((score > k * mad) & (interior > 0)).astype(np.uint8) * 255
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    num, labels, stats, cents = cv2.connectedComponentsWithStats(cand, 8)
    R = 0.5 * max(xs.max() - xs.min(), ys.max() - ys.min())

    best, best_score = None, -1.0
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        bx, by = cents[i]
        if ((bx - cx) ** 2 + (by - cy) ** 2) ** 0.5 > max_move_frac * R:
            continue
        s = score[labels == i].mean() * np.sqrt(stats[i, cv2.CC_STAT_AREA])
        if s > best_score:
            best_score, best = s, (float(bx), float(by))
    return best if best is not None else (cx, cy)
# -----------------------------------------------------------------------------


def polar_raw_to_bgr(raw, representation=REPRESENTATION):
    """Render a raw 4-channel polarisation PNG (or DoFP mosaic) to a natural BGR
    image at full frame resolution, so pen marks read as ordinary photo content.

    Output resolution matches the stored annotation coordinate space."""
    rgba = readPolarPNMToRGBALive(raw)            # H x W x 4 (channels p0,p45,p90,p135)
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError(f"Expected packed 4-channel polar image, got {rgba.shape}")

    if representation == "rgb":
        # p0->R, p45->G, p90->B (drops p135) — matches the annotator's RGBA2BGR view
        return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)

    # average the four polarisation channels -> a clean, natural intensity image
    gray = rgba.astype(np.float32).mean(axis=2)
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    if representation == "clahe":
        gray = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _mask_blobs(mask, min_area=MIN_BLOB_AREA):
    """Return [(cx, cy, area), ...] for each connected component in a binary mask."""
    num, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    blobs = []
    for i in range(1, num):                       # 0 is background
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= min_area:
            cx, cy = centroids[i]
            blobs.append((float(cx), float(cy), area))
    return blobs


class Sam3Client:
    """Thin wrapper around the SAM3 gradio server (text-promptable segmentation).

    The gradio endpoint is: predict(handle_file(image), prompt) -> mask PNG path.
    Connection is lazy so importing this module never touches the network."""

    def __init__(self, ip=SAM3_IP, port=SAM3_PORT):
        self.url = f"http://{ip}:{port}"
        self._client = None

    def _ensure(self):
        if self._client is None:
            from gradio_client import Client      # imported lazily (optional dep)
            print(f"[AutoAnnotator] Connecting to SAM3 at {self.url}")
            self._client = Client(self.url)

    def segment(self, bgr, prompt):
        """Segment everything matching `prompt`; returns a binary uint8 mask (0/255)
        the same H x W as `bgr`."""
        self._ensure()
        from gradio_client import handle_file

        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        cv2.imwrite(path, bgr)
        try:
            result = self._client.predict(handle_file(path), prompt,
                                          api_name="/segment")
        except Exception as e:
            raise RuntimeError(
                "SAM3 server raised an exception. The /segment endpoint is reachable "
                "but failing for every request (server has verbose errors disabled). "
                "Check the SAM3 server is healthy / its model is loaded.\n"
                f"Underlying error: {e}")
        finally:
            os.remove(path)

        # Returned image encodes: R channel = instance IDs, G/B = binary mask.
        out = cv2.imread(result, cv2.IMREAD_UNCHANGED)
        if out is None:
            raise RuntimeError(f"SAM3 returned an unreadable mask: {result!r}")
        if out.ndim == 3:
            mask = (out.max(axis=2) > 0).astype(np.uint8) * 255   # any instance = fg
        else:
            mask = (out > 0).astype(np.uint8) * 255
        if mask.shape[:2] != bgr.shape[:2]:
            mask = cv2.resize(mask, (bgr.shape[1], bgr.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
        return mask


class AutoAnnotator:
    """Pen-mark based annotation propagation between consecutive frames."""

    def __init__(self, ip=SAM3_IP, port=SAM3_PORT, prompt=DEFAULT_PROMPT,
                 representation=REPRESENTATION, mosaic_scale=MOSAIC_SCALE,
                 debug_dir=None):
        self.sam = Sam3Client(ip, port)
        self.prompt = prompt
        self.representation = representation
        self.mosaic_scale = mosaic_scale
        self.debug_dir = debug_dir

    def segment_frame(self, raw, prompt=None):
        """raw = cv2.imread(path, IMREAD_UNCHANGED). Returns (bgr, mask)."""
        bgr = polar_raw_to_bgr(raw, self.representation)
        mask = self.sam.segment(bgr, prompt or self.prompt)
        return bgr, mask

    def detect(self, raw, prompt=None, merge_dist=60, min_area=DETECT_MIN_AREA,
               reject_border=True, border_margin=8,
               refine=REFINE_DEFECT, polarity="low",
               max_area_frac=0.18, max_aspect=4.0):
        """Detect pen-mark(s) on a single frame from scratch (no prior annotation).

        Returns [(x, y, area), ...] in full-MOSAIC coords, largest mark first.
        Blobs whose centroids are within `merge_dist` (render px) of an already-kept
        larger blob are merged away (a thick ring can fragment into a few components).

        `reject_border`: drop marks whose bounding box touches the image edge. A circle
        clipped by the frame border usually means the actual defect is OUTSIDE the image
        (e.g. the scan ran past the sheet), so placing a point at its centroid would be
        wrong — leave those for the operator to judge.

        `refine`: nudge each point from the ring centroid onto the DoLP defect anomaly
        inside the ring (`polarity` "low" for negative dents, "high" for positive).
        """
        _bgr, mask = self.segment_frame(raw, prompt)
        H, W = mask.shape[:2]
        num, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)

        max_area = max_area_frac * H * W
        blobs = []  # (cx, cy, area, touches_border, label_idx)
        for i in range(1, num):
            a = int(stats[i, cv2.CC_STAT_AREA])
            if a < min_area or a > max_area:
                continue                       # too small (speck) or a segmentation blow-out
            x0, y0 = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
            w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            if max(w, h) / max(1, min(w, h)) > max_aspect:
                continue                       # elongated -> scratch/line, not a circular mark
            touches = (x0 <= border_margin or y0 <= border_margin or
                       x0 + w >= W - border_margin or y0 + h >= H - border_margin)
            cx, cy = cents[i]
            blobs.append((cx, cy, a, touches, i))
        blobs.sort(key=lambda b: -b[2])

        merged = []
        for x, y, a, t, idx in blobs:
            if all(((mx - x) ** 2 + (my - y) ** 2) ** 0.5 >= merge_dist
                   for mx, my, _, _, _ in merged):
                merged.append((x, y, a, t, idx))

        kept = [b for b in merged if not (reject_border and b[3])]

        dolp = dolp_from_raw(raw) if (refine and kept) else None
        out = []
        s = self.mosaic_scale
        for x, y, a, _t, idx in kept:
            if refine:
                ring_mask = (labels == idx).astype(np.uint8) * 255
                x, y = refine_defect_in_ring(dolp, ring_mask, x, y, polarity)
            out.append((float(x * s), float(y * s), a))
        return out

    def propagate(self, prev_raw, prev_points, next_raw, prompt=None):
        """Predict where each `prev_points` annotation lands on the next frame.

        prev_points : list of (x, y) in full-MOSAIC coords (as stored in the JSON).
        Returns a list the SAME length as prev_points; each item is a predicted
        (x, y) tuple in full-MOSAIC coords, or None when no mark could be associated.
        """
        prompt = prompt or self.prompt
        prev_bgr, prev_mask = self.segment_frame(prev_raw, prompt)
        next_bgr, next_mask = self.segment_frame(next_raw, prompt)

        prev_blobs = _mask_blobs(prev_mask)
        next_blobs = _mask_blobs(next_mask)

        # work in render (debayered) space; points come in mosaic space
        s = self.mosaic_scale
        pts_render = [(px / s, py / s) for (px, py) in prev_points]

        if self.debug_dir:
            self._dump_debug(prev_bgr, prev_mask, next_bgr, next_mask,
                             pts_render, prev_blobs, next_blobs)

        results = []
        for (px, py) in pts_render:
            if not prev_blobs or not next_blobs:
                results.append(None)
                continue

            # mark on frame N that this annotation sits on / nearest to
            bp = min(prev_blobs, key=lambda b: (b[0] - px) ** 2 + (b[1] - py) ** 2)
            if ((bp[0] - px) ** 2 + (bp[1] - py) ** 2) ** 0.5 > ASSOC_MAX_DIST:
                results.append(None)
                continue
            off_x, off_y = px - bp[0], py - bp[1]

            # same mark on frame N+1 (nearest centroid to the frame-N mark)
            bn = min(next_blobs, key=lambda b: (b[0] - bp[0]) ** 2 + (b[1] - bp[1]) ** 2)
            results.append((float((bn[0] + off_x) * s),
                            float((bn[1] + off_y) * s)))   # back to mosaic

        return results

    def _dump_debug(self, prev_bgr, prev_mask, next_bgr, next_mask,
                    prev_points, prev_blobs, next_blobs):
        os.makedirs(self.debug_dir, exist_ok=True)

        def overlay(bgr, mask, pts, blobs, color_pt):
            vis = bgr.copy()
            vis[mask > 0] = (0.5 * vis[mask > 0] +
                             0.5 * np.array([0, 0, 255])).astype(np.uint8)
            for (cx, cy, _a) in blobs:
                cv2.drawMarker(vis, (int(cx), int(cy)), (0, 255, 255),
                               cv2.MARKER_CROSS, 14, 2)
            for (x, y) in pts:
                cv2.circle(vis, (int(x), int(y)), 16, color_pt, 2)
            return vis

        cv2.imwrite(os.path.join(self.debug_dir, "prev.png"),
                    overlay(prev_bgr, prev_mask, prev_points, prev_blobs, (0, 255, 0)))
        cv2.imwrite(os.path.join(self.debug_dir, "next.png"),
                    overlay(next_bgr, next_mask, [], next_blobs, (0, 255, 0)))
        print(f"[AutoAnnotator] debug images written to {self.debug_dir}")


# --- CLI for quick experimentation without the GUI ---------------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Test pen-mark annotation propagation.")
    ap.add_argument("prev", help="previous frame PNG")
    ap.add_argument("next", help="next frame PNG")
    ap.add_argument("--x", type=float, required=True, help="annotation x on prev frame")
    ap.add_argument("--y", type=float, required=True, help="annotation y on prev frame")
    ap.add_argument("--ip", default=SAM3_IP)
    ap.add_argument("--port", default=SAM3_PORT)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--repr", default=REPRESENTATION, choices=["gray", "rgb"])
    ap.add_argument("--debug", default="./auto_debug", help="dir for overlay images")
    args = ap.parse_args()

    prev_raw = cv2.imread(args.prev, cv2.IMREAD_UNCHANGED)
    next_raw = cv2.imread(args.next, cv2.IMREAD_UNCHANGED)
    if prev_raw is None or next_raw is None:
        raise SystemExit("Could not load one of the input frames.")

    aa = AutoAnnotator(ip=args.ip, port=args.port, prompt=args.prompt,
                       representation=args.repr, debug_dir=args.debug)
    preds = aa.propagate(prev_raw, [(args.x, args.y)], next_raw)
    print("Prediction on next frame:", preds[0])
