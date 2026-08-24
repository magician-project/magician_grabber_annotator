#!/usr/bin/python3

"""
Author : "Ammar Qammaz"
Copyright : "2025 Foundation of Research and Technology, Computer Science Department Greece"
License : "FORTH"

Pure frame-to-frame tracking primitives for the annotator: lighting
fingerprints, blockwise phase-correlation affine estimation, tracking-record
construction and the least-squares pose-graph solve. No wx / UI imports —
extracted from mga/wx_annotator.py (Stage 1 of its refactor) so the Track
button and the Fill Tracking batch pass share one definition of each.
"""

import os
import cv2
import numpy as np

from mga.core.read_data_annotator import readPolarPNMToRGBA
from mga.core.annotation_state import is_near_any

# The illumination cycles through the scene lights during acquisition, so nearby
# frames repeat the same lighting. The Track button records a direct transform to
# the earlier frame whose lighting fingerprint best matches the destination's —
# scanning back at most this many frames (fingerprints, not a fixed period, so
# this stays correct across framerates).
SAME_LIGHT_SEARCH_MAX = 12
# Minimum fingerprint cosine similarity to accept a frame as "same lighting"
# (same-light pairs score >0.95 on FORTH_DoorCase_weld_650, differently-lit
# neighbours <0.7).
SAME_LIGHT_MIN_SIMILARITY = 0.90


def lightingFingerprint(path, grid=4):
    """Compact lighting signature of a frame: the per-channel × grid×grid cell mean
    intensities, mean-subtracted and L2-normalized. Cosine similarity between two
    fingerprints separates the scene-light cycle far better than the coarse 6-region
    lightDirection label, which cannot when the part occupies one image corner.
    Read through readPolarPNMToRGBA so raw .pnm mosaics and already packed .png
    frames — which a dataset can mix — yield the same channel count and therefore
    comparable fingerprints. Returns None when the image cannot be read."""
    raw = readPolarPNMToRGBA(path)
    if raw is None:
        return None
    raw = raw.astype(np.float32)
    H, W, C = raw.shape
    cells = []
    for c in range(C):
        for gy in range(grid):
            for gx in range(grid):
                cells.append(raw[gy*H//grid:(gy+1)*H//grid,
                                 gx*W//grid:(gx+1)*W//grid, c].mean())
    v = np.array(cells, np.float32)
    v -= v.mean()
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def lighting_fingerprint_cached(path, cache):
    """lightingFingerprint(path) with a caller-owned cache dict; None-entries are
    not cached (the same policy as the annotator's old _lightFingerprintCached)."""
    fp = cache.get(path)
    if fp is None:
        fp = lightingFingerprint(path)
        if fp is not None:
            cache[path] = fp
    return fp


def best_same_light_index(images, i, fp_cache,
                          max_back=SAME_LIGHT_SEARCH_MAX,
                          min_similarity=SAME_LIGHT_MIN_SIMILARITY):
    """Index of the frame whose lighting matches images[i] best among the
    max_back frames before it, or None when nothing scores above
    min_similarity. The adjacent frame (i-1) is excluded — it is already the
    tracking record [0]."""
    fp_i = lighting_fingerprint_cached(images[i], fp_cache)
    if fp_i is None:
        return None, 0.0
    best_j, best_sim = None, 0.0
    for j in range(i - 2, max(-1, i - 2 - max_back), -1):
        fp_j = lighting_fingerprint_cached(images[j], fp_cache)
        if fp_j is None:
            continue
        sim = float(fp_i @ fp_j)
        if sim > best_sim:
            best_j, best_sim = j, sim
    if best_j is None or best_sim < min_similarity:
        return None, 0.0
    return best_j, best_sim


def estimateFrameAffine(prev_path, next_path, block=256, step=128, min_block_resp=0.08):
    """Estimate the transform between two consecutive frames as a similarity
    (rotation + scale + translation): the camera motion is not purely 2D, so a
    global translation drifts (~0.4 deg and ~0.3% scale per frame on
    FORTH_DoorCase_weld_650). Phase correlation is the only primitive robust to
    the frame-to-frame lighting cycle, so it is applied per block on a grid and a
    RANSAC similarity is fitted through the block motions; when fewer than 4
    inlier blocks survive, falls back to the global translation.
    Returns (M, (cdx, cdy), response, inliers) with the 2x3 matrix M and the
    displacement (cdx, cdy) of the image centre both in full-mosaic coordinates,
    response the global phase-correlation confidence.
    Both frames are read through readPolarPNMToRGBA, so a raw .pnm mosaic and an
    already packed .png — which a dataset can mix — are correlated at the same
    half-mosaic resolution instead of failing to broadcast."""
    shift_scale = 2.0   # correlation runs on the half-res demosaic
    imgs = []
    for path in (prev_path, next_path):
        img = readPolarPNMToRGBA(path)
        if img is None:
            raise IOError("Could not load %s" % path)
        imgs.append(img.astype(np.float32).mean(axis=2))
    ia, ib = imgs
    H, W = ia.shape
    win = cv2.createHanningWindow((W, H), cv2.CV_32F)
    (dx, dy), response = cv2.phaseCorrelate(ia * win, ib * win)

    src, dst = [], []
    bwin = cv2.createHanningWindow((block, block), cv2.CV_32F)
    for y0 in range(0, H - block + 1, step):
        for x0 in range(0, W - block + 1, step):
            blk = ia[y0:y0 + block, x0:x0 + block]
            if blk.std() < 8:
                continue  # flat/dark block carries no alignment signal
            xs, ys = int(round(x0 + dx)), int(round(y0 + dy))
            if xs < 0 or ys < 0 or xs + block > W or ys + block > H:
                continue
            (bx, by), r = cv2.phaseCorrelate(
                blk * bwin, ib[ys:ys + block, xs:xs + block] * bwin)
            if r < min_block_resp:
                continue
            c = block / 2.0
            src.append([x0 + c, y0 + c])
            dst.append([xs + c + bx, ys + c + by])

    M, inliers = None, 0
    if len(src) >= 4:
        M, inl = cv2.estimateAffinePartial2D(np.float32(src), np.float32(dst),
                                             ransacReprojThreshold=4.0)
        inliers = 0 if inl is None else int(inl.sum())
    if M is None or inliers < 4:
        M, inliers = np.float64([[1, 0, dx], [0, 1, dy]]), 0

    # To mosaic coordinates: the linear part is scale-invariant, translation scales.
    Mm = np.float64(M).copy()
    Mm[:, 2] *= shift_scale
    cx, cy = W * shift_scale / 2.0, H * shift_scale / 2.0
    cdx = Mm[0, 0] * cx + Mm[0, 1] * cy + Mm[0, 2] - cx
    cdy = Mm[1, 0] * cx + Mm[1, 1] * cy + Mm[1, 2] - cy
    return Mm, (float(cdx), float(cdy)), float(response), inliers


def tracking_record(from_path, M, dx, dy, response, inliers, fallback=False,
                    light_similarity=None):
    """One inter-frame transform record in the 'tracking' JSON schema: the
    measured (or fallback) transform from from_path plus bookkeeping. Same
    construction for the adjacent-frame record and the same-lighting record."""
    rec = {"fromFrame": os.path.basename(from_path),
           "shift": [dx, dy],
           "affine": M.tolist() if hasattr(M, "tolist") else M,
           "response": response,
           "inliers": inliers,
           "method": "phaseCorrelateAffine" if inliers else "phaseCorrelate",
           "fallback": fallback}
    if light_similarity is not None:
        rec["lightSimilarity"] = round(light_similarity, 3)
    return rec


def solve_tracking_positions(images, per_frame_records):
    """Weighted least-squares solve of the inter-frame tracking pose graph:
    global positions p_i (p_0 = 0) from all pairwise measurements p_b - p_a = s.

    images: frame paths in capture order (graph node ids).
    per_frame_records: per frame, that frame's 'tracking' records (a bare dict
    from the early format must already be wrapped in a list by the caller).
    Returns {frame_index: (gx, gy)} for every MEASURED frame index >= 1 —
    unmeasured frames are absent so a stored zero is never invented. Weights:
    the record's response (floor 0.01), x0.3 for fallback records, fixed 2.0
    for hand-corrected 'manual' records."""
    name2idx = {os.path.basename(p): i for i, p in enumerate(images)}
    measurements = []
    for i, records in enumerate(per_frame_records):
        for r in records:
            if r.get("method") == "leastSquaresGlobal":
                continue
            a = name2idx.get(r.get("fromFrame", ""))
            s = r.get("shift")
            if a is None or a == i or not s:
                continue
            w = max(float(r.get("response") or 0.0), 0.01)
            if r.get("fallback"):
                w *= 0.3
            if r.get("method") == "manual":
                w = 2.0   # hand-corrected shifts outweigh any estimate
            measurements.append((a, i, float(s[0]), float(s[1]), w))

    if not measurements:
        return {}

    N = len(images)
    A  = np.zeros((len(measurements), N - 1), np.float64)
    bx = np.zeros(len(measurements), np.float64)
    by = np.zeros(len(measurements), np.float64)
    constrained = set()
    for row, (a, b, dx, dy, w) in enumerate(measurements):
        if b > 0:
            A[row, b - 1] += w
        if a > 0:
            A[row, a - 1] -= w
        bx[row] = w * dx
        by[row] = w * dy
        constrained.update((a, b))
    px = np.linalg.lstsq(A, bx, rcond=None)[0]
    py = np.linalg.lstsq(A, by, rcond=None)[0]

    return {i: (float(px[i - 1]), float(py[i - 1]))
            for i in range(1, N) if i in constrained}


# Point-carry and manual-correction helpers (Stage 3d of the wx_annotator
# refactor): the Track button's carry loop and the Nudge dialog's shift/rotate
# previously lived inline in the GUI handlers.
def prior_shift_from_record(record, cur, direction, from_index):
    """Smooth-motion prior from the adjacent-frame tracking record, sign-corrected
    for our direction of travel (+1 next / -1 previous), or None when the record
    is not adjacent or has no shift."""
    s = record.get("shift")
    if s and from_index is not None:
        if from_index == cur - direction:
            return (s[0], s[1])
        elif from_index == cur + direction:
            return (-s[0], -s[1])
    return None


def propagate_points(points, classes, severities, M, W, H,
                     min_dist, default_class, default_severity, existing):
    """Transform each point by the 2x3 affine M into the destination frame;
    drop results landing outside (W, H) or within min_dist of a point the frame
    already has (`existing`) — re-pressing Track or tracking onto a partially
    annotated frame must not double up. Returns (pts, classes, severities) of
    the carried points; their sources are all 'auto' (predictions), matching
    the Track button."""
    carried_pts, carried_cls, carried_sev = [], [], []
    for i, (x, y) in enumerate(points):
        tx = M[0, 0] * x + M[0, 1] * y + M[0, 2]
        ty = M[1, 0] * x + M[1, 1] * y + M[1, 2]
        if not (0 <= tx < W and 0 <= ty < H):
            continue
        if is_near_any(tx, ty, existing, min_dist):
            continue
        carried_pts.append((tx, ty))
        carried_cls.append(classes[i] if i < len(classes) else default_class)
        carried_sev.append(severities[i] if i < len(severities) else default_severity)
    return carried_pts, carried_cls, carried_sev


def nudge_auto_points(points, sources, ddx, ddy):
    """Shift every auto-sourced point by (ddx, ddy) in place (Nudge dialog)."""
    for i, src in enumerate(sources):
        if src == "auto" and i < len(points):
            x, y = points[i]
            points[i] = (x + ddx, y + ddy)


def rotate_auto_points(points, sources, deg):
    """Rotate the auto-sourced block about its own centroid, so it turns in
    place; returns the centroid so the tracking record can fold the same
    rotation, or None when the frame has no auto points. +deg turns clockwise
    on screen (y points down)."""
    idx = [i for i, s in enumerate(sources) if s == "auto" and i < len(points)]
    if not idx:
        return None
    cx = sum(points[i][0] for i in idx) / len(idx)
    cy = sum(points[i][1] for i in idx) / len(idx)
    a = np.radians(deg)
    ca, sa = np.cos(a), np.sin(a)
    for i in idx:
        x, y = points[i]
        dx, dy = x - cx, y - cy
        points[i] = (cx + ca * dx - sa * dy, cy + sa * dx + ca * dy)
    return cx, cy


def nudge_tracking_record(rec, ddx, ddy):
    """Fold a nudge into tracking record rec: shift += (ddx, ddy), the affine
    translation columns follow, method marked 'manual'."""
    sx, sy = rec.get("shift", [0, 0])
    rec["shift"] = [sx + ddx, sy + ddy]
    aff = rec.get("affine")
    if aff:
        aff[0][2] += ddx
        aff[1][2] += ddy
    rec["method"] = "manual"


def rotate_tracking_record(rec, deg, cx, cy):
    """Fold a rotation about (cx, cy) into the record's affine:
    p' = R(Mp - c) + c. 'shift' is untouched — a turn about the centroid moves
    nothing. Marks method 'manual'."""
    aff = rec.get("affine")
    if aff:
        a = np.radians(deg)
        ca, sa = np.cos(a), np.sin(a)
        a00, a01, a02 = aff[0]
        a10, a11, a12 = aff[1]
        aff[0][0] = ca * a00 - sa * a10
        aff[0][1] = ca * a01 - sa * a11
        aff[0][2] = ca * (a02 - cx) - sa * (a12 - cy) + cx
        aff[1][0] = sa * a00 + ca * a10
        aff[1][1] = sa * a01 + ca * a11
        aff[1][2] = sa * (a02 - cx) + ca * (a12 - cy) + cy
    rec["method"] = "manual"
