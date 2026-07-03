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

Key recall trick (validated, recall 0.57 -> 0.84): many pen rings are nearly invisible
in the intensity image but CRISP in the Degree-of-Linear-Polarisation (DoLP) map, because
the ink polarises light differently from the metal. So detect() runs SAM3 on BOTH the
intensity (CLAHE) and the DoLP rendering and unions the rings. The DoLP signal is also
used to (a) recover rings clipped by the frame edge when the defect is provably in-frame,
and (b) nudge each point from the ring centroid onto the actual defect.

v1 task: PROPAGATE an annotation from frame N to frame N+1.
The pen mark is physical ink, so it is robust to the per-frame lighting/polarisation
changes that break naive optical flow. We segment the mark on both frames, match the
blob, and carry the annotated point (keeping its offset inside the mark) to N+1.

Standalone test:
    python3 AutoAnnotator.py prevFrame.png nextFrame.png --x 612 --y 512 \
            --ip 139.91.185.16 --port 7860 --prompt "pen mark" --debug ./dbg
"""

import os
import shutil
import tempfile

import cv2
import numpy as np

from readData import readPolarPNMToRGBALive


# Route ALL gradio_client temp files into a PRIVATE per-process dir instead of the shared
# /tmp/gradio. gradio_client caches BOTH downloaded results AND uploaded inputs (content-sha
# copies, ~3 MB each) under GRADIO_TEMP_DIR — which it captures at import time — so this MUST
# be set before gradio_client is first imported (it is imported lazily in Sam3Client). Using a
# private dir means our per-call cleanup never races with other gradio servers/clients.
if "GRADIO_TEMP_DIR" not in os.environ:
    os.environ["GRADIO_TEMP_DIR"] = tempfile.mkdtemp(prefix="magician_gradio_")
_GRADIO_TMP = os.environ["GRADIO_TEMP_DIR"]


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
REFINE_PEAK_K     = 5.0    # a blob must also PEAK above this (MAD units) to be accepted — gates
                           # out weak/ambiguous anomalies (halves the negative-dent tail; positives,
                           # which have no crisp anomaly, cleanly fall back to the ring centroid).
                           # Tuned on 244 certified-GT rings: mean err 57->53, p90 123->114 mosaic px.
REFINE_MAX_MOVE   = 0.45   # max move from centroid as a fraction of the ring radius
REFINE_MAX_MOVE_ABS = 60   # AND an absolute cap (render px): on a LARGE ring, 0.45*R can be a
                           # big jump that lands a bad anomaly outside the defect. Capping the
                           # move recovers recall@150 (0.734->0.746 on certified GT) without
                           # hurting localization — the true defect is near the ring centre.
REFINE_MIN_AREA   = 10     # min anomaly-cluster area (render px)
REFINE_RING_DILATE = 21    # dilation (px) used to exclude the ink ring from the interior

# --- Recall heuristics (validated on FORTH_NEGA/POSA hand-GT: recall 0.57 -> 0.84) ---
# 1) DoLP-INPUT UNION: the pen ink polarises light, so a ring that is invisible in the
#    intensity (CLAHE) image is often crisp in the Degree-of-Linear-Polarisation map.
#    We run SAM3 on BOTH representations and union the rings — recovers ~64% of the rings
#    CLAHE-only misses. Costs a 2nd SAM3 query per frame (set use_dolp=False to disable).
USE_DOLP_INPUT    = True
# 2) BORDER recovery: a ring clipped by the frame edge is kept (not blanket-rejected) IFF
#    a DoLP anomaly confirms the defect is inside the visible arc (i.e. in-frame). Uses a
#    looser anomaly threshold than the centre refinement because partial arcs are noisier.
BORDER_REFINE_K   = 1.0
BORDER_MAX_MOVE   = 0.7

# --- TEMPORAL OFFSET CONSENSUS (batch passes: Full Auto / rerun) -------------------------
# The pen ring and the defect are both physical, so the vector (defect - ring centre) is
# CONSTANT along a dataset scan (oracle residual ~9 mosaic px median vs ~60 raw). Per-frame
# DoLP anomalies are noisy/absent on weak-contrast frames, but their MEDIAN over a ring
# track is excellent. So batch passes: (1) track rings across frames (velocity-predictive
# nearest-neighbour on ring centres), (2) collect LOOSE anomaly offsets per track,
# (3) if enough offsets agree, snap every track member to centre + median offset (a
# confident per-frame anomaly that already agrees is kept). Border rings keep their DoLP
# keep/drop gate but take their POSITION from the track (extrapolated centre + offset) —
# their own loose-anomaly placement dominates the error tail.
# If offsets do not agree (spread gate) the track is left on per-frame behaviour, which
# makes the pass a strict no-op fallback (e.g. positives with no anomaly signal).
# When the strict consensus fails (weak/absent DoLP picks — e.g. small class-C defects,
# positives), a MODE VOTE over ALL candidate anomaly blobs (DoLP + AoLP-variance maps)
# takes over: every frame's blobs vote with their (blob - centre) offset; the true defect
# offset repeats across frames, distractors don't. Votes are weighted by a centre prior
# (workers draw the ring around the defect: |GT-centre| ~0.1-0.2 of the ring radius).
# Measured on the 7 certified datasets (4200 frames, 2927 manual GT points):
# R 0.803->0.818, P 0.888->0.895, manual-loc median 38.6->23.2 mosaic px (NEGC 68->19,
# NEGB 43->13); positives preserved via the tight prior (POSC 45->44).
TEMPORAL_COLLECT_K      = 2.0   # loose anomaly gate used ONLY to collect offsets
TEMPORAL_COLLECT_PEAK_K = 2.5
TEMPORAL_MIN_SUPPORT    = 4     # >= this many offsets/votes in a track for a consensus
TEMPORAL_SPREAD_MAX     = 12    # render px: median |offset - median| must be below this
TEMPORAL_MAX_JUMP       = 120   # render px/frame: association gate with velocity predict
TEMPORAL_INIT_JUMP      = 300   # render px/frame: association gate before velocity known
TEMPORAL_MAX_GAP        = 5     # frames a track survives without a detection
TEMPORAL_BLEND_R        = 10    # keep a ring's own confident anomaly within this of the
                                # consensus (render px)
TEMPORAL_BORDER_ASSOC_R = 80    # border ring adopts a track position if the extrapolated
                                # centre is within this of its arc centre (render px)
TEMPORAL_BORDER_BLEND_R = 25    # ... unless its own anomaly already agrees this closely
TEMPORAL_BORDER_RESCUE  = True  # border rings DROPPED by the DoLP gate are re-kept when a
                                # consensus track extrapolates onto them (R .812->.817)
TEMPORAL_VOTE_K         = 1.5   # blob threshold (MAD units) for the vote pool
TEMPORAL_VOTE_TOPN      = 8     # strongest blobs per map per frame that may vote
TEMPORAL_VOTE_RHO       = 8.0   # render px: vote cluster radius
TEMPORAL_VOTE_FRAC      = 0.25  # winning cluster must cover this fraction of the track
TEMPORAL_VOTE_PRIOR     = {"low": 0.30, "high": 0.10}   # centre-prior sigma as a fraction
                                # of ring radius, per polarity (positives are drawn more
                                # centred AND their maps have more distractors)
# -----------------------------------------------------------------------------


def dolp_from_raw(raw):
    """Degree of Linear Polarisation map (debayer res, [0,1]) from a raw 4-channel frame.
    Matches visualizeData.py: S0=p0+p90, S1=p0-p90, S2=p45-p135, DoLP=|S|/S0."""
    return polar_maps(raw)[0]


def polar_maps(raw):
    """(dolp, aolp_variance) maps from a raw 4-channel frame (debayer res, one debayer).
    aolp_variance = circular variance of the polarisation ANGLE in a 9x9 window — small
    class-C defects that are invisible in DoLP magnitude still disturb the angle field."""
    rgba = readPolarPNMToRGBALive(raw).astype(np.float32)
    p0, p45, p90, p135 = rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2], rgba[:, :, 3]
    S0 = p0 + p90
    S1 = p0 - p90
    S2 = p45 - p135
    dolp = np.clip(np.sqrt(S1 * S1 + S2 * S2) / (S0 + 1e-6), 0.0, 1.0)
    two_a = np.arctan2(S2, S1)
    c = cv2.blur(np.cos(two_a), (9, 9))
    s = cv2.blur(np.sin(two_a), (9, 9))
    aolpv = 1.0 - np.sqrt(c * c + s * s)
    return dolp, aolpv


def polarity_for_class(defect_class):
    """Which DoLP direction marks the defect for a given class label.
    Positive dents bulge out -> HIGH DoLP; everything else (negative dents, …) -> LOW."""
    return "high" if "positive" in str(defect_class).lower() else "low"


def dolp_to_bgr(dolp):
    """Render a DoLP map ([0,1], render res) to a CLAHE-enhanced BGR image for SAM3.
    The pen ink reads as a crisp ring here even when it is invisible in intensity."""
    g = np.clip(dolp * 255.0, 0, 255).astype(np.uint8)
    g = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(g)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def _dolp_anomaly(dolp, ring_mask, cx, cy, polarity="low",
                  k=REFINE_K, max_move_frac=REFINE_MAX_MOVE,
                  min_area=REFINE_MIN_AREA, ring_dilate=REFINE_RING_DILATE,
                  peak_k=REFINE_PEAK_K, max_move_abs=REFINE_MAX_MOVE_ABS):
    """Find the DoLP defect anomaly inside a ring. Returns (x, y) in render coords, or
    None when no confident anomaly is present (e.g. the defect is outside the frame).

    dolp      : full-frame DoLP map (render res).
    ring_mask : binary mask of THIS ring's connected component (render res).
    polarity  : "low" (negative dent) or "high" (positive dent).
    """
    ys, xs = np.where(ring_mask > 0)
    if len(xs) < 3:
        return None
    # interior = inside the ring (convex hull) minus the ink itself
    hull = cv2.convexHull(np.column_stack((xs, ys)))
    disk = np.zeros_like(ring_mask)
    cv2.fillConvexPoly(disk, hull, 255)
    ring = cv2.dilate(ring_mask, np.ones((ring_dilate, ring_dilate), np.uint8))
    interior = cv2.bitwise_and(disk, cv2.bitwise_not(ring))
    if interior.sum() < 20:
        return None

    dv = dolp[interior > 0]
    med = np.median(dv)
    mad = np.median(np.abs(dv - med)) + 1e-6
    score = (med - dolp) if polarity == "low" else (dolp - med)

    cand = ((score > k * mad) & (interior > 0)).astype(np.uint8) * 255
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    num, labels, stats, cents = cv2.connectedComponentsWithStats(cand, 8)
    R = 0.5 * max(xs.max() - xs.min(), ys.max() - ys.min())

    move_limit = min(max_move_frac * R, max_move_abs)
    best, best_score = None, -1.0
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        bx, by = cents[i]
        if ((bx - cx) ** 2 + (by - cy) ** 2) ** 0.5 > move_limit:
            continue
        blob = (labels == i)
        if score[blob].max() < peak_k * mad:       # require a strong peak, not just wide+weak
            continue
        s = score[blob].mean() * np.sqrt(stats[i, cv2.CC_STAT_AREA])
        if s > best_score:
            best_score, best = s, (float(bx), float(by))
    return best


def _vote_offsets(dolp, aolpv, ring_mask, cx, cy, polarity, R,
                  k=TEMPORAL_VOTE_K, topn=TEMPORAL_VOTE_TOPN,
                  min_area=REFINE_MIN_AREA, ring_dilate=REFINE_RING_DILATE,
                  max_move_frac=REFINE_MAX_MOVE, max_move_abs=REFINE_MAX_MOVE_ABS):
    """All plausible anomaly-blob offsets (blob - centre) of this ring, pooled from the
    DoLP map (dataset polarity) and the AoLP-variance map ("high"), for the temporal
    mode vote. Returns [(dx, dy), ...] (render px), strongest `topn` per map."""
    ys, xs = np.where(ring_mask > 0)
    if len(xs) < 3:
        return []
    margin = ring_dilate * 2
    H, W = ring_mask.shape[:2]
    x0, x1 = max(0, xs.min() - margin), min(W, xs.max() + margin)
    y0, y1 = max(0, ys.min() - margin), min(H, ys.max() + margin)
    rm = ring_mask[y0:y1, x0:x1]
    ys_c, xs_c = np.where(rm > 0)
    hull = cv2.convexHull(np.column_stack((xs_c, ys_c)))
    disk = np.zeros_like(rm)
    cv2.fillConvexPoly(disk, hull, 255)
    ink = cv2.dilate(rm, np.ones((ring_dilate, ring_dilate), np.uint8))
    interior = cv2.bitwise_and(disk, cv2.bitwise_not(ink))
    if interior.sum() < 20:
        return []

    limit = min(max_move_frac * R, max_move_abs)
    offs = []
    for sig, direction in ((dolp, polarity), (aolpv, "high")):
        crop = sig[y0:y1, x0:x1]
        iv = crop[interior > 0]
        med = np.median(iv)
        mad = np.median(np.abs(iv - med)) + 1e-6
        score = (med - crop) / mad if direction == "low" else (crop - med) / mad
        cand = ((score > k) & (interior > 0)).astype(np.uint8) * 255
        cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        num, labels, stats, cents = cv2.connectedComponentsWithStats(cand, 8)
        rows = []
        for i in range(1, num):
            a = int(stats[i, cv2.CC_STAT_AREA])
            if a < min_area:
                continue
            bx, by = cents[i][0] + x0, cents[i][1] + y0
            if ((bx - cx) ** 2 + (by - cy) ** 2) ** 0.5 > limit:
                continue
            rows.append((float(score[labels == i].mean()) * np.sqrt(a),
                         bx - cx, by - cy))
        rows.sort(reverse=True)
        offs.extend((dx, dy) for _s, dx, dy in rows[:topn])
    return offs


def refine_defect_in_ring(dolp, ring_mask, cx, cy, polarity="low", **kw):
    """Backward-compatible wrapper: nudge (cx, cy) onto the DoLP anomaly, or leave it at
    the centroid if none is found."""
    a = _dolp_anomaly(dolp, ring_mask, cx, cy, polarity, **kw)
    return a if a is not None else (cx, cy)


def _circle_center(ring_mask, cx_fallback, cy_fallback):
    """Least-squares (Kasa) circle-fit centre of the ring ink — a better estimate of the DRAWN
    circle's centre (where the defect sits) than the ink centroid, which is biased for partial
    'C' rings. Falls back to the centroid on a degenerate fit. (recall 0.746->0.754, precision
    0.899->0.903 on certified GT.)"""
    ys, xs = np.where(ring_mask > 0)
    if len(xs) < 5:
        return cx_fallback, cy_fallback
    x = xs.astype(np.float64)
    y = ys.astype(np.float64)
    try:
        c, *_ = np.linalg.lstsq(np.c_[2 * x, 2 * y, np.ones(len(x))], x * x + y * y, rcond=None)
        cx, cy = float(c[0]), float(c[1])
    except Exception:
        return cx_fallback, cy_fallback
    R = 0.5 * max(np.ptp(xs), np.ptp(ys))   # ndarray.ptp was removed in numpy 2.x
    if ((cx - xs.mean()) ** 2 + (cy - ys.mean()) ** 2) ** 0.5 > R:   # reject a runaway fit
        return cx_fallback, cy_fallback
    return cx, cy
# -----------------------------------------------------------------------------


def temporal_consensus(frame_cands, mosaic_scale=MOSAIC_SCALE):
    """Batch post-pass over per-frame detect_ex() results of ONE dataset scan.

    frame_cands : list of (frame_idx, cands) — cands as returned by detect_ex(),
                  frame_idx integer position in the scan (consecutive frames ~1 apart).
    Returns {frame_idx: [(x, y, area), ...]} in FULL-MOSAIC coords (detect()-compatible).

    See the TEMPORAL_* constants block for the idea. Rings whose track offsets do not
    agree (or with too little support) keep their per-frame points, so this pass can only
    refine, never destabilise. Border rings keep their keep/drop decision but adopt the
    track-extrapolated centre + consensus offset as their position when available.
    """
    # --- track NON-border rings across frames (velocity-predictive nearest-neighbour) ---
    tracks = []    # each: list of (frame_idx, cand)
    active = []    # [track, last_frame, (cx, cy), vel or None]
    for f, cands in sorted(frame_cands, key=lambda fc: fc[0]):
        used = set()
        nxt = []
        for st in active:
            tr, lf, (lx, ly), vel = st
            gap = f - lf
            if gap > TEMPORAL_MAX_GAP:
                continue
            if vel is not None:
                px, py = lx + vel[0] * gap, ly + vel[1] * gap
                gate = TEMPORAL_MAX_JUMP * gap
            else:
                px, py = lx, ly
                gate = TEMPORAL_INIT_JUMP * gap
            best, bd = None, 1e18
            for i, c in enumerate(cands):
                if i in used or c["touches"]:
                    continue
                d = ((c["cx"] - px) ** 2 + (c["cy"] - py) ** 2) ** 0.5
                if d < bd:
                    bd, best = d, i
            if best is not None and bd <= gate:
                c = cands[best]
                used.add(best)
                tr.append((f, c))
                nxt.append([tr, f, (c["cx"], c["cy"]),
                            ((c["cx"] - lx) / gap, (c["cy"] - ly) / gap)])
            else:
                nxt.append(st)
        for i, c in enumerate(cands):
            if i in used or c["touches"]:
                continue
            tr = [(f, c)]
            tracks.append(tr)
            nxt.append([tr, f, (c["cx"], c["cy"]), None])
        active = nxt

    # --- consensus offset per track; snap members ---
    points = {}          # id(cand) -> (x, y) render coords
    track_info = []      # (track, median_offset or None)
    for tr in tracks:
        # 1) STRICT consensus: median of confident per-frame anomaly picks, spread-gated
        offs = np.array([c["loose_off"] for _f, c in tr if c["loose_off"] is not None])
        mo = None
        if len(offs) >= TEMPORAL_MIN_SUPPORT:
            m = np.median(offs, axis=0)
            spread = float(np.median(np.hypot(offs[:, 0] - m[0], offs[:, 1] - m[1])))
            if spread <= TEMPORAL_SPREAD_MAX:
                mo = m
        # 2) fallback: MODE VOTE over all candidate blobs (selection-robust; centre prior)
        if mo is None:
            mo = _track_vote(tr)
        track_info.append((tr, mo))
        if mo is None:
            continue
        for _f, c in tr:
            own = (c["x"] - c["cx"], c["y"] - c["cy"])
            if (own[0] or own[1]) and \
                    ((own[0] - mo[0]) ** 2 + (own[1] - mo[1]) ** 2) ** 0.5 <= TEMPORAL_BLEND_R:
                continue                     # its own confident anomaly agrees: keep it
            points[id(c)] = (c["cx"] + mo[0], c["cy"] + mo[1])

    # --- border rings: adopt track-extrapolated centre + offset when a track is nearby ---
    for f, cands in frame_cands:
        for c in cands:
            if not c["touches"]:
                continue
            if c["dropped"] and not TEMPORAL_BORDER_RESCUE:
                continue
            bestp, bestd = None, 1e18
            for tr, mo in track_info:
                if mo is None or len(tr) < 2:
                    continue
                mf, mc = min(tr, key=lambda fc: abs(fc[0] - f))
                gap = f - mf
                if abs(gap) > TEMPORAL_MAX_GAP:
                    continue
                vs = [((c2["cx"] - c1["cx"]) / (f2 - f1), (c2["cy"] - c1["cy"]) / (f2 - f1))
                      for (f1, c1), (f2, c2) in zip(tr, tr[1:]) if f2 > f1]
                vx = float(np.median([v[0] for v in vs]))
                vy = float(np.median([v[1] for v in vs]))
                ex, ey = mc["cx"] + vx * gap, mc["cy"] + vy * gap
                d = ((ex - c["cx"]) ** 2 + (ey - c["cy"]) ** 2) ** 0.5
                if d < bestd:
                    bestd, bestp = d, (ex + mo[0], ey + mo[1])
            if bestp is not None and bestd <= TEMPORAL_BORDER_ASSOC_R:
                if not c["dropped"] and \
                        ((c["x"] - bestp[0]) ** 2 + (c["y"] - bestp[1]) ** 2) ** 0.5 \
                        <= TEMPORAL_BORDER_BLEND_R:
                    continue                 # own anomaly agrees with the track: keep it
                points[id(c)] = bestp        # (also RESCUES gate-dropped border rings)

    s = mosaic_scale
    out = {}
    for f, cands in frame_cands:
        row = []
        for c in cands:
            p = points.get(id(c))
            if p is None:
                if c["dropped"]:
                    continue                 # dropped border ring, no track rescued it
                p = (c["x"], c["y"])
            row.append((float(p[0] * s), float(p[1] * s), c["area"]))
        out[f] = row
    return out


def _track_vote(tr):
    """Mode vote over all candidate anomaly-blob offsets of a track. The physical
    (defect - ring centre) offset repeats across frames; distractor blobs don't.
    Votes carry a centre-prior weight. Returns the winning cluster's median offset,
    or None (insufficient / non-dominant support)."""
    rho = TEMPORAL_VOTE_RHO
    offs = []                                # (frame, dx, dy, weight)
    for f, c in tr:
        pf = TEMPORAL_VOTE_PRIOR[c["pol"]]
        for dx, dy in c["vote_offs"]:
            w = float(np.exp(-0.5 * ((dx * dx + dy * dy) / (pf * c["R"]) ** 2)))
            offs.append((f, dx, dy, w))
    if not offs:
        return None
    cell = {}                                # cell -> {frame: max weight}
    for f, dx, dy, w in offs:
        cx, cy = int(np.floor(dx / rho)), int(np.floor(dy / rho))
        for ox in (-1, 0, 1):
            for oy in (-1, 0, 1):
                d = cell.setdefault((cx + ox, cy + oy), {})
                if w > d.get(f, 0.0):
                    d[f] = w
    (bcx, bcy), sup = max(cell.items(), key=lambda kv: sum(kv[1].values()))
    if len(sup) < TEMPORAL_MIN_SUPPORT:
        return None
    if len(sup) < TEMPORAL_VOTE_FRAC * len({f for f, _c in tr}):
        return None
    ccx, ccy = (bcx + 0.5) * rho, (bcy + 0.5) * rho
    sel = np.array([(dx, dy) for f, dx, dy, w in offs
                    if abs(dx - ccx) <= 1.5 * rho and abs(dy - ccy) <= 1.5 * rho])
    return np.median(sel, axis=0)


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
            # downloads default to GRADIO_TEMP_DIR (our private _GRADIO_TMP); uploads go there too
            self._client = Client(self.url, download_files=_GRADIO_TMP)

    def segment(self, bgr, prompt):
        """Segment everything matching `prompt`; returns a binary uint8 mask (0/255)
        the same H x W as `bgr`."""
        self._ensure()
        from gradio_client import handle_file

        # gradio_client's _download_file streams each result into
        # tempfile.gettempdir()/<40-hex-token>/ (in a worker thread, so we can't redirect it),
        # moves the file out, and leaves the empty dir behind — one leaked inode per query.
        # Snapshot the token dirs before the call and remove the new empty ones after.
        tmproot = tempfile.gettempdir()
        _is_token = lambda n: len(n) == 40 and all(c in "0123456789abcdef" for c in n)
        try:
            before = {n for n in os.listdir(tmproot) if _is_token(n)}
        except OSError:
            before = set()

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
            try:
                os.remove(path)
            except OSError:
                pass
            # remove only the NEW, now-empty staging dirs this call created (safe for other
            # processes: their in-flight dirs aren't empty, and we skip anything non-empty)
            try:
                for n in os.listdir(tmproot):
                    if _is_token(n) and n not in before:
                        d = os.path.join(tmproot, n)
                        if os.path.isdir(d) and not os.listdir(d):
                            os.rmdir(d)
            except OSError:
                pass

        # Returned image encodes: R channel = instance IDs, G/B = binary mask.
        out = cv2.imread(result, cv2.IMREAD_UNCHANGED)
        # Wipe our PRIVATE gradio temp dir (both the uploaded-input cache and downloaded
        # results live here). Calls are sequential, so nothing is in flight once the result is
        # read. We never touch the shared /tmp/gradio — that would race with other processes.
        for name in os.listdir(_GRADIO_TMP):
            p = os.path.join(_GRADIO_TMP, name)
            try:
                shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
            except OSError:
                pass
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

    def _mask_candidates(self, mask, min_area, max_area_frac, max_aspect, border_margin):
        """Extract ring candidates from one SAM3 mask.
        Returns [(cx, cy, area, touches_border, ring_mask), ...] in render coords."""
        H, W = mask.shape[:2]
        max_area = max_area_frac * H * W
        num, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
        cands = []
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
            cands.append((cx, cy, a, touches, (labels == i).astype(np.uint8) * 255))
        return cands

    def detect(self, raw, prompt=None, merge_dist=60, min_area=DETECT_MIN_AREA,
               reject_border=True, border_margin=8,
               refine=REFINE_DEFECT, polarity="low",
               max_area_frac=0.18, max_aspect=4.0, use_dolp=USE_DOLP_INPUT):
        """Detect pen-mark(s) on a single frame from scratch (no prior annotation).

        Returns [(x, y, area), ...] in full-MOSAIC coords, largest mark first.

        Recall heuristics (both leverage polarimetry — see module header):
        - `use_dolp`: also run SAM3 on the DoLP map and UNION the rings, recovering faint
          ink invisible in intensity. Costs a 2nd SAM3 query per frame.
        - `reject_border`: a ring clipped by the frame edge is now KEPT iff a DoLP anomaly
          confirms the defect is inside the visible arc (in-frame); otherwise dropped (the
          defect is likely off-frame). Set reject_border=False to keep all border rings.

        Blobs within `merge_dist` (render px) of an already-kept larger blob — including
        across the two representations — are merged away. `refine` nudges each non-border
        point from the ring centroid onto the DoLP defect anomaly (`polarity` "low" for
        negative dents, "high" for positive).
        """
        s = self.mosaic_scale
        return [(c["x"] * s, c["y"] * s, c["area"])
                for c in self.detect_ex(raw, prompt, merge_dist, min_area, reject_border,
                                        border_margin, refine, polarity, max_area_frac,
                                        max_aspect, use_dolp)
                if not c["dropped"]]

    def detect_ex(self, raw, prompt=None, merge_dist=60, min_area=DETECT_MIN_AREA,
                  reject_border=True, border_margin=8,
                  refine=REFINE_DEFECT, polarity="low",
                  max_area_frac=0.18, max_aspect=4.0, use_dolp=USE_DOLP_INPUT):
        """detect() returning rich per-ring dicts in RENDER coords (for batch passes that
        run `temporal_consensus` afterwards):
          x, y      : final per-frame point (as detect() would output, / MOSAIC_SCALE)
          cx, cy    : ring centre estimate (circle fit)
          area      : ink area (render px)
          R         : ring radius estimate (render px)
          touches   : ring clipped by the frame border
          dropped   : border ring REJECTED by the DoLP gate (detect() omits these; the
                      temporal pass may rescue them onto a consensus track)
          pol       : the polarity used ("low"/"high")
          loose_off : (dx, dy) LOOSE DoLP-anomaly offset from centre, for the temporal
                      offset consensus — or None (border rings / no anomaly)
          vote_offs : [(dx, dy), ...] candidate anomaly-blob offsets (DoLP + AoLP-var)
                      for the temporal mode vote
        """
        prompt = prompt or self.prompt
        dolp, aolpv = polar_maps(raw)

        # Segment on intensity (CLAHE) and, optionally, on the DoLP map.
        cands = self._mask_candidates(
            self.sam.segment(polar_raw_to_bgr(raw, self.representation), prompt),
            min_area, max_area_frac, max_aspect, border_margin)
        if use_dolp:
            cands += self._mask_candidates(
                self.sam.segment(dolp_to_bgr(dolp), prompt),
                min_area, max_area_frac, max_aspect, border_margin)

        # Largest-first; drop blobs that duplicate an already-kept one (also de-dups the
        # same ring found in both representations).
        cands.sort(key=lambda b: -b[2])
        merged = []
        for c in cands:
            if all(((m[0] - c[0]) ** 2 + (m[1] - c[1]) ** 2) ** 0.5 >= merge_dist
                   for m in merged):
                merged.append(c)

        out = []
        for cx0, cy0, a, touches, ring_mask in merged:
            ys, xs = np.where(ring_mask > 0)
            R = 0.5 * max(np.ptp(xs), np.ptp(ys)) if len(xs) else 1.0
            cx, cy = _circle_center(ring_mask, cx0, cy0)  # drawn-circle centre beats ink centroid
            x, y = cx, cy
            loose_off, vote_offs, dropped = None, [], False
            if touches and reject_border:
                # keep only if a DoLP anomaly proves the defect is inside the visible arc
                an = _dolp_anomaly(dolp, ring_mask, cx, cy, polarity,
                                   k=BORDER_REFINE_K, max_move_frac=BORDER_MAX_MOVE,
                                   peak_k=BORDER_REFINE_K)   # keep loose: this gates RECALL, not localization
                if an is None:
                    dropped = True
                else:
                    x, y = an
            elif refine:
                an = _dolp_anomaly(dolp, ring_mask, cx, cy, polarity)
                if an is not None:
                    x, y = an
                lo = _dolp_anomaly(dolp, ring_mask, cx, cy, polarity,
                                   k=TEMPORAL_COLLECT_K, peak_k=TEMPORAL_COLLECT_PEAK_K)
                if lo is not None:
                    loose_off = (lo[0] - cx, lo[1] - cy)
                vote_offs = _vote_offsets(dolp, aolpv, ring_mask, cx, cy, polarity, R)
            out.append(dict(x=float(x), y=float(y), cx=float(cx), cy=float(cy),
                            area=a, R=float(R), touches=bool(touches), dropped=dropped,
                            pol=polarity, loose_off=loose_off, vote_offs=vote_offs))
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
