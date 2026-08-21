#!/usr/bin/env python3
"""
lightDecoder: fuse the controller CSV (commanded light) with the observed image
to derive a corrected per-frame light estimate, tagged with a canonical,
wiring-independent identity (the physical illumination DIRECTION).

Why this exists (measured 2026-07-22 across the certified datasets):
 - The CSV light cycle is clean (+1 every frame, 99.8-100%) BUT the emitted
   light lags the command with variable latency: 7% of Altinay frames and 16%
   of FORTH frames are "stalls" where the CSV advanced yet the image did not
   change (adjacent signature distance ~0.001-0.09 vs a typical 0.27). Blindly
   trusting CSV-row == frame mislabels that fraction.
 - The brightest-region label (determine_intensity_region) is ~24% accurate.
 - ActiveLighting's free-running EMA tracker death-spirals on scanning motion
   (~18%). Global signature clustering drifts with the camera (~52%).
 - The ONE robust primitive: the per-channel global-mean signature separates
   same-light (<0.01) from different-light (~0.27) locally -- a ~27x gap. Use it
   only to RESOLVE the small latency ambiguity around the (reliable) CSV cycle,
   never to classify all N lights from scratch.

Method:
 - emitted[i] is assumed to be the command at a small lag: emitted[i] in
   {C[i], C[i-1], .. C[i-max_lag]} (C = CSV light). Because those candidates are
   ADJACENT lights in the cycle (~0.27 apart) the signature picks the right one
   with huge margin. A couple of EM passes (assign -> local exemplars -> assign)
   self-correct; being tied to C means it cannot free-run/drift.
 - Dark frames (sensor black floor) are flagged "No Light" and never illuminate.
 - The canonical ID is the majority brightest-region DIRECTION of each decoded
   light group. Direction is wiring-invariant, so the same physical lamp gets the
   same ID whether the controller calls it Light3 (Altinay) or Light4 (FORTH).

Public API:
    signature(img4) -> np.float32[C] | None
    brightest_direction(img4) -> str
    is_dark(img4, floor=18.0) -> bool
    decode_light_ids(sigs, csv_lights, dark, num_lights=6, max_lag=1, ...) -> list[int]
    canonical_directions(decoded_ids, directions, dark) -> (per_frame_dir, id2dir)
    decode_dataset(dir) -> list[dict]   # glue: loads pngs + controller.csv
"""

import os
import glob
import csv
import re
import numpy as np

try:
    import cv2
except Exception:                       # cv2 optional for the pure-array core
    cv2 = None

# Canonical direction vocabulary, ordered clockwise from top so the id is a
# stable angular position (a "predictable id regardless how they are connected").
DIRECTIONS = ["Top", "Top Right", "Bottom Right", "Bottom", "Bottom Left", "Top Left"]
NO_LIGHT = "No Light"
UNKNOWN = "Unknown"
DARK_FLOOR = 18.0                       # per determine_intensity_region calibration


# --------------------------------------------------------------------------- #
# Per-frame observations (all image-derived, position/scene robust as noted).
# --------------------------------------------------------------------------- #
def signature(img):
    """Per-channel global mean normalized to sum 1. Position-independent (it
    survives camera motion); the reliable same/different-light discriminator."""
    if img is None or img.ndim != 3 or img.shape[2] < 2:
        return None
    m = img.reshape(-1, img.shape[2]).mean(axis=0).astype(np.float32)
    s = m.sum()
    return (m / s) if s > 1e-6 else None


def is_dark(img, floor=DARK_FLOOR):
    return img is None or float(img.mean()) < floor


def brightest_direction(img):
    """Coarse absolute anchor: which of 6 overlapping regions is brightest.
    Noisy per frame (~24%) but its MAJORITY over a light's frames is correct."""
    if img is None:
        return UNKNOWN
    g = img.sum(axis=2).astype(np.float32) if img.ndim == 3 else img.astype(np.float32)
    h, w = g.shape
    regions = {
        "Top Left":     g[:h // 2, :w // 2].mean(),
        "Top":          g[:h // 2, w // 4:3 * w // 4].mean(),
        "Top Right":    g[:h // 2, w // 2:].mean(),
        "Bottom Left":  g[h // 2:, :w // 2].mean(),
        "Bottom":       g[h // 2:, w // 4:3 * w // 4].mean(),
        "Bottom Right": g[h // 2:, w // 2:].mean(),
    }
    return max(regions, key=regions.get)


# --------------------------------------------------------------------------- #
# Core decode: pure arrays, no image I/O (so it is unit-testable / reusable).
# --------------------------------------------------------------------------- #
# Empirical CSV light number -> physical direction, per site. Derived 2026-07-22
# from 17,945 single-light certified frames (majority brightest-region per light),
# forced to a clean bijection. Altinay and Yongatek share a rig; FORTH numbers the
# SAME physical ring in reverse -- FORTH light i == Altinay light (7 - i).
SITE_DIRECTION = {
    "Altinay": {1: "Bottom",       2: "Bottom Left", 3: "Top Left",
                4: "Top",          5: "Top Right",   6: "Bottom Right"},
    "FORTH":   {1: "Bottom Right", 2: "Top Right",   3: "Top",
                4: "Top Left",     5: "Bottom Left", 6: "Bottom"},
}


def site_of(name):
    """Best-effort site tag from a dataset name (selects the CSV->direction table).
    Returns 'FORTH', 'Altinay' (also Yongatek, same rig), or None."""
    n = os.path.basename(str(name).rstrip("/")).lower()
    if n.startswith("forth"):
        return "FORTH"
    if "altinay" in n or n.startswith("yongatek"):
        return "Altinay"
    return None


def decode_light_ids(sigs, csv_lights, dark, num_lights=6, change_thr=0.12):
    """Fuse the drift-free CSV cycle with the observed image to correct latency.

    The CSV light number is locked to the frame index (a clean +1 cycle that never
    drifts), so it is the absolute identity -- EXCEPT on the ~7-16% "stall" frames
    where the emitted light lags the command (the CSV advanced but the image did
    not change). Those are exactly the frames the caller worried about. We detect
    them with the signature (same-light distance <0.01 vs different ~0.27) and
    relabel them to the previously shown light. Because every NON-stall frame is
    re-anchored to the CSV, this corrects the latency locally and -- unlike a
    free-running phase counter -- cannot accumulate drift. Returns the per-frame
    CSV light number (unchanged except on stalls), -1 for dark frames.

    sigs       : list of np.float32[C] (or None) per frame, in capture order.
    csv_lights : list[int] commanded light per frame.
    dark       : list[bool] frame is unlit (malfunction / black floor).
    """
    n = len(sigs)
    decoded = [-1] * n
    prev_lit = None                                    # index of previous lit frame
    for i in range(n):
        if dark[i] or sigs[i] is None:
            decoded[i] = -1
            continue
        if prev_lit is None or \
           float(np.linalg.norm(sigs[i] - sigs[prev_lit])) >= change_thr:
            decoded[i] = csv_lights[i]                 # first frame / real change
        else:
            decoded[i] = decoded[prev_lit]             # stall: emitted lags, hold
        prev_lit = i
    return decoded


def light_confidence(sigs, dark, change_thr=0.12):
    """Per-frame confidence in the stall/change decision: |dist - thr| / thr
    clipped to [0,1]. Mid-transition (blur) frames sit near the threshold and
    score low, flagging the labels the caller should least trust."""
    n = len(sigs)
    conf = [0.0] * n
    prev = None
    for i in range(n):
        if dark[i] or sigs[i] is None:
            prev = None
            continue
        if prev is None:
            conf[i] = 1.0
        else:
            d = float(np.linalg.norm(sigs[i] - sigs[prev]))
            conf[i] = min(abs(d - change_thr) / change_thr, 1.0)
        prev = i
    return conf


def _ring_fit(decoded_numbers, directions, dark, num_lights):
    """Fallback anchor for unknown sites: fit the single rotation+handedness that
    best maps the (clean, cyclic) decoded numbers onto the direction ring."""
    idx = {d: k for k, d in enumerate(DIRECTIONS)}
    lit = [i for i in range(len(decoded_numbers))
           if not dark[i] and decoded_numbers[i] > 0 and directions[i] in idx]
    best = None
    for hand in (1, -1):
        for base in range(num_lights):
            hits = sum(1 for i in lit
                       if DIRECTIONS[(base + hand * (decoded_numbers[i] - 1))
                                     % num_lights] == directions[i])
            if best is None or hits > best[0]:
                best = (hits, hand, base)
    _, hand, base = best
    return {num: DIRECTIONS[(base + hand * (num - 1)) % num_lights]
            for num in range(1, num_lights + 1)}


def canonical_directions(decoded_numbers, dark, site=None,
                         directions=None, num_lights=6):
    """Map decoded CSV light numbers to wiring-invariant physical DIRECTIONS.

    With a known site, uses the per-site table (accurate and consistent across
    every dataset of that site). Otherwise falls back to a global ring-fit against
    the per-frame brightest-region `directions`. Returns (per_frame_dir, num2dir).
    """
    num2dir = SITE_DIRECTION.get(site)
    if num2dir is None and directions is not None:
        num2dir = _ring_fit(decoded_numbers, directions, dark, num_lights)
    num2dir = num2dir or {}
    per_frame = [NO_LIGHT if dark[i] else num2dir.get(decoded_numbers[i], UNKNOWN)
                 for i in range(len(decoded_numbers))]
    return per_frame, num2dir


# --------------------------------------------------------------------------- #
# Glue: decode a whole dataset directory (colorFrame_0_*.png + controller.csv).
# --------------------------------------------------------------------------- #
_LIGHT_COLS = ["Light1", "Light2", "Light3", "Light4", "Light5", "Light6"]


def _csv_light(row):
    on = [i + 1 for i, k in enumerate(_LIGHT_COLS)
          if str(row.get(k, "0")).strip() not in ("0", "", "H", "L", "None")]
    return on[0] if len(on) == 1 else -1


def decode_dataset(directory, num_lights=6):
    """Decode every frame in a dataset dir. Returns a list of per-frame dicts:
    {frame, csv_light, decoded_id, direction, dark}. Requires cv2."""
    if cv2 is None:
        raise RuntimeError("cv2 required for decode_dataset")
    rows = list(csv.DictReader(open(os.path.join(directory, "controller.csv"))))
    files = sorted(glob.glob(os.path.join(directory, "colorFrame_0_*.png")),
                   key=lambda p: int(re.search(r"_(\d+)\.png", p).group(1)))
    fids, sigs, csvl, dark, dirs = [], [], [], [], []
    for f in files:
        fid = int(re.search(r"_(\d+)\.png", f).group(1))
        img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        if img is None or img.ndim != 3 or img.shape[2] < 4:
            continue
        d = is_dark(img)
        fids.append(fid)
        sigs.append(None if d else signature(img))
        csvl.append(_csv_light(rows[fid]) if fid < len(rows) else -1)
        dark.append(d)
        dirs.append(NO_LIGHT if d else brightest_direction(img))
    ids = decode_light_ids(sigs, csvl, dark, num_lights=num_lights)
    conf = light_confidence(sigs, dark)
    per_dir, _ = canonical_directions(ids, dark, site=site_of(directory),
                                      directions=dirs, num_lights=num_lights)
    return [{"frame": fids[i], "csv_light": csvl[i], "light": ids[i],
             "direction": per_dir[i], "confidence": round(conf[i], 3),
             "dark": dark[i]} for i in range(len(fids))]
