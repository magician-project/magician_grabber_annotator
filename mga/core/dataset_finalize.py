#!/usr/bin/python3

"""
Author : "Ammar Qammaz"
Copyright : "2025 Foundation of Research and Technology, Computer Science Department Greece"
License : "FORTH"

Dataset finalize: the batch passes the Finalize button runs — leading-dark
frame detection, defect/severity totals, the fill-tracking pass, the
focus/light backfill, and the info.json update that certifies a dataset.

Extracted from mga/wx_annotator.py (Stage 3c of its refactor). The GUI keeps
thin delegators (streamer access, progress dialogs, the onSave(None) flush,
the stats commit and the frame reload); the per-frame JSON work lives here.
progress(i, msg) -> bool is an optional abort channel (False aborts);
frame_size feeds empty_annotation for frames whose JSON is missing.
No wx / UI imports.
"""

import os
import csv
import json
import glob
import re
import getpass
from datetime import datetime

import cv2

from mga.core.annotation_state import (empty_annotation, normalize_tracking,
                                       write_annotation_json)
from mga.core.tracking import (estimateFrameAffine, best_same_light_index,
                               tracking_record, solve_tracking_positions)
from mga.core.frame_processing import mosaicToBGR
from mga.core.read_data_annotator import annotation_json_path, read_annotation_json
from mga.core.visualize_data import tenengrad_focus_measure, determine_intensity_region
import mga.core.light_decoder as lightDecoder


def leading_dark_frames(images):
    """Count the consecutive dark ('No Light') frames at the very start of the dataset —
    caused by the latency between disabling the light safety and the scene light actually
    operating — so Finalize can set startFrame past them. Scans in sorted order and stops
    at the first correctly-lit frame; a dark frame later in the dataset (a genuine light
    failure) is left alone. An unreadable/placeholder leading frame counts as dark too."""
    leading = 0
    for img in images:
        raw = cv2.imread(img, cv2.IMREAD_UNCHANGED)
        if raw is not None:
            if determine_intensity_region(mosaicToBGR(raw), threshold=0.1) != "No Light":
                break  # first correctly-lit frame — stop
        leading += 1
    return leading


def defect_totals(local_dir):
    """Scan every frame JSON in the dataset and tally defect classes and severities."""
    defect_counts, severity_counts, total = {}, {}, 0
    for jp in glob.glob(os.path.join(local_dir, "colorFrame_0_*.json")):
        try:
            with open(jp) as f:
                d = json.load(f)
        except Exception:
            continue
        for c in d.get("pointClasses", []):
            defect_counts[c] = defect_counts.get(c, 0) + 1
        for s in d.get("pointSeverities", []):
            severity_counts[s] = severity_counts.get(s, 0) + 1
        total += len(d.get("pointClicks", []))
    return defect_counts, severity_counts, total


def fill_tracking(images, fp_cache, progress=None, frame_size=(0, 0)):
    """Batch tracking pass over the whole dataset (also run by Finalize):
      PASS 1 — every frame without 'tracking' records gets the measured transform
               from its previous frame plus the best same-lighting link.
      PASS 2 — all measurements are reconciled with a weighted least-squares solve
               of the resulting pose graph, and each frame's optimized global
               position (relative to the first frame) is stored as an extra
               'leastSquaresGlobal' record.
    Annotation JSONs are read-modified-written, so points/classes are untouched.
    progress(i, msg) -> bool is an optional abort channel (False aborts).
    Returns (filled, skipped, failed, solved, aborted), or None with <2 frames."""
    if len(images) < 2:
        return None

    def report(i, msg):
        return (progress is None) or progress(i, msg)

    # PASS 1 — measure missing records.
    filled = skipped = failed = 0
    aborted = False
    for i in range(1, len(images)):
        if not report(i, f"{i+1}/{len(images)} — {filled} filled, {skipped} had tracking"):
            aborted = True
            break

        jp = annotation_json_path(images[i])
        data = read_annotation_json(jp)
        if normalize_tracking(data) or []:
            skipped += 1
            continue

        try:
            M, (dx, dy), response, inliers = estimateFrameAffine(images[i - 1], images[i])
        except Exception as e:
            print("Fill Tracking: shift failed for", images[i], ":", e)
            failed += 1
            continue
        records = [tracking_record(images[i - 1], M, dx, dy, response, inliers)]

        # Best same-lighting link among the preceding frames (i-1 excluded).
        best_j, best_sim = best_same_light_index(images, i, fp_cache)
        if best_j is not None:
            try:
                sM, (sdx, sdy), sresp, sinl = estimateFrameAffine(images[best_j], images[i])
                records.append(tracking_record(images[best_j], sM, sdx, sdy, sresp, sinl,
                                               light_similarity=best_sim))
            except Exception as e:
                print("Fill Tracking: same-light shift failed for", images[i], ":", e)

        if not data:
            data = empty_annotation(frame_size[0], frame_size[1])
        data["tracking"] = records
        if write_annotation_json(jp, data, tag="Fill Tracking"):
            filled += 1
        else:
            failed += 1

    # PASS 2 — weighted least squares over the pose graph: solve for per-frame
    # global positions p_i (p_0 = 0) from all pairwise measurements p_b - p_a = s
    # (the solve itself lives in mga.core.tracking.solve_tracking_positions).
    solved = 0
    if not aborted:
        frame_json = []   # (json_path, data) per frame, aligned with images
        for i, img in enumerate(images):
            jp = annotation_json_path(img)
            data = read_annotation_json(jp)
            frame_json.append((jp, data))

        positions = solve_tracking_positions(
            images, [normalize_tracking(data) or [] for _jp, data in frame_json])

        first = os.path.basename(images[0])
        for i, (gx, gy) in positions.items():
            jp, data = frame_json[i]
            if not data:
                continue
            recs = [r for r in (normalize_tracking(data) or [])
                    if r.get("method") != "leastSquaresGlobal"]
            recs.append({"fromFrame": first,
                         "shift": [gx, gy],
                         "method": "leastSquaresGlobal"})
            data["tracking"] = recs
            if write_annotation_json(jp, data, tag="Fill Tracking"):
                solved += 1

    return filled, skipped, failed, solved, aborted


def compute_focus_light(images, progress=None, frame_size=(0, 0)):
    """Compute Tenengrad focus + a latency-corrected light direction for every
    frame and store them in each frame's JSON. Light is decoded SEQUENTIALLY by
    lightDecoder (drift-free CSV cycle corrected by the observed signature — see
    lightDecoder.py): it fixes the controller's per-frame latency stalls, tags a
    wiring-invariant canonical direction, and preserves the 'No Light' malfunction
    flag (same mean<18 test). Falls back to per-frame brightest-region when there
    is no controller.csv. Returns the number of frames written."""
    if not images:
        return 0

    def _frameno(p):
        m = re.search(r"colorFrame_\d+_(\d+)", os.path.basename(p))
        return int(m.group(1)) if m else None

    # Decode order = capture order (frame number); frames without one sort last.
    images = sorted(images, key=lambda p: (_frameno(p) is None, _frameno(p) or 0))
    img_dir = os.path.dirname(images[0])
    csv_rows = None
    csv_path = os.path.join(img_dir, "controller.csv")
    if os.path.isfile(csv_path):
        try:
            with open(csv_path, newline="") as cf:
                csv_rows = list(csv.DictReader(cf))
        except Exception as e:
            print("Finalize: controller.csv unreadable —", e)

    # Pass 1: read every frame once; gather focus + the decoder's observations.
    jps, datas, focus_list = [], [], []
    sigs, dark, csvl, dirs = [], [], [], []
    for i, img in enumerate(images):
        if progress is not None and not progress(i, f"{i+1}/{len(images)} frames"):
            break

        jp = annotation_json_path(img)
        data = read_annotation_json(jp)

        # raw stays as decoded: lightDecoder's signature/is_dark want the original
        # 4-channel packed PNG, not the repacked 2D mosaic.
        raw = cv2.imread(img, cv2.IMREAD_UNCHANGED)
        if raw is None:
            jps.append(jp); datas.append(data); focus_list.append(None)
            sigs.append(None); dark.append(True); csvl.append(-1)
            dirs.append(lightDecoder.UNKNOWN)
            continue
        imgCV = mosaicToBGR(raw)

        dk = lightDecoder.is_dark(raw)
        fn = _frameno(img)
        jps.append(jp); datas.append(data)
        focus_list.append(float(tenengrad_focus_measure(imgCV)))
        dark.append(dk)
        sigs.append(None if dk else lightDecoder.signature(raw))
        dirs.append(lightDecoder.NO_LIGHT if dk else lightDecoder.brightest_direction(raw))
        csvl.append(lightDecoder._csv_light(csv_rows[fn])
                    if (csv_rows and fn is not None and fn < len(csv_rows)) else -1)

    # Decode the whole sequence (CSV cycle + signatures); else brightest-region.
    use_decoder = bool(csv_rows) and any(c > 0 for c in csvl) and any(s is not None for s in sigs)
    if use_decoder:
        light_nums = lightDecoder.decode_light_ids(sigs, csvl, dark)
        conf       = lightDecoder.light_confidence(sigs, dark)
        light_dirs, _ = lightDecoder.canonical_directions(
            light_nums, dark, site=lightDecoder.site_of(img_dir), directions=dirs)
    else:
        light_nums = [-1] * len(sigs)
        conf       = [1.0] * len(sigs)
        light_dirs = dirs                      # legacy brightest-region behaviour

    # Pass 2: write focus + decoded light back into each frame JSON.
    updated = 0
    for k, jp in enumerate(jps):
        if focus_list[k] is None:
            continue                            # unreadable frame
        data = datas[k]
        if not data:
            data = empty_annotation(frame_size[0], frame_size[1])
        data["tenengradFocusMeasure"] = focus_list[k]
        data["lightDirection"]        = light_dirs[k]
        if light_nums[k] > 0:
            data["lightNumber"]     = light_nums[k]
            data["lightConfidence"] = round(conf[k], 3)
        if write_annotation_json(jp, data, tag="Finalize: focus/light"):
            updated += 1
    print(f"Finalize focus/light: {updated} frames written"
          + ("" if use_decoder else " (no controller.csv — brightest-region fallback)"))
    return updated


def update_info_json(info_path, defect_counts, severity_counts, total_defects,
                     stats, leading_dark_fn=None):
    """Read, update and write the dataset's info.json with certification info,
    the accumulated annotation-effort statistics, and the dataset-wide
    defect/severity totals. stats is a dict with keys active_seconds, clicks,
    keystrokes, points_added, points_deleted. leading_dark_fn() is called only
    when no startFrame has been set yet (the frame-scan is not free), mirroring
    the Finalize flow. Returns the written info dict, or None when the write
    failed (the caller shows the message)."""
    # Preserve existing (camera) fields; tolerate a missing or malformed info.json.
    info = {}
    if os.path.isfile(info_path):
        try:
            with open(info_path) as f:
                info = json.load(f)
        except Exception as e:
            print(f"Finalize: existing info.json unreadable ({e}); starting fresh.")

    # Auto-skip the leading dark frames caused by light-safety-off latency: when no
    # startFrame has been set yet, point it at the first correctly-lit frame.
    if "startFrame" not in info:
        leading_dark = leading_dark_fn() if leading_dark_fn is not None else 0
        if leading_dark > 0:
            info["startFrame"] = leading_dark
            print(f"Finalize: detected {leading_dark} leading dark frame(s); "
                  f"setting startFrame={leading_dark}")

    info["certified_by"]    = getpass.getuser()
    info["annotated_at"]    = datetime.now().strftime("%Y/%m/%d %H:%M")
    info["annotation_time"] = int(info.get("annotation_time", 0)) + int(round(stats["active_seconds"]))
    info["clicks"]          = int(info.get("clicks", 0)) + stats["clicks"]
    info["keystrokes"]      = int(info.get("keystrokes", 0)) + stats["keystrokes"]
    info["points_added"]    = int(info.get("points_added", 0)) + stats["points_added"]
    info["points_deleted"]  = int(info.get("points_deleted", 0)) + stats["points_deleted"]
    info["defect_counts"]   = defect_counts
    info["severity_counts"] = severity_counts
    info["total_defects"]   = total_defects

    if not write_annotation_json(info_path, info, indent=1):
        return None
    return info
