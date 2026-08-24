#!/usr/bin/python3

"""
Author : "Ammar Qammaz"
Copyright : "2025 Foundation of Research and Technology, Computer Science Department Greece"
License : "FORTH"

Pure frame decoding / processing helpers for the annotator: no wx / UI imports.
Extracted from mga/wx_annotator.py (Stage 1 of its refactor) so the render
pipeline, the classifier paths and the batch passes share one definition of
"decode a frame".

The canonical-light remap (canonicalize_lighting) is also here; it is pure
except for the caller-owned exemplar `cache` dict (one entry per dataset
directory). The dict is mutated lazily on first use of a directory — a
prefetch worker and the main thread may race to bootstrap the same directory,
which is harmless (both compute identical exemplars and one write wins).
"""

import os
import cv2
import numpy as np

from mga.core.read_data_annotator import repackPolarToMosaic

# processor name -> processingWay number, mirroring the 27-branch chain this
# replaces (the choices list itself lives in wx_annotator.py as `processors`).
PROCESSOR_WAYS = {
    "PolarizationRGB1":       0,
    "PolarizationRGB2":       1,
    "PolarizationRGB3":       2,
    "Sobel":                  3,
    "Visible":                4,
    "Polarization_0_degree":  5,
    "Polarization_45_degree": 6,
    "Polarization_90_degree": 7,
    "Polarization_135_degree":8,
    "AoLP":                   9,
    "DoLP":                  10,
    "Intensity":             11,
    "s0":                    12,
    "s1":                    13,
    "s2":                    14,
    "s3":                    15,
    "AoLP (light)":          16,
    "AoLP (dark)":           17,
    "DoP":                   18,
    "DoCP":                  19,
    "ToP":                   20,
    "CoP":                   21,
    "RetardationMag":        22,
    "MaxMinAvgRGB":          23,
    "Normals":               24,
}


def mosaicToBGR(decoded):
    """BGR image for the renderers from a decoded frame: 4-channel packed PNGs
    (p0,p45,p90,p135) are re-packed to the original 2x2 mosaic and replicated
    over 3 channels, gray images are converted, BGR passed through. No
    canonical-light remap — callers that want one use loadFrameMosaic."""
    if (decoded.ndim == 3) and (decoded.shape[2] == 4):
        p0   = decoded[:, :, 0]
        p45  = decoded[:, :, 1]
        p90  = decoded[:, :, 2]
        p135 = decoded[:, :, 3]
        mosaic = repackPolarToMosaic(p0, p45, p90, p135)   # now 2D, as classifier expects
        return cv2.merge([mosaic, mosaic, mosaic])
    return (decoded if (decoded.ndim == 3 and decoded.shape[2] == 3)
            else cv2.cvtColor(decoded, cv2.COLOR_GRAY2BGR))


def loadFrameMosaic(path, canonical_cache=None):
    """Load a frame as (imgPNM, imgCV): the 2D mosaic the classifier expects
    and the BGR image the renderers use (see mosaicToBGR). Returns (None, None)
    when the file cannot be read.

    canonical_cache: when given, the strobed light is remapped to light #0
    first (see canonicalize_lighting); pass None to skip. Touches no UI and no
    shared state beyond the caller's cache, so it is safe to run in a worker
    thread."""
    imgPNM = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if imgPNM is None:
        return None, None
    if (imgPNM.ndim == 3) and (imgPNM.shape[2] == 4):
        p0   = imgPNM[:, :, 0]
        p45  = imgPNM[:, :, 1]
        p90  = imgPNM[:, :, 2]
        p135 = imgPNM[:, :, 3]
        imgPNM = repackPolarToMosaic(p0, p45, p90, p135)   # now 2D, as classifier expects
        if canonical_cache is not None:
            imgPNM = canonicalize_lighting(imgPNM, path, canonical_cache)
    elif (imgPNM.ndim == 2) and canonical_cache is not None:
        imgPNM = canonicalize_lighting(imgPNM, path, canonical_cache)  # .pnm mosaic path
    imgCV = mosaicToBGR(imgPNM)
    return imgPNM, imgCV


def _bootstrapLightExemplars(dirpath, numLights):
    """Exemplar signatures for the dataset's first numLights frames, assumed to
    be one clean strobe cycle (same bootstrap as ActiveLighting), or None."""
    try:
        import glob as _glob
        frames = sorted(_glob.glob(os.path.join(dirpath, "colorFrame_0_*.pnm")) +
                        _glob.glob(os.path.join(dirpath, "colorFrame_0_*.png")))[:numLights]
        exemplars = []
        for f in frames:
            img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
            if img is not None and img.ndim == 3 and img.shape[2] == 4:
                img = repackPolarToMosaic(img[:, :, 0], img[:, :, 1], img[:, :, 2], img[:, :, 3])
            if img is None or img.ndim != 2:
                continue
            quads = (img[0::2, 0::2], img[0::2, 1::2], img[1::2, 0::2], img[1::2, 1::2])
            means = np.array([float(q.mean()) for q in quads], dtype=np.float32)
            exemplars.append({'means': np.maximum(means, 1e-6),
                              'sig': means / max(float(means.sum()), 1e-6)})
        if len(exemplars) == numLights:
            print(f"[CanonicalLight] bootstrapped {numLights} light exemplars from {dirpath}")
            return exemplars
        else:
            print(f"[CanonicalLight] bootstrap failed ({len(exemplars)}/{numLights} usable frames)")
    except Exception as e:
        print(f"[CanonicalLight] bootstrap error: {e}")
    return None


def canonicalize_lighting(mosaic, filepath, cache, numLights=6):
    """Remap the strobed light of this frame so it renders as light #0.

    Light identity is resolved with the ActiveLighting signature (per-channel
    global mean proportions — position independent). Exemplars come from the
    dataset's first numLights frames, assumed to be one clean strobe cycle
    (same bootstrap as ActiveLighting). The remap is a per-channel gain
    exemplar0/exemplarK applied on the 2x2 mosaic quadrants, so it works for
    both .pnm mosaics and re-bayered .png frames.

    `cache` is a plain dict keyed by dataset directory: exemplars are computed
    once per directory and stored (a None entry marks a failed bootstrap so it
    is not retried on every frame)."""
    if mosaic is None or mosaic.ndim != 2:
        return mosaic
    dirpath = os.path.dirname(filepath)
    if dirpath not in cache:
        cache[dirpath] = _bootstrapLightExemplars(dirpath, numLights)
    exemplars = cache[dirpath]
    if not exemplars:
        return mosaic

    quads = (mosaic[0::2, 0::2], mosaic[0::2, 1::2], mosaic[1::2, 0::2], mosaic[1::2, 1::2])
    means = np.array([float(q.mean()) for q in quads], dtype=np.float32)
    sig = means / max(float(means.sum()), 1e-6)
    dists = [float(np.linalg.norm(sig - e['sig'])) for e in exemplars]
    k = int(np.argmin(dists))
    srt = sorted(dists)
    print(f"[CanonicalLight] frame light={k} (distance {srt[0]:.4f}, margin {srt[1]-srt[0]:.4f})"
          + ("" if k else " — already canonical"))
    if k == 0:
        return mosaic
    gains = exemplars[0]['means'] / exemplars[k]['means']
    maxval = 65535.0 if mosaic.dtype == np.uint16 else 255.0
    out = mosaic.astype(np.float32)
    out[0::2, 0::2] *= float(gains[0])
    out[0::2, 1::2] *= float(gains[1])
    out[1::2, 0::2] *= float(gains[2])
    out[1::2, 1::2] *= float(gains[3])
    return np.clip(out, 0, maxval).astype(mosaic.dtype)
