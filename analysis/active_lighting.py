#!/usr/bin/env python3
"""
ActiveLighting: identify which of N cycling scene lights illuminates a frame,
score how good each light is at the current viewpoint, and suggest the next one.

Method (validated on FORTH_DoorCase_weld_650, all 600 frames, 2026-07-09):
 - Light signature = per-channel global mean of the (4-channel polarization)
   image, normalized to sum 1. Position-independent: it survives camera motion,
   unlike spatial illumination fingerprints. Same-light distance 0.007-0.02,
   different-light 0.04-0.20.
 - The lights fire in a FIXED cyclic order but the light clock is not locked to
   the camera framerate: per frame the light stays (dwell), advances one, or
   rarely skips one. So the candidate pool is {L, L+1, L+2} given previous
   light L. (frame_index mod numLights is wrong ~35% of the time.)
 - Per-light exemplars are rolling EMAs updated on EVERY pick - freshness beats
   confidence (a confident-only update rule goes stale and death-spirals).
 - The first numLights frames are assumed to be one clean cycle (bootstrap).

Goodness of a light = how well it exposes surface detail at the current
viewpoint: usable-exposure fraction x mean gradient energy (Tenengrad) over
the usable pixels. Tracked per light with an EMA so it follows the camera.

Usage:
    from ActiveLighting import ActiveLighting
    al = ActiveLighting(numLights=6)
    info = al.processFile(path)   # {"light":.., "margin":.., "goodness":..}
    al.suggestNext()              # best light to fire next (active control)
    al.predictNext()              # light the passive cycle will show next

Standalone demo (read-only):
    venv/bin/python ActiveLighting.py /path/to/dataset [--limit N]
"""

import os
import numpy as np
import cv2


class ActiveLighting:
    def __init__(self, numLights=6, emaAlpha=0.5, goodnessAlpha=0.3):
        self.numLights = numLights
        self.emaAlpha = emaAlpha          # exemplar signature EMA weight (new)
        self.goodnessAlpha = goodnessAlpha  # goodness EMA weight (new)
        self.exemplars = {}               # light id -> signature 4-vector
        self.goodnessEMA = {}             # light id -> EMA goodness
        self.state = None                 # light id of the last frame
        self.frameCount = 0
        self.transitionCounts = [0] * 3   # stay / advance / skip

    # ------------------------------------------------------------------ #
    @staticmethod
    def signature(image):
        """Per-channel global mean, normalized to sum 1. None if unusable."""
        if image is None or image.ndim != 3 or image.shape[2] < 2:
            return None
        m = image.reshape(-1, image.shape[2]).mean(axis=0).astype(np.float32)
        s = m.sum()
        if s <= 1e-6:
            return None
        return m / s

    @staticmethod
    def goodness(image):
        """Illumination quality at this viewpoint.
        Returns (score, detail, darkFraction, saturatedFraction).
        score = mean Tenengrad gradient energy over usable pixels, scaled to
        ~0..1, times the usable-pixel fraction. Higher is better."""
        if image is None:
            return 0.0, 0.0, 1.0, 0.0
        scale = 65535.0 if image.dtype == np.uint16 else 255.0
        img = image.astype(np.float32)
        if img.ndim == 3:
            intensity = img.mean(axis=2)
            chanMax = img.max(axis=2)
        else:
            intensity = img
            chanMax = img
        # half resolution is plenty for exposure/detail statistics
        intensity = cv2.resize(intensity, None, fx=0.5, fy=0.5,
                               interpolation=cv2.INTER_AREA)
        chanMax = cv2.resize(chanMax, None, fx=0.5, fy=0.5,
                             interpolation=cv2.INTER_AREA)
        dark = chanMax < 0.04 * scale
        saturated = chanMax > 0.96 * scale
        usable = ~(dark | saturated)
        usableFrac = float(usable.mean())
        if usableFrac < 0.01:
            return 0.0, 0.0, float(dark.mean()), float(saturated.mean())
        gx = cv2.Sobel(intensity, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(intensity, cv2.CV_32F, 0, 1, ksize=3)
        tenengrad = (gx * gx + gy * gy)[usable].mean()
        detail = float(np.sqrt(tenengrad) / scale)  # ~0..1
        return detail * usableFrac, detail, float(dark.mean()), float(saturated.mean())

    # ------------------------------------------------------------------ #
    def process(self, image, name=""):
        """Identify the light of one frame (frames must be fed in sequence).
        Returns a dict: light, margin, distance, transition, goodness,
        detail, dark, saturated, ok."""
        sig = self.signature(image)
        good, detail, dark, sat = self.goodness(image)
        result = {"name": name, "goodness": good, "detail": detail,
                  "dark": dark, "saturated": sat, "ok": sig is not None}

        if sig is None:
            # unreadable/corrupt frame: hold the current state
            result.update({"light": self.state, "margin": 0.0,
                           "distance": float("nan"), "transition": "hold"})
            return result

        if self.frameCount < self.numLights:
            # bootstrap: first numLights frames define the lights + cycle order
            light = self.frameCount
            self.exemplars[light] = sig
            result.update({"light": light, "margin": float("inf"),
                           "distance": 0.0, "transition": "bootstrap"})
        else:
            cands = [(self.state + k) % self.numLights for k in range(3)]
            dists = [float(np.linalg.norm(sig - self.exemplars[c]))
                     for c in cands]
            j = int(np.argmin(dists))
            light = cands[j]
            srt = sorted(dists)
            self.transitionCounts[j] += 1
            a = self.emaAlpha
            self.exemplars[light] = (1.0 - a) * self.exemplars[light] + a * sig
            result.update({"light": light, "margin": srt[1] - srt[0],
                           "distance": dists[j],
                           "transition": ("stay", "advance", "skip")[j]})

        g = self.goodnessAlpha
        if light in self.goodnessEMA:
            self.goodnessEMA[light] = (1.0 - g) * self.goodnessEMA[light] + g * good
        else:
            self.goodnessEMA[light] = good
        self.state = light
        self.frameCount += 1
        return result

    def processFile(self, path):
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        return self.process(image, name=os.path.basename(path))

    # ------------------------------------------------------------------ #
    def suggestNext(self):
        """Best light to fire next (active control): highest EMA goodness.
        Returns (lightId, {lightId: emaGoodness}) or (None, {}) before any data."""
        if not self.goodnessEMA:
            return None, {}
        best = max(self.goodnessEMA, key=self.goodnessEMA.get)
        return best, dict(self.goodnessEMA)

    def predictNext(self):
        """Light the passive cycle will most likely show on the next frame,
        based on the observed stay/advance/skip statistics."""
        if self.state is None:
            return None
        if sum(self.transitionCounts) == 0:
            step = 1  # no statistics yet: assume the cycle advances
        else:
            step = int(np.argmax(self.transitionCounts))
        return (self.state + step) % self.numLights


# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ActiveLighting demo (read-only)")
    parser.add_argument("directory", help="dataset directory with colorFrame_*.png")
    parser.add_argument("--limit", type=int, default=0, help="max frames (0 = all)")
    parser.add_argument("--verbose", action="store_true", help="print every frame")
    args = parser.parse_args()

    files = sorted(f for f in os.listdir(args.directory)
                   if f.startswith("colorFrame_") and f.endswith(".png"))
    if args.limit:
        files = files[:args.limit]

    al = ActiveLighting(numLights=6)
    lowMargin = 0
    for f in files:
        r = al.processFile(os.path.join(args.directory, f))
        if r["ok"] and r["transition"] not in ("bootstrap",) and r["margin"] < 0.005:
            lowMargin += 1
        if args.verbose or not r["ok"]:
            print("%s light=%s %-9s margin=%.4f goodness=%.4f dark=%.2f sat=%.3f%s"
                  % (r["name"], r["light"], r["transition"], r["margin"],
                     r["goodness"], r["dark"], r["saturated"],
                     "" if r["ok"] else "  <-- UNREADABLE"))

    print("\n%u frames, transitions stay/advance/skip = %s, ambiguous(<0.005) = %u"
          % (al.frameCount, al.transitionCounts, lowMargin))
    best, table = al.suggestNext()
    print("per-light goodness (EMA at end of sequence):")
    for light in sorted(table):
        print("  light %u : %.4f%s" % (light, table[light],
                                       "   <-- suggested next" if light == best else ""))
    print("passive cycle prediction for next frame: light %s (current %s)"
          % (al.predictNext(), al.state))
