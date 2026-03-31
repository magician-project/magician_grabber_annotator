#!/usr/bin/python3
"""
rlAnnotator.py - Reinforcement Learning annotation pass.

Iterates every image in the local dataset directory, runs the classifier,
and for each AI detection that is:
  • further than `radius` pixels from every user annotation, AND
  • further than `radius` pixels from any existing RLClean / Clean annotation
adds an "RLClean" point annotation to the frame's JSON.
"""

import json
import math
import os
import threading

import wx


# ---------------------------------------------------------------------------
# Geometry helper
# ---------------------------------------------------------------------------

def _dist(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _any_within(pt, point_list, radius):
    return any(_dist(pt, q) <= radius for q in point_list)


# ---------------------------------------------------------------------------
# Per-frame JSON read/write helpers (no UI dependency)
# ---------------------------------------------------------------------------

_CLEAN_LABELS = {"rlclean", "clean"}


def _load_json(json_path):
    """Return (points, classes, severities, full_dict) or empty defaults."""
    if json_path and os.path.isfile(json_path):
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
            pts  = [tuple(p) for p in data.get("pointClicks",    [])]
            cls  = list(data.get("pointClasses",   []))
            sev  = list(data.get("pointSeverities", []))
            return pts, cls, sev, data
        except Exception:
            pass
    return [], [], [], {}


def _save_json(json_path, data, pts, cls, sev):
    """Write updated points/classes/severities back into the JSON."""
    data["pointClicks"]     = [list(p) for p in pts]
    data["pointClasses"]    = cls
    data["pointSeverities"] = sev
    with open(json_path, "w") as f:
        json.dump(data, f, sort_keys=False)


def _resolve_json(image_path):
    """Return preferred JSON annotation path for an image (new-style)."""
    root, _ = os.path.splitext(image_path)
    new_style = root + ".json"
    # Check legacy names too
    for candidate in (new_style,
                      image_path + ".json",
                      root + ".pnm.json",
                      root + ".png.json"):
        if os.path.isfile(candidate):
            return candidate
    return new_style   # default write target even if it doesn't exist yet


# ---------------------------------------------------------------------------
# Core per-frame processing (no wx, safe to call from background thread)
# ---------------------------------------------------------------------------

def _load_image_for_classifier(image_path):
    """
    Load an image the same way wxAnnotator does before passing it to the classifier:
      - cv2.imread with IMREAD_UNCHANGED
      - if 4-channel PNG (p0,p45,p90,p135), repack to 2-D mosaic via repackPolarToMosaic
    Returns the image array, or None on failure.
    """
    import cv2
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 3 and img.shape[2] == 4:
        from readData import repackPolarToMosaic
        p0, p45, p90, p135 = img[:, :, 0], img[:, :, 1], img[:, :, 2], img[:, :, 3]
        img = repackPolarToMosaic(p0, p45, p90, p135)
    return img


def process_frame(image_path, classifier, radius):
    """
    Run classifier on `image_path`, then update the annotation JSON.

    Returns (new_count, skipped_count, error_str_or_None).
    """
    # --- read image ---
    img = _load_image_for_classifier(image_path)
    if img is None:
        return 0, 0, f"Could not read image: {image_path}"

    # --- run classifier ---
    try:
        _, _occupancy, ai_ann = classifier.forward(img, majorityVote=True)
    except Exception as e:
        return 0, 0, f"Classifier error: {e}"

    if ai_ann is None:
        return 0, 0, None

    ai_pts = [tuple(p) for p in ai_ann.get("points", [])]
    if not ai_pts:
        return 0, 0, None

    # --- load existing annotations ---
    json_path              = _resolve_json(image_path)
    pts, cls, sev, full    = _load_json(json_path)

    user_pts   = [pts[i] for i in range(len(pts))
                  if cls[i].lower() not in _CLEAN_LABELS]
    clean_pts  = [pts[i] for i in range(len(pts))
                  if cls[i].lower() in _CLEAN_LABELS]

    added   = 0
    skipped = 0

    for ai_pt in ai_pts:
        # Skip if close to any real user annotation
        if _any_within(ai_pt, user_pts, radius):
            skipped += 1
            continue
        # Skip if already have a clean/RLClean marker nearby
        if _any_within(ai_pt, clean_pts, radius):
            skipped += 1
            continue
        # Add RLClean
        pts.append(ai_pt)
        cls.append("RLClean")
        sev.append("Unknown")
        clean_pts.append(ai_pt)   # so later points in this frame also see it
        added += 1

    if added > 0:
        _save_json(json_path, full, pts, cls, sev)

    return added, skipped, None


# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------

class RLAnnotatorDialog(wx.Dialog):
    """
    Iterates all images in local_dir, runs the classifier, and writes
    RLClean annotations for confirmed false-positive detections that are
    beyond `radius` pixels from any real or existing clean annotation.
    """

    def __init__(self, parent, classifier, local_dir, radius):
        super().__init__(parent, title="Reinforcement Learning Annotation",
                         size=(560, 310))
        self.classifier = classifier
        self.local_dir  = local_dir
        self.radius     = radius
        self._stop          = False
        self._modal_closed  = False

        vbox = wx.BoxSizer(wx.VERTICAL)

        self.status_lbl = wx.StaticText(
            self, label=f"Scanning '{local_dir}' with radius={radius}px…")
        vbox.Add(self.status_lbl, 0, wx.ALL | wx.EXPAND, 8)

        self.gauge = wx.Gauge(self, range=100)
        vbox.Add(self.gauge, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM | wx.EXPAND, 8)

        self.detail_lbl = wx.StaticText(self, label="")
        vbox.Add(self.detail_lbl, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM | wx.EXPAND, 8)

        self.summary_lbl = wx.StaticText(self, label="")
        vbox.Add(self.summary_lbl, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM | wx.EXPAND, 8)

        btn_row = wx.BoxSizer(wx.HORIZONTAL)
        self.stop_btn  = wx.Button(self, label="Stop")
        self.close_btn = wx.Button(self, wx.ID_CLOSE, label="Close")
        self.close_btn.Disable()
        btn_row.AddStretchSpacer()
        btn_row.Add(self.stop_btn,  0, wx.RIGHT, 8)
        btn_row.Add(self.close_btn, 0)
        vbox.Add(btn_row, 0, wx.ALL | wx.EXPAND, 8)

        self.SetSizer(vbox)

        self.stop_btn.Bind(wx.EVT_BUTTON,  self._on_stop)
        self.close_btn.Bind(wx.EVT_BUTTON, self._on_close)
        self.Bind(wx.EVT_CLOSE, self._on_window_close)

        threading.Thread(target=self._run, daemon=True).start()

    # ------------------------------------------------------------------

    def _ui(self, fn, *args, **kwargs):
        if not self._modal_closed:
            try:
                wx.CallAfter(fn, *args, **kwargs)
            except Exception:
                pass

    def _run(self):
        from readData import list_image_files
        images = list_image_files(self.local_dir)
        total  = len(images)
        if total == 0:
            self._ui(self.status_lbl.SetLabel, "No images found in directory.")
            self._ui(self.close_btn.Enable)
            self._ui(self.stop_btn.Disable)
            return

        total_added   = 0
        total_skipped = 0
        total_errors  = 0

        for i, img_path in enumerate(images):
            if self._stop:
                break

            fname = os.path.basename(img_path)
            self._ui(self.detail_lbl.SetLabel,
                     f"[{i+1}/{total}] {fname}")
            self._ui(self.gauge.SetValue, int((i + 1) / total * 100))

            added, skipped, err = process_frame(img_path, self.classifier, self.radius)
            if err:
                print(f"[RL] {fname}: {err}")
                total_errors += 1
            else:
                total_added   += added
                total_skipped += skipped

            self._ui(self.summary_lbl.SetLabel,
                     f"Added: {total_added}  Skipped: {total_skipped}  Errors: {total_errors}")

        label = ("Stopped early." if self._stop else "Complete.") + \
                f"  Added {total_added} RLClean annotations across {total} frames."
        self._ui(self.status_lbl.SetLabel, label)
        self._ui(self.stop_btn.Disable)
        self._ui(self.close_btn.Enable)

    # ------------------------------------------------------------------

    def _on_stop(self, _evt):
        self._stop = True
        self.stop_btn.Disable()

    def _on_close(self, _evt):
        self._end_modal_safe(wx.ID_CLOSE)

    def _on_window_close(self, _evt):
        self._stop = True
        self._end_modal_safe(wx.ID_CLOSE)

    def _end_modal_safe(self, code):
        if self._modal_closed:
            return
        self._modal_closed = True
        try:
            if self.IsModal():
                self.EndModal(code)
            else:
                self.Destroy()
        except Exception:
            pass
