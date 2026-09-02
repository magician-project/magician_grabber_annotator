#!/usr/bin/env python3
"""
VlmClient — thin wrapper around a gradio-hosted chat VLM (the deepseek-vl2-style
/reset_state -> /transfer_input -> /predict flow used by legacy/vlm_client.py),
used from the Automation tab to ask free-text questions about the current frame
and, when the answer contains coordinates, parse them out for review.

NOTE (see knowledge/PLAN.md, "Tested & rejected"): this VLM was already tried for
defect LOCALIZATION and found unreliable on this imagery (err 84-699px, defaults
to image centre/corner, run-to-run inconsistent) — SAM3 pen-mark detection
(mga.core.auto_annotator) remains the primary auto-annotation path. This client
is for ad-hoc Q&A / experimentation, not a trusted detector.
"""

import os
import re
import tempfile

import cv2

VLM_IP = "147.52.17.119"
VLM_PORT = "8083"


def _sanitize(s):
    """Trim + strip characters that would break downstream display/parsing
    (mirrors legacy/vlm_client.py's sanitize_string)."""
    if not isinstance(s, str):
        raise ValueError("VLM response is not a string.")
    s = s.strip().replace("\\", "")
    s = re.sub(r"[\x00-\x1F\x7F]", "", s)
    return s.replace("\n", " ").replace("\r", "")


# Bracket/paren groups of 2 numbers ("a point": [x, y] / (x, y)) or 4 numbers
# ("a box": [x1, y1, x2, y2]).
_COORD_RE = re.compile(
    r"[\[\(]\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)"
    r"(?:\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?))?\s*[\]\)]")


def parse_grounded_points(text, img_w, img_h):
    """Best-effort extraction of (x, y) pixel points from free-text VLM output.

    Recognises bracket/paren groups of 2 numbers (a point) or 4 numbers (a box,
    reduced to its centre). A group is treated as a 0..1 fraction of (img_w,
    img_h) when every number in it is <= 1, otherwise as already being pixel
    coordinates. The VLM's actual output format/units are not guaranteed —
    this is a heuristic for the common cases, not a protocol."""
    pts = []
    for m in _COORD_RE.finditer(text):
        vals = [float(g) for g in m.groups() if g is not None]
        if len(vals) == 2:
            x, y = vals
        elif len(vals) == 4:
            x = (vals[0] + vals[2]) / 2.0
            y = (vals[1] + vals[3]) / 2.0
        else:
            continue
        if max(abs(v) for v in vals) <= 1.0:
            x, y = x * img_w, y * img_h
        pts.append((x, y))
    return pts


class VlmClient:
    """Talks to a gradio chat-VLM server (reset_state -> transfer_input -> predict).
    Connection is lazy so importing this module never touches the network."""

    def __init__(self, ip=VLM_IP, port=VLM_PORT):
        self.ip = ip
        self.port = port
        self._client = None

    def _ensure(self):
        if self._client is None:
            from gradio_client import Client      # imported lazily (optional dep)
            url = f"http://{self.ip}:{self.port}"
            print(f"[VlmClient] Connecting to {url}")
            self._client = Client(url)

    def ask(self, bgr, question, temperature=0.6, top_p=0.9, max_tokens=200):
        """Send one image + free-text question, return the sanitized text answer."""
        self._ensure()
        from gradio_client import handle_file

        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        cv2.imwrite(path, bgr)
        try:
            self._client.predict(api_name="/reset_state")
            self._client.predict(
                input_images=[handle_file(path)],
                input_text=question,
                api_name="/transfer_input")
            result = self._client.predict(
                chatbot=[],
                temperature=temperature,
                top_p=top_p,
                max_length_tokens=max_tokens,
                repetition_penalty=1.1,
                max_context_length_tokens=4096,
                greek_translation=False,
                api_name="/predict")
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

        try:
            return _sanitize(result[0][0][1])
        except (IndexError, TypeError) as e:
            raise RuntimeError(f"Unexpected VLM response structure: {e} ({result!r})")
