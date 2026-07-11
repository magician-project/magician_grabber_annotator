#!/usr/bin/python3
"""
classifierGrading.py - Correlates AI classifier output with user ground-truth.
"""

import math
from collections import defaultdict


# ---------------------------------------------------------------------------
# Module-level helper so it is usable without an instance
# ---------------------------------------------------------------------------

def _metrics(tp, fp, fn):
    """Return (precision, recall, f1) from raw counts."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def _new_bucket():
    return {"tp": 0, "fp": 0, "fn": 0}


# ---------------------------------------------------------------------------

class AnnotationCorrelationStats:
    """
    Correlates AI classifier annotations with user ground-truth annotations.

    Handles two model types:
        • "binary_xxxxx"   → binary defect / clean stats
        • "allclass_xxxxx" → per-defect-type and per-severity-class stats

    ``update()`` accepts user classes in human-readable form
    ("Positive Dent", severity "Class A") and AI classes in the compact label
    form ("class_PositiveDentClassA").  Both are normalised internally so
    comparisons are meaningful.

    Call ``reset()`` when switching models.
    """

    def __init__(self, classifier_name: str, hit_radius: int = 40):
        self.classifier_name = classifier_name.lower()
        self.hit_radius      = hit_radius
        self.reset()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _distance(p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def _match_ai_to_user(self, ai_pts, user_pts):
        """
        Nearest-neighbour greedy matching within hit_radius.

        All candidate (ai, user) pairs within the radius are sorted by
        distance; pairs are accepted in that order, ensuring each point
        appears in at most one match (closest pair wins).

        Returns:
            matches        – list of (ai_idx, user_idx)
            unmatched_ai   – set of AI indices without a match
            unmatched_user – set of user indices without a match
        """
        candidates = []
        for ai_i, ai_pt in enumerate(ai_pts):
            for u_i, u_pt in enumerate(user_pts):
                d = self._distance(ai_pt, u_pt)
                if d <= self.hit_radius:
                    candidates.append((d, ai_i, u_i))
        candidates.sort()

        matched_ai   = set()
        matched_user = set()
        matches      = []
        for _, ai_i, u_i in candidates:
            if ai_i not in matched_ai and u_i not in matched_user:
                matches.append((ai_i, u_i))
                matched_ai.add(ai_i)
                matched_user.add(u_i)

        unmatched_ai   = set(range(len(ai_pts)))   - matched_ai
        unmatched_user = set(range(len(user_pts))) - matched_user
        return matches, unmatched_ai, unmatched_user

    def _combine_class_severity(self, cls, sev):
        """
        "Positive Dent" + "Class A"  →  "class_positivedentclassa"
        Returns None when cls is blank.
        """
        cls = (cls or "").strip()
        sev = (sev or "").strip()
        if not cls:
            return None
        base = cls.lower().replace(" ", "")
        if sev:
            return f"class_{base}{sev.lower().replace(' ', '')}"
        return f"class_{base}"

    def parse_label(self, label):
        """
        "class_positivedentclassa"  →  ("positivedent", "a")
        "class_clean"               →  ("clean",        None)
        Returns (None, None) on parse failure.
        """
        if not label or not label.startswith("class_"):
            return None, None
        body = label[len("class_"):]       # e.g. "positivedentclassa"
        idx  = body.rfind("class")
        if idx == -1:
            return body, None              # label has no severity part
        return body[:idx], body[idx + len("class"):]

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset(self):
        """Reset all accumulated statistics (call when switching models)."""
        self.frames_processed = 0
        self._hz_sum   = 0.0
        self._hz_count = 0

        self.binary_stats = _new_bucket()

        self.multi_stats = {
            "by_defect_class": defaultdict(_new_bucket),
            "by_defect":       defaultdict(_new_bucket),
            "by_class":        defaultdict(_new_bucket),
        }

    # ------------------------------------------------------------------
    # Update — primary API called from wxAnnotator
    # ------------------------------------------------------------------

    def update(self, frame_id, user_ann, ai_ann, hz=0.0):
        """
        Update statistics for one frame.

        user_ann  –  {"points": [(x,y)…], "classes": ["Positive Dent"…],
                      "severities": ["Class A"…]}
        ai_ann    –  {"points": [(x,y)…], "classes": ["class_PositiveDentClassA"…]}
                     May be None (classifier produced no output).
        hz        –  Classifier throughput for this frame (frames/sec).
        """
        if ai_ann is None:
            ai_ann = {"points": [], "classes": []}

        self.frames_processed += 1
        if hz > 0:
            self._hz_sum   += hz
            self._hz_count += 1

        u_pts = user_ann.get("points", [])
        a_pts = ai_ann.get("points", [])

        # Normalise user labels: "Positive Dent" + "Class A" → "class_positivedentclassa"
        raw_classes = user_ann.get("classes", [])
        raw_sevs    = user_ann.get("severities", [])
        u_labels = []
        for i, c in enumerate(raw_classes):
            s      = raw_sevs[i] if i < len(raw_sevs) else ""
            merged = self._combine_class_severity(c, s)
            u_labels.append((merged or c).lower().strip())

        a_labels = [c.lower().strip() for c in ai_ann.get("classes", [])]

        if self.classifier_name.startswith("binary_"):
            self._update_binary(u_pts, a_pts)
            return

        self._update_multiclass(u_pts, u_labels, a_pts, a_labels)

    def _update_binary(self, u_pts, a_pts):
        """Binary (defect-present / clean) stats for one frame."""
        if not u_pts and not a_pts:
            self.binary_stats["tp"] += 1
            return
        matches, unmatched_ai, unmatched_user = self._match_ai_to_user(a_pts, u_pts)
        self.binary_stats["tp"] += len(matches)
        self.binary_stats["fp"] += len(unmatched_ai)
        self.binary_stats["fn"] += len(unmatched_user)

    def _update_multiclass(self, u_pts, u_labels, a_pts, a_labels):
        """Multi-class stats for one frame."""
        # Both empty → clean frame TP
        if not u_pts and not a_pts:
            for bucket in self.multi_stats.values():
                bucket["clean"]["tp"] += 1
            return

        # Parse normalised labels into (defect_type, severity_class)
        u_parsed = [self.parse_label(L) for L in u_labels]
        a_parsed = [self.parse_label(L) for L in a_labels]
        u_def = [p[0] for p in u_parsed]
        u_cls = [p[1] for p in u_parsed]
        a_def = [p[0] for p in a_parsed]
        a_cls = [p[1] for p in a_parsed]

        matches, unmatched_ai, unmatched_user = self._match_ai_to_user(a_pts, u_pts)

        dc = self.multi_stats["by_defect_class"]
        dd = self.multi_stats["by_defect"]
        cc = self.multi_stats["by_class"]

        for ai_i, u_i in matches:
            u_dc = u_labels[u_i]         if u_i  < len(u_labels) else None
            a_dc = a_labels[ai_i]        if ai_i < len(a_labels) else None
            u_d  = u_def[u_i]            if u_i  < len(u_def)    else None
            a_d  = a_def[ai_i]           if ai_i < len(a_def)    else None
            u_c  = u_cls[u_i]            if u_i  < len(u_cls)    else None
            a_c  = a_cls[ai_i]           if ai_i < len(a_cls)    else None

            _tally(dc, u_dc, a_dc)
            _tally(dd, u_d,  a_d)
            _tally(cc, u_c,  a_c)

        for u_i in unmatched_user:
            if u_i < len(u_labels): dc[u_labels[u_i]]["fn"] += 1
            if u_i < len(u_def):    dd[u_def[u_i]]["fn"]    += 1
            if u_i < len(u_cls):    cc[u_cls[u_i]]["fn"]    += 1

        for ai_i in unmatched_ai:
            if ai_i < len(a_labels): dc[a_labels[ai_i]]["fp"] += 1
            if ai_i < len(a_def):    dd[a_def[ai_i]]["fp"]    += 1
            if ai_i < len(a_cls):    cc[a_cls[ai_i]]["fp"]    += 1

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    @property
    def avg_hz(self):
        """Average classifier throughput in Hz over all recorded frames."""
        return self._hz_sum / self._hz_count if self._hz_count > 0 else 0.0

    def get_stats(self):
        """Return raw accumulated statistics dict."""
        if self.classifier_name.startswith("binary_"):
            return self.binary_stats
        return self.multi_stats

    def get_summary_string(self):
        """Return a compact single-line summary suitable for a UI label."""
        hz_str = f"  {self.avg_hz:.2f} Hz" if self._hz_count > 0 else ""

        if self.classifier_name.startswith("binary_"):
            tp = self.binary_stats["tp"]
            fp = self.binary_stats["fp"]
            fn = self.binary_stats["fn"]
            p, r, f1 = _metrics(tp, fp, fn)
            return (f"Frames:{self.frames_processed}{hz_str}  "
                    f"TP={tp} FP={fp} FN={fn}  "
                    f"P={p:.2f} R={r:.2f} F1={f1:.2f}")

        # Macro-average over defect types
        rows = [
            (s["tp"], s["fp"], s["fn"])
            for s in self.multi_stats["by_defect"].values()
            if s["tp"] + s["fp"] + s["fn"] > 0
        ]
        if not rows:
            return f"Frames:{self.frames_processed}{hz_str}  (no data yet)"

        ps, rs, f1s = zip(*[_metrics(*r) for r in rows])
        n = len(rows)
        return (f"Frames:{self.frames_processed}{hz_str}  "
                f"Macro P={sum(ps)/n:.2f} R={sum(rs)/n:.2f} F1={sum(f1s)/n:.2f}  "
                f"({n} defect types)")

    def format_stats(self):
        """Return a full formatted statistics string."""
        hz_line = (f" Avg throughput:   {self.avg_hz:.2f} Hz"
                   f"  (over {self._hz_count} frames)")
        lines = [
            "",
            "=" * 58,
            f" Annotation Statistics  [{self.classifier_name}]",
            f" Frames processed: {self.frames_processed}",
            hz_line,
        ]
        run_info = getattr(self, "run_info", "")
        if run_info:
            lines.append(f" Run settings:     {run_info}")
        lines.append("=" * 58)

        if self.classifier_name.startswith("binary_"):
            tp = self.binary_stats["tp"]
            fp = self.binary_stats["fp"]
            fn = self.binary_stats["fn"]
            p, r, f1 = _metrics(tp, fp, fn)
            lines += [
                "\n[Binary Mode]",
                f"  TP={tp}  FP={fp}  FN={fn}",
                f"  Precision={p:.3f}  Recall={r:.3f}  F1={f1:.3f}",
                "",
            ]
            return "\n".join(lines)

        def _section(title, stats_dict):
            out = [f"\n--- {title} ---"]
            all_p, all_r, all_f1 = [], [], []
            for name, s in sorted(stats_dict.items(), key=lambda kv: str(kv[0])):
                tp, fp, fn = s["tp"], s["fp"], s["fn"]
                if tp + fp + fn == 0:
                    continue
                p, r, f1 = _metrics(tp, fp, fn)
                out.append(
                    f"  {str(name):<34}  "
                    f"TP={tp:4d}  FP={fp:4d}  FN={fn:4d}  "
                    f"P={p:.3f}  R={r:.3f}  F1={f1:.3f}"
                )
                all_p.append(p); all_r.append(r); all_f1.append(f1)
            if all_p:
                n = len(all_p)
                out.append(
                    f"  {'[macro avg]':<34}  "
                    f"{'':>28}"
                    f"P={sum(all_p)/n:.3f}  R={sum(all_r)/n:.3f}  F1={sum(all_f1)/n:.3f}"
                )
            return out

        lines += _section("Per Defect Type",         self.multi_stats["by_defect"])
        lines += _section("Per Severity Class",      self.multi_stats["by_class"])
        lines += _section("Per Defect+Class Label",  self.multi_stats["by_defect_class"])
        lines.append("")
        return "\n".join(lines)

    def print_stats(self):
        """Print formatted statistics to stdout."""
        print(self.format_stats())


# ---------------------------------------------------------------------------
# Helper used inside _update_multiclass — avoids repetition
# ---------------------------------------------------------------------------

def _tally(bucket_dict, user_key, ai_key):
    """Increment TP/FP/FN buckets for one matched pair."""
    if user_key == ai_key:
        bucket_dict[user_key]["tp"] += 1
    else:
        if user_key is not None:
            bucket_dict[user_key]["fn"] += 1
        if ai_key is not None:
            bucket_dict[ai_key]["fp"] += 1
