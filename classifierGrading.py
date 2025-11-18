import math
from collections import defaultdict


class AnnotationCorrelationStats:
    """
    Correlates AI classifier annotations with user ground-truth annotations.

    Handles two model types:
        • "binary_xxxxx"  → match on class_defect only
        • "allclass_xxxxx" → keep separate stats for defect type AND severity

    Features:
        • Case-insensitive comparison
        • Configurable radius (hit distance)
        • Maintains cumulative statistics over many frames
        • Supports reset() when switching models
    """

    def __init__(self, classifier_name: str, hit_radius: int = 40):
        self.classifier_name = classifier_name.lower()
        self.hit_radius = hit_radius

        self.reset()
        """
        # For "binary_xxxxx"
        self.binary_stats = {
            "tp": 0,   # true positives (found inside hit region)
            "fp": 0,   # false positives (AI predicted defect where none exists)
            "fn": 0    # false negatives (AI failed to detect user defect)
        }

        # For "allclass_xxxxx"
        self.multi_stats = {
            "by_class": defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0}),
            "by_severity": defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0}),
        }
        """

    # -----------------------------
    # Utility
    # -----------------------------
    @staticmethod
    def _distance(p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def _match_ai_to_user(self, ai_pts, user_pts):
        """
        Returns:
            matches: list of (ai_idx, user_idx)
            unmatched_ai: set
            unmatched_user: set
        """

        matches = []
        unmatched_ai = set(range(len(ai_pts)))
        unmatched_user = set(range(len(user_pts)))

        for ai_i, ai_pt in enumerate(ai_pts):
            for user_i, user_pt in enumerate(user_pts):
                dist = self._distance(ai_pt, user_pt)
                if dist <= self.hit_radius:
                    matches.append((ai_i, user_i))
                    unmatched_ai.discard(ai_i)
                    unmatched_user.discard(user_i)
                    break  # AI tile can only match one user tile

        return matches, unmatched_ai, unmatched_user

    # -----------------------------
    # -----------------------------

    def _combine_class_severity(self, cls, sev):
        """
        Combines user class + severity into the same format the AI outputs.
        """
        if cls is None or cls.strip() == "":
            return None
        if sev is None or sev.strip() == "":
            return None

        # Canonical format used by AI:
        # class_<classname without spaces><severity>
        base = cls.lower().replace(" ", "")
        sev  = sev.lower().replace(" ", "")
        return f"class_{base}{sev}"

    # -----------------------------
    # Public API
    # -----------------------------
    def updatePerDefect(self, frame_id, user_ann, ai_ann):
        """
        user_ann = {
            "points": [(x,y), ...],
            "classes": ["Positive Dent", ...],
            "severities": ["Class A", ...]
        }

        ai_ann = {
            "points": [(x,y), ...],
            "classes": [...],
            "severities": [...],     # only for allclass models
        }
        """
        if (ai_ann is None):
               return

        u_pts = user_ann.get("points", [])
        a_pts = ai_ann.get("points", [])

        #u_cls = [c.lower() for c in user_ann.get("classes", [])]
        #u_sev = [s.lower() for s in user_ann.get("severities", [])]


        raw_classes = user_ann.get("classes", [])
        raw_sevs    = user_ann.get("severities", [])

        u_cls = []
        u_sev = []

        u_combined = []   # merged class_severity labels

        for c, s in zip(raw_classes, raw_sevs):
            c = c.lower()
            s = s.lower()
            u_cls.append(c)
            u_sev.append(s)
            merged = self._combine_class_severity(c, s)
            u_combined.append(merged)

        a_cls = [c.lower() for c in ai_ann.get("classes", [])]
        a_sev = [s.lower() for s in ai_ann.get("severities", [])]

        matches, unmatched_ai, unmatched_user = self._match_ai_to_user(a_pts, u_pts)

        # --- Binary mode: only defect vs clean ---
        if self.classifier_name.startswith("binary_"):
            for ai_i, user_i in matches:
                if a_cls[ai_i] == u_cls[user_i]:
                    self.binary_stats["tp"] += 1
                else:
                    self.binary_stats["fp"] += 1

            # AI detected something but no user defect
            self.binary_stats["fp"] += len(unmatched_ai)

            # User defect not detected by AI
            self.binary_stats["fn"] += len(unmatched_user)

        # --- Multi-class mode ---
        elif self.classifier_name.startswith("allclass_"):
            # matched items
            for ai_i, user_i in matches:
                u_class = u_cls[user_i]
                a_class = a_cls[ai_i]
                u_sev   = u_sev[user_i]
                a_sev   = a_sev[ai_i] if ai_i < len(a_sev) else "unknown"

                if a_class == u_class:
                    self.multi_stats["by_class"][u_class]["tp"] += 1
                else:
                    self.multi_stats["by_class"][u_class]["fn"] += 1
                    self.multi_stats["by_class"][a_class]["fp"] += 1

                if a_sev == u_sev:
                    self.multi_stats["by_severity"][u_sev]["tp"] += 1
                else:
                    self.multi_stats["by_severity"][u_sev]["fn"] += 1
                    self.multi_stats["by_severity"][a_sev]["fp"] += 1

            # unmatched AI → false positives
            for ai_i in unmatched_ai:
                cls = a_cls[ai_i]
                self.multi_stats["by_class"][cls]["fp"] += 1

                if ai_i < len(a_sev):
                    self.multi_stats["by_severity"][a_sev[ai_i]]["fp"] += 1

            # unmatched user → false negatives
            for user_i in unmatched_user:
                cls = u_cls[user_i]
                self.multi_stats["by_class"][cls]["fn"] += 1

                if user_i < len(u_sev):
                    self.multi_stats["by_severity"][u_sev[user_i]]["fn"] += 1



    def parse_label(self,label):
        """
        Input:  "class_positivedentclassa"
        Output: ("positivedent", "a")
        """
        if not label.startswith("class_"):
            return None, None

        body = label[len("class_"):]    # "positivedentclassa"

        # split into defect part + class part
        # find last occurrence of "class"
        idx = body.rfind("class")
        if idx == -1:
            return None, None

        defect = body[:idx]             # "positivedent"
        clazz  = body[idx+len("class"):]  # "a"

        print("parse label resolved ",label," to defect=",defect," class=",clazz)
        return defect, clazz

    def update(self, frame_id, user_ann, ai_ann):
        """
        user_ann = {
            "points": [(x,y), ...],
            "classes": ["class_positivedentclassa", ...]
        }

        ai_ann = same structure.
        """

        # ------------------------------------------------------------
        # Handle missing AI annotations (clean frame)
        # ------------------------------------------------------------
        if ai_ann is None:
            ai_ann = {
                "points": [],
                "classes": []
            }

        u_pts = user_ann.get("points", [])
        a_pts = ai_ann.get("points", [])

        # Keep original labels EXACTLY
        u_labels = [c.lower().strip() for c in user_ann.get("classes", [])]
        a_labels = [c.lower().strip() for c in ai_ann.get("classes", [])]

        # Parse user labels
        u_def = []
        u_class = []
        for L in u_labels:
            d, c = self.parse_label(L)
            u_def.append(d)
            u_class.append(c)

        # Parse AI labels
        a_def = []
        a_class = []
        for L in a_labels:
            d, c = self.parse_label(L)
            a_def.append(d)
            a_class.append(c)

        # defect_class stays EXACTLY as given
        u_defclass = u_labels
        a_defclass = a_labels

        # ------------------------------------------------------------
        # CASE 1: No defects at all → Clean frame
        # ------------------------------------------------------------
        if len(u_pts) == 0 and len(a_pts) == 0:
            self.multi_stats["by_defect_class"]["clean"]["tp"] += 1
            self.multi_stats["by_defect"]["clean"]["tp"] += 1
            self.multi_stats["by_class"]["clean"]["tp"] += 1
            return

        # ------------------------------------------------------------
        # MATCH AI → USER by positions
        # ------------------------------------------------------------
        matches, unmatched_ai, unmatched_user = \
            self._match_ai_to_user(a_pts, u_pts)

        dc_stats = self.multi_stats["by_defect_class"]
        d_stats  = self.multi_stats["by_defect"]
        c_stats  = self.multi_stats["by_class"]

        # ------------------------------------------------------------
        # POINT-LEVEL MATCHES
        # ------------------------------------------------------------
        for ai_i, u_i in matches:

            # Expected (user)
            u_dc = u_defclass[u_i]
            u_d  = u_def[u_i]
            u_c  = u_class[u_i]

            # Predicted (AI)
            a_dc = a_defclass[ai_i]
            a_d  = a_def[ai_i]
            a_c  = a_class[ai_i]

            # ----- defect_class -----
            if u_dc == a_dc:
                dc_stats[u_dc]["tp"] += 1
            else:
                dc_stats[u_dc]["fn"] += 1
                dc_stats[a_dc]["fp"] += 1

            # ----- defect only -----
            if u_d == a_d:
                d_stats[u_d]["tp"] += 1
            else:
                d_stats[u_d]["fn"] += 1
                d_stats[a_d]["fp"] += 1

            # ----- class only -----
            if u_c == a_c:
                c_stats[u_c]["tp"] += 1
            else:
                c_stats[u_c]["fn"] += 1
                c_stats[a_c]["fp"] += 1

        # ------------------------------------------------------------
        # FRAME-LEVEL FN (user defect missing)
        # ------------------------------------------------------------
        for u_i in unmatched_user:
            dc_stats[u_defclass[u_i]]["fn"] += 1
            d_stats[u_def[u_i]]["fn"] += 1
            c_stats[u_class[u_i]]["fn"] += 1

        # ------------------------------------------------------------
        # FRAME-LEVEL FP (false AI detection)
        # ------------------------------------------------------------
        for ai_i in unmatched_ai:
            dc_stats[a_defclass[ai_i]]["fp"] += 1
            d_stats[a_def[ai_i]]["fp"] += 1
            c_stats[a_class[ai_i]]["fp"] += 1


    def updateOLD(self, frame_id, user_ann, ai_ann):
        """
        user_ann = {
            "points": [(x,y), ...],
            "classes": [...],
            "severities": [...]
        }

        ai_ann = {
            "points": [(x,y), ...],
            "classes": [...],
            "severities": [...]
        }
        """

        # ------------------------------------------------------------
        # Handle missing AI annotations (clean frame)
        # ------------------------------------------------------------
        if ai_ann is None:
            ai_ann = {
                "points": [],
                "classes": [],
                "severities": []
            }

        print("AI Annotations ", ai_ann)
        print("User Annotations ", user_ann)

        if (len(ai_ann["points"])==0) and (len(user_ann["points"])==0):
           print("Both annotations and classifier are empty, success!")

        u_pts = user_ann.get("points", [])
        a_pts = ai_ann.get("points", [])

        # Normalize labels
        u_cls = [c.lower() for c in user_ann.get("classes", [])]
        u_sev = [s.lower() for s in user_ann.get("severities", [])]

        a_cls = [c.lower() for c in ai_ann.get("classes", [])]
        a_sev = [s.lower() for s in ai_ann.get("severities", [])]

        # ------------------------------------------------------------
        # CASE 1: No AI defects AND no user defects  → Correct frame
        # ------------------------------------------------------------
        if len(a_pts) == 0 and len(u_pts) == 0:
            if self.classifier_name.startswith("binary_"):
                # For binary classifiers, count as a TP frame
                self.binary_stats["tp"] += 1
            else:
                # For allclass classifiers, count as TP on a virtual "clean" class
                class_stats = self.multi_stats["by_class"]
                class_stats["clean"]["tp"] += 1
            return


        # --- Otherwise, normal correlation ---
        matches, unmatched_ai, unmatched_user = \
            self._match_ai_to_user(a_pts, u_pts)

        # ------------------------------------------------------------
        # BINARY CLASSIFIER LOGIC
        # ------------------------------------------------------------
        if self.classifier_name.startswith("binary_"):

            # If any match → TP for the frame
            if len(matches) > 0:
                self.binary_stats["tp"] += 1

            # If user has defects that AI missed → FN (once per frame)
            if len(unmatched_user) > 0 and len(u_pts) > 0:
                self.binary_stats["fn"] += 1

            # If AI predicted defects with no user → FP (once per frame)
            if len(unmatched_ai) > 0 and len(a_pts) > 0:
                self.binary_stats["fp"] += 1

            return

        # ------------------------------------------------------------
        # MULTI-CLASS CLASSIFIER LOGIC
        # ------------------------------------------------------------
        class_stats = self.multi_stats["by_class"]
        sev_stats   = self.multi_stats["by_severity"]

        # --- Generate combined user labels ---
        # class + severity => "class_negativedentclassa"
        u_combined = []
        for c, s in zip(u_cls, u_sev):
            if c and s:
                merged = f"class_{c.replace(' ', '')}{s.replace(' ', '')}"
            else:
                merged = None
            u_combined.append(merged)

        # --- Count matched tiles (individual) ---
        for ai_i, user_i in matches:
            user_c = u_combined[user_i]   # <<< FIXED
            ai_c   = a_cls[ai_i]

            user_s = u_sev[user_i]
            ai_s   = a_sev[ai_i] if ai_i < len(a_sev) else None

            # CLASS evaluation
            if user_c == ai_c:
                class_stats[user_c]["tp"] += 1
            else:
                class_stats[user_c]["fn"] += 1
                class_stats[ai_c]["fp"] += 1

            # SEVERITY evaluation
            if ai_s == user_s:
                sev_stats[user_s]["tp"] += 1
            else:
                sev_stats[user_s]["fn"] += 1
                if ai_s:
                    sev_stats[ai_s]["fp"] += 1

        # ------------------------------------------------------------
        # Frame-level FN (once per frame)
        # ------------------------------------------------------------
        if len(unmatched_user) > 0 and len(u_pts) > 0:
            added = set()
            for user_i in unmatched_user:
                uc = u_combined[user_i]   # <<< FIXED
                us = u_sev[user_i]

                if ("c", uc) not in added:
                    class_stats[uc]["fn"] += 1
                if ("s", us) not in added:
                    sev_stats[us]["fn"] += 1

                added.add(("c", uc))
                added.add(("s", us))


        # ------------------------------------------------------------
        # Frame-level FP (once per frame)
        # ------------------------------------------------------------
        if len(unmatched_ai) > 0 and len(a_pts) > 0:
            added = set()
            for ai_i in unmatched_ai:
                ac = a_cls[ai_i]
                asv = a_sev[ai_i] if ai_i < len(a_sev) else None
                if ("c", ac) not in added:
                    class_stats[ac]["fp"] += 1
                if asv is not None and ("s", asv) not in added:
                    sev_stats[asv]["fp"] += 1
                added.add(("c", ac))
                if asv is not None:
                    added.add(("s", asv))


    def reset(self):
        """Reset accumulated statistics (e.g. when changing model)."""
        self.binary_stats = {"tp": 0, "fp": 0, "fn": 0}

        self.multi_stats = {
            "by_defect_class": defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0}),
            "by_defect":       defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0}),
            "by_class":        defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0}),
            "by_severity":        defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0}),
        }

    def resetOLD(self):
        """Reset accumulated statistics (e.g. when changing model)."""
        self.binary_stats = {"tp": 0, "fp": 0, "fn": 0}
        self.multi_stats = {
            "by_class": defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0}),
            "by_severity": defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0}),
        }

    def get_stats(self):
        """Return current accumulated statistics."""
        if self.classifier_name.startswith("binary_"):
            return self.binary_stats
        else:
            return self.multi_stats


    def print_stats(self):
        """Prints human-readable statistics to stdout."""
        print("\n====================")
        print(" Annotation Statistics")
        print("====================")

        print("\n====================")
        print(" UNDER CONSTRUCTION Statistics are not perfect yet")
        print("====================")

        # -----------------------
        # BINARY CLASSIFIER STATS
        # -----------------------
        if self.classifier_name.startswith("binary_"):
            tp = self.binary_stats["tp"]
            fp = self.binary_stats["fp"]
            fn = self.binary_stats["fn"]

            print("\n[Binary Classifier Mode]\n")
            print(f"True Positives : {tp}")
            print(f"False Positives: {fp}")
            print(f"False Negatives: {fn}")

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

            print("\nMetrics:")
            print(f" Precision: {precision:.3f}")
            print(f" Recall   : {recall:.3f}")
            print(f" F1 Score : {f1:.3f}")

            print("\n====================\n")
            return

        # ------------------------------------------------------------
        # MULTI-CLASS CLASSIFIER STATS
        # ------------------------------------------------------------
        print("\n[Multi-Class Classifier Mode]")

        # ------------------------------------------------------------
        # BY CLASS
        # ------------------------------------------------------------
        print("\n--- Per Class (a, b, c...) ---")
        for cls, stats in self.multi_stats["by_class"].items():
            tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
            if tp == fp == fn == 0:
                continue

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

            print(f"\nClass: {cls}")
            print(f"  TP={tp}, FP={fp}, FN={fn}")
            print(f"    Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")

        # ------------------------------------------------------------
        # BY DEFECT (positivedent, negativedent, welding...)
        # ------------------------------------------------------------
        print("\n--- Per Defect (dent, welding...) ---")
        for defect, stats in self.multi_stats["by_defect"].items():
            tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
            if tp == fp == fn == 0:
                continue

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

            print(f"\nDefect: {defect}")
            print(f"  TP={tp}, FP={fp}, FN={fn}")
            print(f"    Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")

        # ------------------------------------------------------------
        # BY DEFECT_CLASS (original full string: class_positivedentclassa)
        # ------------------------------------------------------------
        print("\n--- Per Defect-Class (full label) ---")
        for dc, stats in self.multi_stats["by_defect_class"].items():
            tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
            if tp == fp == fn == 0:
                continue

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

            print(f"\nDefect-Class: {dc}")
            print(f"  TP={tp}, FP={fp}, FN={fn}")
            print(f"    Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")

        print("\n====================\n")


