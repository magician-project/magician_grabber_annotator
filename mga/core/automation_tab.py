#!/usr/bin/python3

"""
Author : "Ammar Qammaz"
Copyright : "2025 Foundation of Research and Technology, Computer Science Department Greece"
License : "FORTH"

Automation tab for the annotator GUI — SAM3 segmentation server/keyword/knobs
(mga.core.auto_annotator.AutoAnnotator) and a VLM "ask a grounded question"
panel (mga.core.vlm_client.VlmClient), following the ClassifierTabMixin
pattern: PhotoCtrl mixes this in, widgets live on self, and self._WA() (from
ClassifierTabMixin) resolves the wx_annotator module for shared names
(options, severities, SAM3_IP, ...).
"""

import os

import cv2
import wx

from mga.core.annotation_state import add_point
from mga.core.vlm_client import VlmClient, VLM_IP, VLM_PORT, parse_grounded_points

try:
    from mga.core.auto_annotator import (AutoAnnotator, polar_raw_to_bgr,
                                          SAM3_IP, SAM3_PORT, DEFAULT_PROMPT,
                                          REPRESENTATION, MOSAIC_SCALE)
except Exception:
    AutoAnnotator = None
    SAM3_IP, SAM3_PORT, DEFAULT_PROMPT, REPRESENTATION, MOSAIC_SCALE = (
        "139.91.185.16", "7860", "drawn circle", "clahe", 2.0)


class AutomationTabMixin:
    """Builds the Automation tab: SAM3 server/prompt/knobs + VLM ask panel."""

    def _buildAutomationTab(self, parent):
        s = wx.BoxSizer(wx.VERTICAL)

        # ===================== SAM3 section =====================
        s.Add(wx.StaticText(parent, label="SAM3 Segmentation Server"), 0, wx.ALL, 5)

        ipRow = wx.BoxSizer(wx.HORIZONTAL)
        ipRow.Add(wx.StaticText(parent, label="IP:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 4)
        self.samIpCtrl = wx.TextCtrl(parent, value=SAM3_IP)
        ipRow.Add(self.samIpCtrl, 1, wx.RIGHT, 8)
        ipRow.Add(wx.StaticText(parent, label="Port:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 4)
        self.samPortCtrl = wx.TextCtrl(parent, value=str(SAM3_PORT), size=(70, -1))
        ipRow.Add(self.samPortCtrl, 0)
        s.Add(ipRow, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM | wx.EXPAND, 5)

        s.Add(wx.StaticText(parent, label="Segmentation keyword / prompt"),
              0, wx.LEFT | wx.RIGHT | wx.TOP, 5)
        self.samPromptCtrl = wx.TextCtrl(parent, value=DEFAULT_PROMPT)
        self.samPromptCtrl.SetToolTip(
            "Text prompt sent to SAM3. 'drawn circle' is the validated working "
            "prompt — generic prompts ('circle', 'pen mark', 'defect', ...) "
            "mostly return an empty mask.")
        s.Add(self.samPromptCtrl, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM | wx.EXPAND, 5)

        reprRow = wx.BoxSizer(wx.HORIZONTAL)
        reprRow.Add(wx.StaticText(parent, label="Representation:"),
                    0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 6)
        self.samRepresentationCombo = wx.ComboBox(
            parent, choices=["clahe", "gray", "rgb"], style=wx.CB_READONLY)
        self.samRepresentationCombo.SetValue(REPRESENTATION)
        self.samRepresentationCombo.SetToolTip(
            "How the raw 4-channel polarisation frame is rendered before being "
            "sent to SAM3/VLM: clahe = CLAHE-enhanced gray (faint marks pop), "
            "gray = averaged polarisation, rgb = 0/45/90deg -> B/G/R false colour.")
        reprRow.Add(self.samRepresentationCombo, 0)
        s.Add(reprRow, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 5)

        self.samUseDolpCheckbox = wx.CheckBox(
            parent, label="Also segment DoLP map (union) — better recall")
        self.samUseDolpCheckbox.SetValue(True)
        s.Add(self.samUseDolpCheckbox, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 5)

        self.samRejectBorderCheckbox = wx.CheckBox(
            parent, label="Reject border rings without a DoLP anomaly")
        self.samRejectBorderCheckbox.SetValue(True)
        s.Add(self.samRejectBorderCheckbox, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 5)

        self.samRefineCheckbox = wx.CheckBox(
            parent, label="Refine point onto DoLP defect anomaly")
        self.samRefineCheckbox.SetValue(True)
        s.Add(self.samRefineCheckbox, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 5)

        knobRow1 = wx.BoxSizer(wx.HORIZONTAL)
        knobRow1.Add(wx.StaticText(parent, label="Min area (px):"),
                     0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 4)
        self.samMinAreaCtrl = wx.TextCtrl(parent, value="3000", size=(60, -1))
        knobRow1.Add(self.samMinAreaCtrl, 0, wx.RIGHT, 10)
        knobRow1.Add(wx.StaticText(parent, label="Merge dist (px):"),
                     0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 4)
        self.samMergeDistCtrl = wx.TextCtrl(parent, value="60", size=(50, -1))
        knobRow1.Add(self.samMergeDistCtrl, 0)
        s.Add(knobRow1, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 5)

        knobRow2 = wx.BoxSizer(wx.HORIZONTAL)
        knobRow2.Add(wx.StaticText(parent, label="Max area frac:"),
                     0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 4)
        self.samMaxAreaFracCtrl = wx.TextCtrl(parent, value="0.18", size=(55, -1))
        knobRow2.Add(self.samMaxAreaFracCtrl, 0, wx.RIGHT, 10)
        knobRow2.Add(wx.StaticText(parent, label="Max aspect:"),
                     0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 4)
        self.samMaxAspectCtrl = wx.TextCtrl(parent, value="4.0", size=(50, -1))
        knobRow2.Add(self.samMaxAspectCtrl, 0)
        s.Add(knobRow2, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 5)

        self.samApplyBtn = wx.Button(parent, label="Apply / Reconnect SAM3")
        self.samApplyBtn.Bind(wx.EVT_BUTTON, self.onApplySamSettings)
        s.Add(self.samApplyBtn, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM | wx.EXPAND, 5)

        self.samStatusLabel = wx.StaticText(parent, label="Using defaults — connects on first Auto/Track use.")
        s.Add(self.samStatusLabel, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM | wx.EXPAND, 5)

        s.Add(wx.StaticLine(parent), 0, wx.ALL | wx.EXPAND, 8)

        # ===================== VLM section =====================
        s.Add(wx.StaticText(parent, label="VLM — Ask a Grounded Question"), 0, wx.ALL, 5)

        vlmIpRow = wx.BoxSizer(wx.HORIZONTAL)
        vlmIpRow.Add(wx.StaticText(parent, label="IP:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 4)
        self.vlmIpCtrl = wx.TextCtrl(parent, value=VLM_IP)
        vlmIpRow.Add(self.vlmIpCtrl, 1, wx.RIGHT, 8)
        vlmIpRow.Add(wx.StaticText(parent, label="Port:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 4)
        self.vlmPortCtrl = wx.TextCtrl(parent, value=str(VLM_PORT), size=(70, -1))
        vlmIpRow.Add(self.vlmPortCtrl, 0)
        s.Add(vlmIpRow, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM | wx.EXPAND, 5)

        paramRow = wx.BoxSizer(wx.HORIZONTAL)
        paramRow.Add(wx.StaticText(parent, label="Temp:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 4)
        self.vlmTemperatureCtrl = wx.TextCtrl(parent, value="0.6", size=(45, -1))
        paramRow.Add(self.vlmTemperatureCtrl, 0, wx.RIGHT, 8)
        paramRow.Add(wx.StaticText(parent, label="Top-p:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 4)
        self.vlmTopPCtrl = wx.TextCtrl(parent, value="0.9", size=(45, -1))
        paramRow.Add(self.vlmTopPCtrl, 0, wx.RIGHT, 8)
        paramRow.Add(wx.StaticText(parent, label="Max tokens:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 4)
        self.vlmMaxTokensCtrl = wx.TextCtrl(parent, value="200", size=(50, -1))
        paramRow.Add(self.vlmMaxTokensCtrl, 0)
        s.Add(paramRow, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 5)

        s.Add(wx.StaticText(parent, label="Question (asked about the current frame)"),
              0, wx.LEFT | wx.RIGHT | wx.TOP, 5)
        self.vlmQuestionCtrl = wx.TextCtrl(
            parent, size=(-1, 60), style=wx.TE_MULTILINE,
            value="Where is the drawn pen circle? Answer with pixel coordinates as [x, y].")
        s.Add(self.vlmQuestionCtrl, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM | wx.EXPAND, 5)

        self.vlmAskBtn = wx.Button(parent, label="Ask VLM about current frame")
        self.vlmAskBtn.Bind(wx.EVT_BUTTON, self.onAskVlm)
        s.Add(self.vlmAskBtn, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM | wx.EXPAND, 5)

        s.Add(wx.StaticText(parent, label="Response"), 0, wx.LEFT | wx.RIGHT | wx.TOP, 5)
        self.vlmResponseCtrl = wx.TextCtrl(
            parent, size=(-1, 100), style=wx.TE_MULTILINE | wx.TE_READONLY)
        s.Add(self.vlmResponseCtrl, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM | wx.EXPAND, 5)

        s.Add(wx.StaticText(parent, label="Parsed coordinates"), 0, wx.LEFT | wx.RIGHT | wx.TOP, 5)
        self.vlmPointsList = wx.ListBox(parent, size=(-1, 60), choices=[])
        s.Add(self.vlmPointsList, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM | wx.EXPAND, 5)

        self.vlmAddPointsBtn = wx.Button(parent, label="Add parsed coordinates as points")
        self.vlmAddPointsBtn.Bind(wx.EVT_BUTTON, self.onAddVlmPoints)
        self.vlmAddPointsBtn.Enable(False)
        s.Add(self.vlmAddPointsBtn, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM | wx.EXPAND, 5)

        note = wx.StaticText(parent, label=(
            "Note: this VLM was previously found unreliable for precise defect\n"
            "localisation on this imagery (see knowledge/PLAN.md). SAM3 pen-mark\n"
            "detection (above) remains the primary auto-annotation path — use\n"
            "this panel for ad-hoc questions, not as a trusted detector."))
        note.SetForegroundColour(wx.Colour(150, 100, 0))
        s.Add(note, 0, wx.ALL, 5)

        s.AddStretchSpacer(1)
        parent.SetSizer(s)
        self._vlmParsedPoints = []

    # ---------------------------------------------------------------- shared
    def _currentFrameRaw(self):
        if not self.filepath or not os.path.isfile(self.filepath):
            return None
        return cv2.imread(self.filepath, cv2.IMREAD_UNCHANGED)

    # ---------------------------------------------------------------- SAM3
    def onApplySamSettings(self, _evt):
        if AutoAnnotator is None:
            wx.MessageBox("AutoAnnotator unavailable — is gradio_client installed?",
                          "SAM3", wx.OK | wx.ICON_ERROR)
            return
        ip = self.samIpCtrl.GetValue().strip() or SAM3_IP
        port = self.samPortCtrl.GetValue().strip() or str(SAM3_PORT)
        prompt = self.samPromptCtrl.GetValue().strip() or DEFAULT_PROMPT
        representation = self.samRepresentationCombo.GetValue()
        try:
            self.autoAnnotator = AutoAnnotator(ip=ip, port=port, prompt=prompt,
                                               representation=representation)
        except Exception as e:
            self.samStatusLabel.SetLabel(f"Failed: {e}")
            wx.MessageBox(f"Could not configure AutoAnnotator:\n{e}",
                          "SAM3", wx.OK | wx.ICON_ERROR)
            return
        self.samStatusLabel.SetLabel(
            f"Configured for http://{ip}:{port} (prompt '{prompt}') — "
            f"will connect on next Auto/Track use.")

    def _samDetectKwargs(self):
        """Extra detect()/detect_ex() kwargs sourced from the Automation tab.
        Falls back to detect()'s own defaults (returns {}) on bad input."""
        try:
            return dict(
                min_area=int(float(self.samMinAreaCtrl.GetValue())),
                merge_dist=int(float(self.samMergeDistCtrl.GetValue())),
                max_area_frac=float(self.samMaxAreaFracCtrl.GetValue()),
                max_aspect=float(self.samMaxAspectCtrl.GetValue()),
                use_dolp=self.samUseDolpCheckbox.GetValue(),
                reject_border=self.samRejectBorderCheckbox.GetValue(),
                refine=self.samRefineCheckbox.GetValue(),
            )
        except (ValueError, AttributeError):
            return {}

    # ---------------------------------------------------------------- VLM
    def onAskVlm(self, _evt):
        raw = self._currentFrameRaw()
        if raw is None:
            wx.MessageBox("Could not load the current frame image.",
                          "Ask VLM", wx.OK | wx.ICON_ERROR)
            return
        question = self.vlmQuestionCtrl.GetValue().strip()
        if not question:
            wx.MessageBox("Type a question first.", "Ask VLM", wx.OK | wx.ICON_WARNING)
            return

        ip = self.vlmIpCtrl.GetValue().strip() or VLM_IP
        port = self.vlmPortCtrl.GetValue().strip() or str(VLM_PORT)
        try:
            temperature = float(self.vlmTemperatureCtrl.GetValue())
            top_p = float(self.vlmTopPCtrl.GetValue())
            max_tokens = int(self.vlmMaxTokensCtrl.GetValue())
        except ValueError:
            wx.MessageBox("Temperature/Top-p/Max tokens must be numbers.",
                          "Ask VLM", wx.OK | wx.ICON_WARNING)
            return

        if getattr(self, "_vlmClient", None) is None or \
                (self._vlmClient.ip, self._vlmClient.port) != (ip, port):
            self._vlmClient = VlmClient(ip=ip, port=port)

        bgr = polar_raw_to_bgr(raw, self.samRepresentationCombo.GetValue())

        wx.BeginBusyCursor()
        try:
            response = self._vlmClient.ask(bgr, question, temperature=temperature,
                                           top_p=top_p, max_tokens=max_tokens)
        except Exception as e:
            wx.EndBusyCursor()
            wx.MessageBox(f"VLM query failed:\n{e}", "Ask VLM", wx.OK | wx.ICON_ERROR)
            return
        wx.EndBusyCursor()

        self.vlmResponseCtrl.SetValue(response)
        h, w = bgr.shape[:2]
        pts = parse_grounded_points(response, w, h)
        self._vlmParsedPoints = pts
        self.vlmPointsList.Clear()
        for (x, y) in pts:
            self.vlmPointsList.Append(f"({x:.0f}, {y:.0f}) render px")
        self.vlmAddPointsBtn.Enable(bool(pts))
        if pts:
            self.instructLbl.SetLabel(f"VLM: parsed {len(pts)} coordinate(s) — verify before adding.")
        else:
            self.instructLbl.SetLabel("VLM: no coordinates found in the response — see Response box.")

    def onAddVlmPoints(self, _evt):
        WA = self._WA()
        added = 0
        for (x, y) in self._vlmParsedPoints:
            add_point(self.points_of_interest, self.points_classes,
                      self.points_severities, self.points_sources,
                      x * MOSAIC_SCALE, y * MOSAIC_SCALE,
                      self.defectComboBox.GetValue() or WA.options[0],
                      self.severityComboBox.GetValue() or WA.severities[0],
                      "vlm")
            added += 1
        if added:
            self._stat_points_added += added
            self.updatePointList()
            self.onView()
        self.instructLbl.SetLabel(f"VLM: added {added} point(s) from the grounded answer.")
