#!/usr/bin/python3

"""
Author : "Ammar Qammaz"
Copyright : "2025 Foundation of Research and Technology, Computer Science Department Greece"
License : "FORTH"

Classifier tab for the annotator GUI — extracted from mga/wx_annotator.py
(Stage 2 of its refactor). PhotoCtrl mixes this in: the tab's widgets are
stored on the app instance like every other widget, so the mixin only needs
self plus a lazy handle on the wx_annotator module for the classifier-glue
names that live there (ClassifierPnm, locate_model, GATE_*, the repository
globals…). The lazy import avoids the circular import (wx_annotator imports
this module) and is safe because every name it touches is bound at module
level before PhotoCtrl is ever constructed.

The nested closures of the original _buildClassifierTab are real methods now
(one handler per widget event), and the four label+slider+value rows share
the _slider_row helper.
"""

import os
import json
import wx

from mga.core.model_updater import ModelUpdaterDialog
from mga.core.rl_annotator import RLAnnotatorDialog
from mga.core.read_data_annotator import list_image_files

# Marker appended to names in the Online combo that are also present locally.
LOCAL_MARK = " [have local copy]"


class ClassifierTabMixin:
    """Builds the Classifier tab with model select, threshold, majority voting,
       tile size (4..128), and two-stage classification toggle."""

    def _WA(self):
        """The module that defines PhotoCtrl, cached on self. Resolved through the
        class's __module__, never through `import mga.wx_annotator`: when the app
        is started with `python3 -m mga.wx_annotator`, runpy executes the file
        under __main__ WITHOUT registering it, so a plain import would execute
        the file a second time as a separate module — and writes like
        classifier_model_path would land in the ghost instance's globals, where
        the app never reads them."""
        WA = getattr(self, "_wx_annotator_module", None)
        if WA is None:
            import sys
            WA = sys.modules[self.__class__.__module__]
            self._wx_annotator_module = WA
        return WA

    def _slider_row(self, parent, label, value, min_value, max_value,
                    on_change=None, tooltip=None):
        """Label + slider + live value-label row (the classifier sliders share
        this shape). on_change(evt) fires on every slider move.
        Returns (row_sizer, label_widget, slider, value_ctrl)."""
        row = wx.BoxSizer(wx.HORIZONTAL)
        lbl = wx.StaticText(parent, label=label)
        slider = wx.Slider(parent, value=value, minValue=min_value,
                           maxValue=max_value, style=wx.SL_HORIZONTAL)
        value_ctrl = wx.StaticText(parent, label=str(value))
        if tooltip:
            lbl.SetToolTip(tooltip)
        if on_change is not None:
            slider.Bind(wx.EVT_SLIDER, on_change)
        row.Add(lbl, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 8)
        row.Add(slider, 1, wx.RIGHT, 8)
        row.Add(value_ctrl, 0, wx.ALIGN_CENTER_VERTICAL)
        return row, lbl, slider, value_ctrl

    def _buildClassifierTab(self, parent):
        s = wx.BoxSizer(wx.VERTICAL)
        WA = self._WA()

        # --- 0. Deployment preset (optional) ---
        # If the classifier repo ships recommended_configuration.json, follow its FIRST
        # entry for the startup model and gate, so the annotator and the ROS node agree on
        # what "the recommended setup" is and both pick up changes from a git pull.
        # Everything below stays editable in the GUI -- this only sets the defaults, and
        # falls back to the previous behaviour (alphabetically-first local model, gate
        # defect_mass @ 0.85) when the file or the preset's model is unavailable.
        self._preset = None
        if WA.load_recommended_configuration is not None and WA.recommended_configuration_available():
            try:
                self._preset = WA.load_recommended_configuration()
            except Exception as e:
                print("[preset] recommended_configuration.json unusable:", repr(e))

        # --- 1. Get available models from directory ---
        model_dir = WA.classifier_relative_directory
        self._classifier_model_dir = model_dir
        available_models = WA.ClassifierPnm.model_scan(model_dir)

        # Put the preset's model first so it becomes the startup selection, but only if it
        # is already present locally -- the annotator must not block on a download here.
        preset_model = None
        if self._preset is not None:
            preset_model = self._preset.get("model")
        if preset_model is not None and preset_model in available_models:
            reordered = [preset_model]
            for m in available_models:
                if m != preset_model:
                    reordered.append(m)
            available_models = reordered
            print("[preset] '%s' -> startup model %s" % (self._preset.get("name"), preset_model))
        elif preset_model is not None:
            fallback_name = available_models[0] if available_models else "none"
            print("[preset] '%s' recommends %s, which is not in %s. Using %s instead — fetch "
                  "it from the Online list, or with `python3 -m mvc.inference.model_download %s` "
                  "from the classifier repo root."
                  % (self._preset.get("name"), preset_model, model_dir, fallback_name, preset_model))
            self._preset = None      # its gate belongs to a model we are not loading

        startup = WA.locate_model(model_dir, available_models[0]) if available_models else None
        if startup is not None:
            WA.classifier_model_path, WA.classifier_cfg_path = startup
        else:
            if available_models:
                print("[models] '%s' disappeared between scan and load" % available_models[0])
            available_models = ["(none)"]

        self.initializeModels() #<- initialize models here

        # --- 2. Model selection combo box ---
        modelRow = wx.BoxSizer(wx.HORIZONTAL)
        modelLbl = wx.StaticText(parent, label="Model")
        self.classifierModelCombo = wx.ComboBox(
            parent,
            choices=available_models,
            style=wx.CB_READONLY
        )
        self.classifierModelCombo.SetValue(available_models[0])
        modelRow.Add(modelLbl, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 8)
        modelRow.Add(self.classifierModelCombo, 1, wx.EXPAND)

        # --- 2b. Online repository row: separate list of downloadable models ---
        remoteRow = wx.BoxSizer(wx.HORIZONTAL)
        self._classifier_remote_row = remoteRow  # _refresh_model_lists re-layouts it
        self.remoteModelsLbl = wx.StaticText(parent, label="Online")
        remote_models = self._list_remote(available_models)

        self.remoteModelsLbl.SetLabel(self._remote_summary(remote_models))
        self.remoteModelsLbl.SetToolTip("Models on the online repository; entries marked "
                                        "[have local copy] re-download the newest archive")
        # Fixed min width: long entries otherwise inflate the row's minimum size
        # beyond the panel width and GTK paints overflowing children over neighbors
        self.classifierRemoteCombo = wx.ComboBox(
            parent,
            choices=remote_models if remote_models else ["(repository unreachable)"],
            style=wx.CB_READONLY
        )
        self.classifierRemoteCombo.SetValue((remote_models or ["(repository unreachable)"])[0])
        self.downloadModelBtn = wx.Button(parent, label="Download && Use")
        # label + combo on one row, download button below -> fits a narrow tab pane
        remoteRow.Add(self.remoteModelsLbl, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 8)
        remoteRow.Add(self.classifierRemoteCombo, 1, wx.EXPAND)

        self.downloadModelBtn.Bind(wx.EVT_BUTTON, self.onDownloadModel)

        # --- 3. Callback to reload model when changed ---
        self.classifierModelCombo.Bind(wx.EVT_COMBOBOX, self.onClassifierModelChanged)

        # --- 4. Decision gate: how a tile becomes "clean" vs "some defect" -------
        # The threshold slider below cuts on THIS mode's score, so the same slider
        # value means different things per mode and is not portable between them.
        gateRow = wx.BoxSizer(wx.HORIZONTAL)
        gateLbl = wx.StaticText(parent, label="Gate")
        self.classifierGateMode = wx.ComboBox(
            parent,
            choices=[WA.GATE_DEFECT_MASS, WA.GATE_MAX_PROB, WA.GATE_OFF],
            style=wx.CB_READONLY
        )
        self.classifierGateMode.SetValue(WA.GATE_DEFECT_MASS)
        self.classifierGateMode.SetToolTip(
            "How a tile is judged clean vs defect (mvc.inference.classifier_pnm.gate_tiles):\n"
            "  defect_mass : score = 1 - P(clean), the total probability on ANY defect\n"
            "                class. Flags tiles the model is sure are defective even\n"
            "                when it cannot say which defect. Recommended.\n"
            "  max_prob    : score = max class probability (legacy). Throws away a\n"
            "                0.40 Welding / 0.40 Seal / 0.20 clean tile as clean even\n"
            "                though it is 80% likely a defect.\n"
            "  off         : no gate, plain argmax; the threshold is ignored.\n"
            "On val_altinay/customwide at THIS slider's default 0.85: max_prob gives\n"
            "false-alarm 4.2% / miss 74.7%, defect_mass gives 10.6% / 41.6%."
        )
        self.classifierBestDefectClass = wx.CheckBox(parent, label="Best defect class")
        self.classifierBestDefectClass.SetValue(True)
        self.classifierBestDefectClass.SetToolTip(
            "Above the gate, label the tile with its best DEFECT class instead of the\n"
            "plain argmax, which can still be clean when the probability mass is\n"
            "spread thinly across defect classes (0.8% of gate-passing tiles). Off,\n"
            "those tiles pass the gate and are then labelled clean anyway, so they go\n"
            "undetected. Only applies to the defect_mass gate."
        )
        gateRow.Add(gateLbl, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 8)
        gateRow.Add(self.classifierGateMode, 1, wx.RIGHT, 8)
        gateRow.Add(self.classifierBestDefectClass, 0, wx.ALIGN_CENTER_VERTICAL)

        # --- 4b. Threshold slider (cuts on the gate mode's score) ---
        thrRow, thrLbl, self.classifierThreshold, self.classifierThresholdValue = self._slider_row(
            parent, "Threshold", 85, 0, 100, on_change=self._on_thr)
        self._classifier_threshold_label = thrLbl  # _gate_hint re-labels it per mode

        # Adopt the preset's gate as the starting position of the three gate widgets. The
        # slider is integer percent, so a preset threshold is rounded to the nearest 0.01 --
        # 0.885 becomes 0.89, which is within the sweep's own 0.005 step. The user can still
        # move any of these; this only decides where they start.
        if self._preset is not None:
            preset_gate = self._preset.get("gate") or {}
            preset_mode = preset_gate.get("mode")
            if preset_mode in (WA.GATE_DEFECT_MASS, WA.GATE_MAX_PROB, WA.GATE_OFF):
                self.classifierGateMode.SetValue(preset_mode)
            preset_thr = preset_gate.get("threshold")
            if preset_thr is not None:
                slider_pct = int(round(float(preset_thr) * 100.0))
                if slider_pct < 0:
                    slider_pct = 0
                elif slider_pct > 100:
                    slider_pct = 100
                self.classifierThreshold.SetValue(slider_pct)
                self.classifierThresholdValue.SetLabel("%.2f" % (slider_pct / 100.0))
            if "assign_best_defect_class" in preset_gate:
                self.classifierBestDefectClass.SetValue(bool(preset_gate["assign_best_defect_class"]))
            measured = self._preset.get("measured") or {}
            if "detected" in measured and "false_alarm" in measured:
                print("[preset] gate %s @ %.3f -> detects %.1f%% of defect tiles, "
                      "false-alarms on %.2f%% of clean tiles (%s)"
                      % (preset_gate.get("mode"), float(preset_thr),
                         100.0 * measured["detected"], 100.0 * measured["false_alarm"],
                         measured.get("source", "measured")))

        self.classifierGateMode.Bind(wx.EVT_COMBOBOX, self._on_gate)
        self._gate_hint()

        # --- 5. Majority voting checkbox ---
        self.classifierMajorityVoting = wx.CheckBox(parent, label="Use majority voting")
        self.classifierMajorityVoting.SetValue(True)

        # --- 6. Tile size slider ---
        tileRow, _lbl, self.classifierTileSize, self.classifierTileSizeValue = self._slider_row(
            parent, "Step size", 16, 4, 128, on_change=self._on_tile)

        # --- Erode Kernel Size  ---
        erodeKernelRow, _lbl, self.erodeKernelSize, self.erodeKernelValue = self._slider_row(
            parent, "Vote Neighborhood (kernel)", 1, 0, 8,
            on_change=self._on_erodkrnthr,
            tooltip="Radius k of the tile-voting neighborhood: votes are counted "
                    "over the (2k+1)x(2k+1) tiles around each activation")

        # --- Erode Threshold Value ---
        erodeThresholdRow, _lbl, self.erodeThreshold, self.erodeThresholdValue = self._slider_row(
            parent, "Min Votes to Keep Tile", 2, 0, 8,
            on_change=self._on_erodthr,
            tooltip="Activated tiles (including the tile itself) required inside the "
                    "vote neighborhood for an activation to be accepted; 0/1 = voting off. "
                    "Same setting as the ROS set_min_votes service.")

        # --- 7. Two-stage classification checkbox ---
        self.classifierTwoStage = wx.CheckBox(parent, label="Enable two-stage classification")
        self.parallellTwoStage  = wx.CheckBox(parent, label="Two-stage parallelism (VRAM intensive)")
        self.parallellTwoStage.SetValue(True)
        self.classifierTwoStage.Bind(wx.EVT_CHECKBOX, self._on_two_stage_toggled)

        # --- 7b. Min Hz filter for ensemble (applied at init time) ---
        minHzRow = wx.BoxSizer(wx.HORIZONTAL)
        minHzLbl = wx.StaticText(parent, label="Ensemble min Hz filter:")
        self.ensembleMinHz = wx.TextCtrl(parent, value="10", size=(55, -1), style=wx.TE_PROCESS_ENTER)
        self.ensembleMinHz.SetToolTip(
            "Drop ensemble models slower than this Hz.\n"
            "0 = keep all models.  Press Enter or click away to apply immediately.")
        minHzRow.Add(minHzLbl, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 6)
        minHzRow.Add(self.ensembleMinHz, 0, wx.ALIGN_CENTER_VERTICAL)
        self.ensembleMinHz.Bind(wx.EVT_TEXT_ENTER, self._apply_min_hz)
        self.ensembleMinHz.Bind(wx.EVT_KILL_FOCUS,  self._apply_min_hz)

        # --- 8. "Disabled Model" checkbox (active by default — NN off until user enables it) ---
        self.classifierDisabledCheckbox = wx.CheckBox(parent, label="Disable Neural Network Model (For Speed)")
        self.classifierDisabledCheckbox.SetValue(True)

        # --- 9. Layout ---
        s.Add(modelRow, 0, wx.ALL | wx.EXPAND, 10)
        s.Add(remoteRow, 0, wx.LEFT | wx.RIGHT | wx.TOP | wx.EXPAND, 10)
        s.Add(self.downloadModelBtn, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM | wx.EXPAND, 10)
        s.Add(gateRow, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM | wx.EXPAND, 10)
        s.Add(thrRow, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM | wx.EXPAND, 10)
        s.Add(self.classifierMajorityVoting, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)
        s.Add(tileRow, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM | wx.EXPAND, 10)

        s.Add(erodeKernelRow, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)
        s.Add(erodeThresholdRow, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

        s.Add(self.classifierTwoStage, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)
        s.Add(self.parallellTwoStage, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)
        s.Add(minHzRow, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)
        s.Add(self.classifierDisabledCheckbox, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

        # --- "Check for Model Updates" button ---
        self.checkUpdatesBtn = wx.Button(parent, label="Check for Model Updates…")
        s.Add(self.checkUpdatesBtn, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)
        self.checkUpdatesBtn.Bind(wx.EVT_BUTTON, self._on_check_updates)

        # --- "Check Model Statistics" button ---
        self.checkStatsBtn = wx.Button(parent, label="Check Model Statistics")
        s.Add(self.checkStatsBtn, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)
        self.checkStatsBtn.Bind(wx.EVT_BUTTON, self._on_check_stats)

        # --- "Reinforcement Learning" button + pixel-distance textbox ---
        rl_row = wx.BoxSizer(wx.HORIZONTAL)
        self.rlBtn = wx.Button(parent, label="Reinforcement Learning")
        rl_row.Add(self.rlBtn, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 8)
        rl_row.Add(wx.StaticText(parent, label="Radius (px):"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 4)
        self.rlRadiusCtrl = wx.TextCtrl(parent, value="120", size=(55, -1))
        rl_row.Add(self.rlRadiusCtrl, 0, wx.ALIGN_CENTER_VERTICAL)
        s.Add(rl_row, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)
        self.rlBtn.Bind(wx.EVT_BUTTON, self._on_rl)

        # --- "Purge R/L Labels" button ---
        self.purgeRLBtn = wx.Button(parent, label="Purge R/L Labels")
        s.Add(self.purgeRLBtn, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)
        self.purgeRLBtn.Bind(wx.EVT_BUTTON, self._on_purge_rl)

        self.classifierInfo = wx.StaticText(parent, label="No classifier run yet.")
        s.Add(self.classifierInfo, 0, wx.ALL | wx.EXPAND, 5)

        s.AddStretchSpacer(1)

        parent.SetSizer(s)

    def _list_remote(self, local_models):
        """Models on the online zip repository (CameraV2Models). Already-local names
        are listed too — downloading them fetches the server's newest archive."""
        try:
            # cross-repo: magician_vision_classifier/mvc/inference/model_download.py
            from mvc.inference.model_download import remote_model_names
            local = set(local_models)
            return [n + (LOCAL_MARK if n in local else "")
                    for n in remote_model_names(timeout=5)]
        except Exception as e:
            print(f"[Models] Online repository unavailable: {e}")
            return []

    def _remote_summary(self, remote):
        if not remote:
            return "Online (offline)"
        have = sum(1 for m in remote if m.endswith(LOCAL_MARK))
        return f"Online ({have}/{len(remote)} local)"

    def _refresh_model_lists(self, select=None):
        model_dir = self._classifier_model_dir
        WA = self._WA()
        local = WA.ClassifierPnm.model_scan(model_dir)
        self.classifierModelCombo.Clear()
        for m in local:
            self.classifierModelCombo.Append(m)
        if select and select in local:
            self.classifierModelCombo.SetValue(select)
        elif local:
            self.classifierModelCombo.SetValue(local[0])
        remote = self._list_remote(local)
        self.remoteModelsLbl.SetLabel(self._remote_summary(remote))
        self.classifierRemoteCombo.Clear()
        for m in (remote if remote else ["(repository unreachable)"]):
            self.classifierRemoteCombo.Append(m)
        self.classifierRemoteCombo.SetValue((remote or ["(repository unreachable)"])[0])
        self._classifier_remote_row.Layout()

    def onDownloadModel(self, _evt):
        name = self.classifierRemoteCombo.GetValue()
        if not name or name.startswith("("):
            return
        if name.endswith(LOCAL_MARK):
            name = name[:-len(LOCAL_MARK)]
        model_dir = self._classifier_model_dir
        WA = self._WA()
        self.classifierInfo.SetLabel(f"Downloading '{name}' from the model repository...")
        busy = wx.BusyCursor()
        wx.Yield()
        try:
            # cross-repo: magician_vision_classifier/mvc/inference/model_download.py
            from mvc.inference.model_download import download_model
            # include_plots: the GUI reads confusion/threshold PNGs for the model's
            # report views, so a manual download shouldn't leave it report-less
            # (matches ensure_model's default, used by the web annotator's
            # auto-download-on-select).
            download_model(name, model_dir, include_plots=True)
        except Exception as e:
            wx.MessageBox(f"Failed to download '{name}':\n{e}",
                          "Model Download Error", wx.OK | wx.ICON_ERROR)
            return
        finally:
            del busy
        self._refresh_model_lists(select=name)
        # Load the freshly downloaded model right away
        if WA.useClassifier and self.ClassifierPnm is not None:
            if self.ClassifierPnm.reload_model(model_dir, name):
                self.stats.classifier_name = name.lower()
                self.stats.reset()
                self.classifierInfo.SetLabel(f"Downloaded and switched to '{name}'.")
            else:
                wx.MessageBox(f"Downloaded '{name}' but failed to load it.",
                              "Model Load Error", wx.OK | wx.ICON_ERROR)

    def onClassifierModelChanged(self, evt):
        model_name = self.classifierModelCombo.GetValue()
        model_dir = self._classifier_model_dir
        WA = self._WA()
        print(f"[INFO] Changing classifier model to: {model_name}")
        if WA.useClassifier and self.ClassifierPnm is not None:
            success = self.ClassifierPnm.reload_model(model_dir, model_name)
            if success:
                print(f"Successfully reloaded model: {model_name}")
                self.stats.classifier_name = model_name.lower()
                self.stats.reset()
                self.classifierInfo.SetLabel(f"Model changed to '{model_name}' — statistics reset.")
            else:
                found = WA.locate_model(model_dir, model_name)
                pth = found[0] if found else os.path.join(model_dir, f"{model_name}.pth")
                answer = wx.MessageBox(
                    f"Failed to load '{model_name}'.\n\n"
                    f"The file may be corrupted or incomplete:\n{pth}\n\n"
                    f"Re-download it now?",
                    "Model Load Error", wx.YES_NO | wx.ICON_ERROR
                )
                if answer == wx.YES:
                    dlg = ModelUpdaterDialog(self.frame, WA.classifier_online_repository, model_dir)
                    # Pre-select only the failed model
                    def _preselect(results, err, _dlg=dlg, _name=model_name):
                        if err or not results:
                            return
                        for i, r in enumerate(_dlg._model_data):
                            _dlg.list_ctrl.Check(i, r['name'] == _name)
                    dlg._post_check_hook = _preselect
                    dlg.ShowModal()
                    # Retry loading after download
                    retry = self.ClassifierPnm.reload_model(model_dir, model_name)
                    if retry:
                        print(f"Successfully reloaded model after re-download: {model_name}")
                    else:
                        wx.MessageBox(f"Still failed to load '{model_name}' after re-download.",
                                      "Error", wx.OK | wx.ICON_ERROR)
                    dlg.Destroy()
        else:
            print("[WARN] No classifier_instance found on self.")
        evt.Skip()

    def _gate_hint(self):
        """Spell out what the slider currently thresholds — it changes per mode."""
        WA = self._WA()
        mode = self.classifierGateMode.GetValue()
        thr  = self.classifierThreshold.GetValue() / 100.0
        if mode == WA.GATE_OFF:
            self.classifierThreshold.Enable(False)
            self.classifierBestDefectClass.Enable(False)
            self._classifier_threshold_label.SetLabel("Threshold (unused)")
        else:
            self.classifierThreshold.Enable(True)
            self.classifierBestDefectClass.Enable(mode == WA.GATE_DEFECT_MASS)
            self._classifier_threshold_label.SetLabel("Defect mass >=" if mode == WA.GATE_DEFECT_MASS else "Max prob >=")
        self.classifierThresholdValue.SetLabel(f"{thr:.2f}")

    def _on_thr(self, evt):
        self._gate_hint()
        evt.Skip()

    def _on_gate(self, evt):
        self._gate_hint()
        # Thresholds are not comparable between modes; say so rather than silently
        # reinterpreting the slider the user already set.
        print(f"[Gate] mode={self.classifierGateMode.GetValue()} "
              f"threshold={self.classifierThreshold.GetValue()/100.0:.2f} "
              f"— thresholds are NOT comparable across modes, re-tune the slider.")
        evt.Skip()

    def _on_tile(self, evt):
        self.classifierTileSizeValue.SetLabel(str(self.classifierTileSize.GetValue()))
        evt.Skip()

    def _on_erodkrnthr(self, evt):
        self.erodeKernelValue.SetLabel(f"{self.erodeKernelSize.GetValue()}")
        evt.Skip()

    def _on_erodthr(self, evt):
        self.erodeThresholdValue.SetLabel(f"{self.erodeThreshold.GetValue()}")
        evt.Skip()

    def _on_two_stage_toggled(self, evt):
        if self.classifierTwoStage.GetValue():
            self.stats.classifier_name = "allclass_ensemble"
        else:
            self.stats.classifier_name = self.classifierModelCombo.GetValue().lower()
        self.stats.reset()
        self.classifierInfo.SetLabel(
            f"Switched to {'two-stage ensemble' if self.classifierTwoStage.GetValue() else self.classifierModelCombo.GetValue()} — statistics reset.")
        evt.Skip()

    def _apply_min_hz(self, _evt=None):
        if not self._WA().useClassifier:
            return
        try:
            val = float(self.ensembleMinHz.GetValue())
        except ValueError:
            return
        if getattr(self, 'EnsembleClassifierPnm', None) is None:
            return
        self.EnsembleClassifierPnm.apply_min_hz(val)
        self.classifierInfo.SetLabel(
            f"Ensemble filter: {len(self.EnsembleClassifierPnm.classifiers)}"
            f"/{len(self.EnsembleClassifierPnm._all_classifiers)} models active "
            f"(min {val:.1f} Hz)")

    def _on_check_updates(self, _evt):
        WA = self._WA()
        dlg = ModelUpdaterDialog(self.frame, WA.classifier_online_repository, WA.classifier_relative_directory)
        dlg.ShowModal()
        # Refresh model list after dialog closes
        updated_models = WA.ClassifierPnm.model_scan(WA.classifier_relative_directory)
        if updated_models:
            self.classifierModelCombo.Clear()
            for m in updated_models:
                self.classifierModelCombo.Append(m)
            self.classifierModelCombo.SetValue(updated_models[0])
        dlg.Destroy()

    def _on_check_stats(self, _evt):
        text = self.stats.format_stats()
        if self.classifierTwoStage.GetValue() and hasattr(self.EnsembleClassifierPnm, 'model_perf'):
            perf = self.EnsembleClassifierPnm.model_perf
            if perf:
                lines = ["\n" + "=" * 58,
                         f" Ensemble per-model performance  (ensemble Hz: {self.EnsembleClassifierPnm.hz:.2f})",
                         "=" * 58]
                for name, hz in sorted(perf.items(), key=lambda kv: -kv[1]):
                    bar = "#" * min(38, max(1, int(hz * 2)))
                    lines.append(f"  {name:<43}  {hz:6.2f} Hz  {bar}")
                lines.append("=" * 58)
                text = text + "\n".join(lines)
        dlg  = wx.Dialog(self.frame, title="Classifier Accuracy Statistics", size=(700, 520))
        vsz  = wx.BoxSizer(wx.VERTICAL)
        tc   = wx.TextCtrl(dlg, value=text,
                           style=wx.TE_MULTILINE | wx.TE_READONLY | wx.HSCROLL | wx.TE_RICH2)
        tc.SetFont(wx.Font(9, wx.FONTFAMILY_TELETYPE, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        vsz.Add(tc, 1, wx.ALL | wx.EXPAND, 8)
        btn_row = wx.BoxSizer(wx.HORIZONTAL)
        reset_btn = wx.Button(dlg, label="Reset Statistics")
        close_btn = wx.Button(dlg, wx.ID_CLOSE, label="Close")
        btn_row.Add(reset_btn, 0, wx.RIGHT, 8)
        btn_row.AddStretchSpacer()
        btn_row.Add(close_btn, 0)
        vsz.Add(btn_row, 0, wx.ALL | wx.EXPAND, 8)
        dlg.SetSizer(vsz)

        def _on_reset(_e):
            self.stats.reset()
            tc.SetValue(self.stats.format_stats())

        reset_btn.Bind(wx.EVT_BUTTON, _on_reset)
        close_btn.Bind(wx.EVT_BUTTON, lambda _e: dlg.EndModal(wx.ID_CLOSE))
        dlg.Bind(wx.EVT_CLOSE, lambda _e: dlg.EndModal(wx.ID_CLOSE))
        dlg.ShowModal()
        dlg.Destroy()

    def _on_rl(self, _evt):
        try:
            radius = int(self.rlRadiusCtrl.GetValue())
            if radius <= 0:
                raise ValueError
        except ValueError:
            wx.MessageBox("Please enter a positive integer for the radius.",
                          "Invalid Radius", wx.OK | wx.ICON_WARNING)
            return

        local_dir = getattr(self.folderStreamer, "local_dir", None)
        if not local_dir or not os.path.isdir(local_dir):
            wx.MessageBox("No local dataset directory is open.\n"
                          "Open a dataset folder first.",
                          "No Dataset", wx.OK | wx.ICON_WARNING)
            return

        classifier = (self.EnsembleClassifierPnm
                      if self.classifierTwoStage.GetValue()
                      else self.ClassifierPnm)
        if classifier is None:
            wx.MessageBox("Two-stage mode needs ensemble models (allclass_*.pth/.json),\n"
                          "but none were loaded — falling back to the single classifier.",
                          "Ensemble Not Available", wx.OK | wx.ICON_WARNING)
            classifier = self.ClassifierPnm

        dlg = RLAnnotatorDialog(self.frame, classifier, local_dir, radius)
        dlg.ShowModal()
        dlg.Destroy()

    def _on_purge_rl(self, _evt):
        local_dir = getattr(self.folderStreamer, "local_dir", None)
        if not local_dir or not os.path.isdir(local_dir):
            wx.MessageBox("No local dataset directory is open.\n"
                          "Open a dataset folder first.",
                          "No Dataset", wx.OK | wx.ICON_WARNING)
            return

        answer = wx.MessageBox(
            f"This will permanently remove all RLClean annotations\n"
            f"from the .json files in:\n{local_dir}\n\n"
            f"Continue?",
            "Purge R/L Labels", wx.YES_NO | wx.ICON_WARNING
        )
        if answer != wx.YES:
            return

        from mga.core.rl_annotator import _resolve_json
        images   = list_image_files(local_dir)
        purged   = 0
        modified = 0

        for img_path in images:
            json_path = _resolve_json(img_path)
            if not os.path.isfile(json_path):
                continue
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)
            except Exception:
                continue

            pts = data.get("pointClicks",    [])
            cls = data.get("pointClasses",   [])
            sev = data.get("pointSeverities", [])

            new_pts, new_cls, new_sev = [], [], []
            for p, c, sv in zip(pts, cls, sev):
                if c == "RLClean":
                    purged += 1
                else:
                    new_pts.append(p)
                    new_cls.append(c)
                    new_sev.append(sv)

            if purged > (len(new_pts) + purged - len(pts)):  # something was removed
                pass  # counted above
            if len(new_pts) != len(pts):
                data["pointClicks"]     = new_pts
                data["pointClasses"]    = new_cls
                data["pointSeverities"] = new_sev
                try:
                    with open(json_path, "w") as f:
                        json.dump(data, f, sort_keys=False)
                    modified += 1
                except Exception as e:
                    print(f"[Purge] Failed writing {json_path}: {e}")

        wx.MessageBox(
            f"Purge complete.\n"
            f"Removed {purged} RLClean annotation(s) from {modified} file(s).",
            "Purge R/L Labels", wx.OK | wx.ICON_INFORMATION
        )
        # Refresh current frame view in case it was affected
        self.onProcessNewImageSample(self.filepath)
        self.onView()
