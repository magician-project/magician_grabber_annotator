#!/usr/bin/python3
""" 
Author : "Ammar Qammaz"
Copyright : "2025 Foundation of Research and Technology, Computer Science Department Greece"
License : "FORTH" 
"""


"""

In a machine using :
 Ubuntu 22.04.5 with Python 3.10 
  or
 Ubuntu 24.04.3 with Python 3.12.3


python3 -m venv venv
source venv/bin/activate
python3 -m pip install wxPython opencv-python numpy
Should prepare a venv with the needed dependencies

You can then run:
python3 -m mga.wx_annotator --from /path/to/dataset/here/

"""

import wx
import cv2
import csv
import json
import os
import sys
import numpy as np
import time
import threading
import glob


"""
Configurations in one central place
"""

version         = "0.74"
combineChannels = True
options         = ["Unknown", "Material Defect", "Positive Dent", "Negative Dent", "Deformation", "Seal", "Welding", "Clean", "Suspicious", "Dust", "DontUseThisFrame", "RLClean"]
severities      = ["Class A","Class B","Class C"]
directions      = ["Unknown","No Light","Bottom Left","Top Left","Top","Top Right", "Bottom Right", "Bottom"]
processors      = ["PolarizationRGB1","PolarizationRGB2","PolarizationRGB3", "Polarization_0_degree","Polarization_45_degree","Polarization_90_degree", "Polarization_135_degree", "AoLP", "DoLP", "Normals", "Intensity", "s0", "s1", "s2", "s3", "AoLP (light)", "AoLP (dark)", "DoP", "DoCP", "ToP", "CoP", "RetardationMag", "MaxMinAvgRGB", "Sobel","Visible"]


#classifier_relative_directory = "../classifier" #Old Name
# The entire cross-repo surface into magician_vision_classifier (guarded mvc
# imports, gate names, model helpers) moved to mga/core/classifier_hub.py
# (Stage 3b of this file's refactor) and is re-exported here so
# ClassifierTabMixin (which resolves these names through sys.modules) and
# web_annotator keep working unchanged.
from mga.core.classifier_hub import (useClassifier, benchmark, GATE_DEFECT_MASS,
                                     GATE_MAX_PROB, GATE_OFF,
                                     load_recommended_configuration,
                                     recommended_configuration_available,
                                     ClassifierPnm, EnsembleClassifierPnm,
                                     classifier_online_repository,
                                     classifier_relative_directory,
                                     locate_model, web_model_scan,
                                     ensure_model_downloaded)
# Mutable module state stays here: classifier_tab writes these through WA.
classifier_model_path         = None
classifier_cfg_path           = None


FALLBACK_SCREEN_RES = (1920, 1080)

# Usable desktop for the annotator window, empirically what this setup grants:
# LxQt taskbar eats vertical space and a terminal stays visible on the extended
# desktop (measured optimum: frame 2427x1048). The detected desktop is clamped
# to this so the window/view sizing targets space the WM will actually grant.
USABLE_DESKTOP = (2427, 1048)

def detect_screen_resolution():
    """Return the total desktop resolution (width, height) as a tuple of ints.

    Tries four methods in order, most-reliable first:
      1. xrandr monitor bounding box — spans all monitors in extended desktops
         even when 'Screen 0: current' underreports (Wayland/XWayland/multi-GPU).
      2. xrandr 'Screen 0: current W x H'.
      3. xdpyinfo 'dimensions: WxH pixels'.
      4. tkinter root window geometry.
    Falls back to FALLBACK_SCREEN_RES if every method fails.
    """
    import re as _re
    import subprocess as _sp
    try:
        out = _sp.run(["xrandr"], capture_output=True, text=True, timeout=3)
        if out.returncode == 0:
            monitors = _re.findall(
                r'\bconnected\b(?:\s+primary)?\s+(\d+)x(\d+)\+(\d+)\+(\d+)',
                out.stdout,
            )
            if monitors:
                total_w = max(int(mx) + int(mw) for mw, mh, mx, my in monitors)
                total_h = max(int(my) + int(mh) for mw, mh, mx, my in monitors)
                print(f"Screen resolution computed from {len(monitors)} active "
                      f"xrandr monitor(s): {total_w}x{total_h}")
                return total_w, total_h
            m = _re.search(r'current\s+(\d+)\s*x\s*(\d+)', out.stdout)
            if m:
                w, h = int(m.group(1)), int(m.group(2))
                print(f"Screen resolution detected via xrandr (Screen 0 current): {w}x{h}")
                return w, h
    except Exception:
        pass
    try:
        out = _sp.run(["xdpyinfo"], capture_output=True, text=True, timeout=3)
        if out.returncode == 0:
            m = _re.search(r'dimensions:\s+(\d+)x(\d+)', out.stdout)
            if m:
                w, h = int(m.group(1)), int(m.group(2))
                print(f"Screen resolution detected via xdpyinfo: {w}x{h}")
                return w, h
    except Exception:
        pass
    try:
        import tkinter as tk
        root = tk.Tk(); root.withdraw()
        w, h = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
        print(f"Screen resolution detected via tkinter: {w}x{h}")
        return w, h
    except Exception:
        pass
    print(f"Could not detect screen resolution; using fallback "
          f"{FALLBACK_SCREEN_RES[0]}x{FALLBACK_SCREEN_RES[1]}")
    return FALLBACK_SCREEN_RES

from mga.core.folder_stream import FolderStreamer
from mga.core.classifier_grading import AnnotationCorrelationStats
from mga.core.download_all_frames import BatchProcessDialog
from mga.core.magnifier import MagnifierFrame
from mga.core.classifier_tab import ClassifierTabMixin

# AutoAnnotator needs gradio_client (optional). Import lazily-safe so the GUI still
# launches if the dependency / servers are unavailable; onAuto reports the error.
try:
    from mga.core.auto_annotator import AutoAnnotator, temporal_consensus
except Exception as _autoErr:
    AutoAnnotator = None
    _autoImportError = _autoErr


from mga.core.read_data_annotator import repackPolarToMosaic,readPolarPNMToRGBALive

#-------------------------------------------------------------------------------
# Make Classifier completely seperatable from the rest of the codebase: the
# mvc import gate, the gate-name mirrors and the model helpers (locate_model,
# web_model_scan, ensure_model_downloaded) moved to mga/core/classifier_hub.py
# (Stage 3b of this file's refactor) and are re-exported at the top of this
# file. The cross-repo import checklist lives in the hub's docstring. Two
# further cross-repo ties that are NOT imports: mga/dataset_creator.py writes
# dataset.h5 in the layout mvc.core.dataset_converter.HDF5Dataset reads, and
# mga/stream_dataset.py uses mvc.core.shared_memory.SharedMemoryManager.
# (readData.py was renamed to readDataAnnotator.py on 2026-08-03 because it
# shadowed the classifier's same-named module — it must keep this name.)
#-------------------------------------------------------------------------------

from mga.core.read_data_annotator import resolve_annotation_json_path, checkIfFileExists, checkIfPathIsDirectory
from mga.core.read_data_annotator import annotation_json_path, read_annotation_json, dataset_images
from mga.core.frame_processing import loadFrameMosaic, PROCESSOR_WAYS, pixels_to_mm
from mga.core.tracking import (estimateFrameAffine,
                               SAME_LIGHT_SEARCH_MAX, SAME_LIGHT_MIN_SIMILARITY,
                               lighting_fingerprint_cached, tracking_record,
                               prior_shift_from_record, propagate_points,
                               nudge_auto_points, rotate_auto_points,
                               nudge_tracking_record, rotate_tracking_record)
from mga.core.visualize_data import convertPolarCVMATToRGB, convertRGBCVMATToRGB, tenengrad_focus_measure, determine_intensity_region, detect_sobel_edges
import re
from mga.core.annotation_state import (empty_annotation, align_sources,
                                       normalize_parallel, annotation_to_dict,
                                       annotation_from_dict, normalize_tracking,
                                       write_annotation_json, add_point,
                                       remove_point, nearest_point_sq,
                                       is_near_any)
from mga.core.dataset_finalize import (leading_dark_frames, defect_totals,
                                       fill_tracking, compute_focus_light,
                                       update_info_json)


# Lighting fingerprints, affine estimation, tracking records and the pose-graph
# solve moved to mga/core/tracking.py (Stage 1 of this file's refactor); the
# names are re-imported above so the rest of this file is unchanged. The frame
# annotation schema + parallel-list bookkeeping moved to
# mga/core/annotation_state.py (Stage 3a).






def dataset_defaults(name):
    """(widget attribute, value) pairs to apply, in order, for a dataset name —
    one definition shared by sensibleDefaults() and the --from argument handling
    (the two used to drift apart). Every keyword that matches contributes, so
    "weld_class-b" still ends on Class B. Names the combo attributes directly
    so both call sites stay a one-line loop."""
    n = name.lower()
    out = []
    if ("weld" in n):
        out += [("defectComboBox", "Welding"), ("severityComboBox", "Class A")]
    if ("positive" in n):
        out.append(("defectComboBox", "Positive Dent"))
    if ("negative" in n):
        out.append(("defectComboBox", "Negative Dent"))
    if ("class-a" in n):
        out.append(("severityComboBox", "Class A"))
    if ("class-b" in n):
        out.append(("severityComboBox", "Class B"))
    if ("class-c" in n):
        out.append(("severityComboBox", "Class C"))
    if ("pda" in n) or ("posa" in n):
        out += [("defectComboBox", "Positive Dent"), ("severityComboBox", "Class A")]
    if ("pdb" in n) or ("posb" in n):
        out += [("defectComboBox", "Positive Dent"), ("severityComboBox", "Class B")]
    if ("pdc" in n) or ("posc" in n):
        out += [("defectComboBox", "Positive Dent"), ("severityComboBox", "Class C")]
    if ("nda" in n) or ("nega" in n):
        out += [("defectComboBox", "Negative Dent"), ("severityComboBox", "Class A")]
    if ("ndb" in n) or ("negb" in n):
        out += [("defectComboBox", "Negative Dent"), ("severityComboBox", "Class B")]
    if ("ndc" in n) or ("negc" in n):
        out += [("defectComboBox", "Negative Dent"), ("severityComboBox", "Class C")]
    if ("positive-dent-a" in n):
        out += [("defectComboBox", "Positive Dent"), ("severityComboBox", "Class A")]
    if ("positive-dent-b" in n):
        out += [("defectComboBox", "Positive Dent"), ("severityComboBox", "Class B")]
    if ("positive-dent-c" in n):
        out += [("defectComboBox", "Positive Dent"), ("severityComboBox", "Class C")]
    if ("negative-dent-a" in n):
        out += [("defectComboBox", "Negative Dent"), ("severityComboBox", "Class A")]
    if ("negative-dent-b" in n):
        out += [("defectComboBox", "Negative Dent"), ("severityComboBox", "Class B")]
    if ("negative-dent-c" in n):
        out += [("defectComboBox", "Negative Dent"), ("severityComboBox", "Class C")]
    return out


class PhotoCtrl(wx.App, ClassifierTabMixin):
   def __init__(self, redirect=False, filename=None):
        
        wx.App.__init__(self, redirect, filename)


        screen_width, screen_height = detect_screen_resolution()
        print("Screen Resolution: {}x{}".format(screen_width, screen_height))
        usable_w = min(screen_width,  USABLE_DESKTOP[0])
        usable_h = min(screen_height, USABLE_DESKTOP[1])
        print("Usable desktop for the annotator: {}x{}".format(usable_w, usable_h))

        # Invert the window formula (300 + 2*panelW, panelH + 220) to get the
        # largest panel box that fits the usable desktop; the image (1224x1024,
        # aspect-preserved) is fitted inside that box, 100% native = hard ceiling.
        self._usableDesktop = (usable_w, usable_h)
        # conservative first guess (the real chrome is MEASURED after startup by
        # _calibrateWindowToUsableDesktop and the panels resized to fit exactly)
        avail_w = max(300, (usable_w - 340) // 2)
        avail_h = max(260, usable_h - 310)
        self._viewScaleMax = min(avail_w / 1224.0, avail_h / 1024.0, 1.0)
        self.PhotoMaxSizeWidth   = int(1224 * self._viewScaleMax)
        self.PhotoMaxSizeHeight  = int(1024 * self._viewScaleMax)
        print("View scale ceiling %.2f -> panels %ux%u" %
              (self._viewScaleMax, self.PhotoMaxSizeWidth, self.PhotoMaxSizeHeight))
         
        windowTitle = 'Magician Annotator Tool v%s'%version
        windowPosition = wx.Point(10,10)
        windowSize = wx.Size(300+self.PhotoMaxSizeWidth*2,self.PhotoMaxSizeHeight+220)
        print("Set window frame to ",windowSize)

        self.SetOutputWindowAttributes(title=windowTitle, pos=windowPosition, size=windowSize)
        self.frame = wx.Frame(None, size=windowSize, title=windowTitle, style=wx.DEFAULT_FRAME_STYLE)
        self.panel = wx.Panel(self.frame, size=windowSize)

        self.folderStreamer =  FolderStreamer()

        self.regions_of_interest = []
        self.points_of_interest  = []
        self.points_classes      = []
        self.points_severities   = []
        self.points_sources      = []  # parallel to points_of_interest: "auto" | "manual"
        self.tracking            = None  # list of inter-frame transform records from the Track button (persisted as 'tracking' in the JSON)
        self._light_fp_cache     = {}    # path -> lighting fingerprint vector (see lightingFingerprint)
        self._canonCache         = {}    # dirpath -> canonical-light exemplars (see canonicalize_lighting)
        self.lastFrameFile       = None  # per-dataset last.frame path, set in openDataset
        self._base_cache         = None  # (img, fg, left_bmp, right_bmp, left_ok): annotation-free bases for fast onView redraws
        self.leftViewImage       = None  # processed image shown in the left panel (imageCtrl)
        self.rightViewImage      = None  # foreground/visualization image shown in the right panel (secondaryImageCtrl)
        # --- annotation-effort statistics for the current dataset session (committed to info.json on Finalize) ---
        self._stat_clicks         = 0
        self._stat_keystrokes     = 0
        self._stat_points_added   = 0
        self._stat_points_deleted = 0
        self._stat_active_seconds = 0.0
        self._stat_last_interaction = None   # timestamp of the previous click/keystroke
        self._STAT_IDLE_CAP = 60.0           # gaps longer than this are treated as idle (not annotation time)
        self._prefetch        = None     # (key, data): background-rendered NEXT frame, key=(path,way,brightness,contrast)
        self._prefetch_lock   = threading.Lock()
        self._prefetch_thread = None
        self._pf_next         = None     # params captured for the current frame, used to prefetch N+1 at end of load
        self.AIAnnotations       = None

        self.width  = 0
        self.height = 0

        self.filehash = ""
        self.filepath = ""
        self.filePathIsDirectory = False
        self.metadata = None
        self.tenengrad_focus_measure = 0.0

        self.createWidgets()
        self.frame.Show()
        self.frame.SetSize(windowPosition.x, windowPosition.y, -1, -1)  # Initialize frame position
        self.frame.SetClientSize(windowSize)  # Set the exact client area size
        self.frame.Centre()  # Optional: Center the frame on the screen
        # After startup loading (classifier, online model listing, first frame
        # render) settles, measure the REAL window chrome and fit the panels to
        # the usable desktop exactly — the +220-era estimates undershot in Y.
        wx.CallLater(800, self._calibrateWindowToUsableDesktop)
        print("Final Frame Size: ", self.frame.GetSize())
        print("Final Client Size: ", self.frame.GetClientSize())
        self.x = 0 
        self.y = 0
        self.clickRatioX = 1.0
        self.clickRatioY = 1.0

        self.datasetStartFrame = 0
        self.datasetEndFrame   = -1   # will become max-1

        self.viewedImageFullWidth  = 0
        self.viewedImageFullHeight = 0
        self.viewedImageViewWidth  = 0
        self.viewedImageViewHeight = 0           
        self.processingWay     = 0
        self.brightness_offset = 0
        self.contrast_offset   = 0
        self.scrollStep        = 10

        # --- Where to draw point overlays (static "const") ---
        self.DRAW_TARGET_LEFT  = 1
        self.DRAW_TARGET_RIGHT = 2
        self.DRAW_TARGET_BOTH  = self.DRAW_TARGET_LEFT | self.DRAW_TARGET_RIGHT

        # Change this to control drawing:
        #   DRAW_TARGET_LEFT / DRAW_TARGET_RIGHT / DRAW_TARGET_BOTH
        self.DRAW_TARGET = self.DRAW_TARGET_BOTH

        self.magnifier_source = "left"  # or "right"
        self.magnifier = None
        self.autoAnnotator = None  # created on first use of the Auto button

        self.local_base_path = "./"
        self.controlsData = []

        # --- Measuring tool state ---
        self.measureMode   = False
        self.measurePoints = []   # up to 2 points, stored in full (raw mosaic) coords

        # Create global instance once
        self.stats = AnnotationCorrelationStats(classifier_name=self.classifierModelCombo.GetValue(),hit_radius=60)



   def _clamp_range(self, start, end, total):
       if total <= 0:
           return 0, -1
       start = 0 if start is None else int(start)
       end   = (total - 1) if end is None else int(end)
   
       start = max(0, min(start, total - 1))
       end   = max(0, min(end, total - 1))
       if end < start:
           end = start
       return start, end

   def _ui_max(self):
       return max(0, self.datasetEndFrame - self.datasetStartFrame)

   def _stream_from_ui(self, ui_idx):
       return self.datasetStartFrame + ui_idx

   def _ui_from_stream(self, stream_idx):
       return stream_idx - self.datasetStartFrame

   def _applyDatasetRangeFromMetadata(self):
       total = self.folderStreamer.max()
   
       start = None
       end   = None
       if self.metadata:
           start = self.metadata.get("startFrame", None)
           end   = self.metadata.get("endFrame", None)
   
       start, end = self._clamp_range(start, end, total)

       self.datasetStartFrame = start
       self.datasetEndFrame   = end
   
       print("Dataset range:", self.datasetStartFrame, "..", self.datasetEndFrame, "total:", total)


   def _scan_allclass_models(self, directory):
        """Return a list of (pth, json) pairs for all valid allclass_* models — the flat
           deployment directory first, then runs filed under experiments/<campaign>/<run>/
           (see locate_model)."""
        import zipfile, glob
        result = []
        if not os.path.isdir(directory):
            return result
        candidates = sorted(glob.glob(os.path.join(directory, "allclass_*.pth")))
        candidates += sorted(glob.glob(os.path.join(directory, "experiments", "*", "*",
                                                   "allclass_*.pth")))
        seen = set()
        for pth_path in candidates:
            base = os.path.splitext(os.path.basename(pth_path))[0]
            if base in seen:
                continue
            cfg_path = os.path.join(os.path.dirname(pth_path), f"{base}.json")
            if not os.path.isfile(cfg_path):
                continue
            if not zipfile.is_zipfile(pth_path):
                print(f"[Ensemble] Skipping corrupted/incomplete: {pth_path}")
                continue
            print(f"[Ensemble] Adding model: {base}")
            seen.add(base)
            result.append((pth_path, cfg_path))
        if not result:
            print("[Ensemble] Warning: no valid allclass_* models found in", directory)
        return result

   def initializeModels(self):
        if useClassifier and classifier_model_path is not None:
          _classifier_ok = True
          try:
              self.ClassifierPnm = ClassifierPnm(model_path=classifier_model_path,cfg_path=classifier_cfg_path,precache=benchmark)
          except RuntimeError as e:
              wx.MessageBox(
                  f"Failed to load classifier model:\n{classifier_model_path}\n\nThe file may be corrupted or incomplete.\n\nError: {e}",
                  "Classifier Load Error",
                  wx.OK | wx.ICON_ERROR
              )
              self.ClassifierPnm = None
              _classifier_ok = False
          if _classifier_ok:
           try:
              _min_hz = float(self.ensembleMinHz.GetValue())
           except Exception:
              _min_hz = 0.0
           _ensemble_initial = locate_model(classifier_relative_directory, "allclass_verysmall_cnn")
           _ensemble_models  = self._scan_allclass_models(classifier_relative_directory)
           if (not _ensemble_models) or (_ensemble_initial is None):
              wx.MessageBox(
                  "Ensemble classifier disabled: no usable models found.\n\n"
                  f"The ensemble scans {classifier_relative_directory} (and its\n"
                  "experiments/<campaign>/<run>/ subdirectories) for pairs named\n"
                  "allclass_<name>.pth + allclass_<name>.json, and additionally needs\n"
                  "allclass_verysmall_cnn.pth/.json as the fast pre-filter model.\n\n"
                  "Train models (or symlink existing .pth/.json pairs to allclass_* names)\n"
                  "to enable the ensemble. The single-model classifier still works.",
                  "Ensemble Models Not Found",
                  wx.OK | wx.ICON_WARNING
              )
              self.EnsembleClassifierPnm = None
           else:
              self.EnsembleClassifierPnm = EnsembleClassifierPnm(
                                                            #("../magician_vision_classifier/binary_small_cnn.pth","../magician_vision_classifier/binary_small_cnn.json")
                                                            initial_model_cfg = _ensemble_initial,
                                                            model_cfg_list=_ensemble_models,
                                                            min_hz=_min_hz,
                                                            precache=benchmark)

   def createWidgets(self):
    # ----- Menus (unchanged) -------------------------------------------------
    menuBar = wx.MenuBar()

    fileMenu = wx.Menu()
    itemOpen    = fileMenu.Append(wx.ID_FILE, "&Open Image", "Open an image file")
    itemOpenDir = fileMenu.Append(wx.ID_OPEN, "Open &Directory", "Open a directory")
    itemOpenNet = fileMenu.Append(wx.ID_HOME, "Open &Network", "Open network server")
    itemUpload  = fileMenu.Append(wx.ID_UP,   "Upload &Annotations", "Upload annotations to server")
    self.Bind(wx.EVT_MENU, self.onUploadAnnotations, itemUpload)
    itemBatch   = fileMenu.Append(wx.ID_DOWN, "Download &All Frames", "Process multiple files automatically")
    self.Bind(wx.EVT_MENU, self.onRunBatch, itemBatch)

    itemSave    = fileMenu.Append(wx.ID_SAVE, "&Save", "Save the current file")
    fileMenu.AppendSeparator()
    itemGen     = fileMenu.Append(wx.ID_NEW, "&Generate JSON", "Generate JSON for all files")
    itemDebug   = fileMenu.Append(wx.ID_MORE, "Debug", "Debug GUI")
    fileMenu.AppendSeparator()
    itemExit    = fileMenu.Append(wx.ID_EXIT, "E&xit", "Exit the application")

    self.Bind(wx.EVT_MENU, self.onBrowse, itemOpen)
    self.Bind(wx.EVT_MENU, self.onOpenDirectory, itemOpenDir)
    self.Bind(wx.EVT_MENU, self.onOpenNetwork, itemOpenNet)
    self.Bind(wx.EVT_MENU, self.onGenerateJSON, itemGen)
    self.Bind(wx.EVT_MENU, self.onSave, itemSave)
    self.Bind(wx.EVT_MENU, self.onDebug, itemDebug)
    self.Bind(wx.EVT_MENU, self.onExit, itemExit)

    menuBar.Append(fileMenu, "&File")

    toolsMenu = wx.Menu()
    itemMagnify       = toolsMenu.Append(wx.ID_ZOOM_IN,  "&Magnifier", "Magnifier")
    itemRecordDataset = toolsMenu.Append(wx.ID_STOP,     "&Record Raw Dataset", "Record Raw Dataset")
    itemCreateDataset = toolsMenu.Append(wx.ID_EDIT,     "&Create Training Dataset", "Create Training Dataset")
    itemTileExplorer  = toolsMenu.Append(wx.ID_FIND,     "&Tile Explorer", "Tile Explorer")
    itemStreamer      = toolsMenu.Append(wx.ID_FORWARD,  "&Stream To Shared Memory", "Stream To Shared Memory")
    itemBenchmarkPerf = toolsMenu.Append(wx.ID_INDENT,   "&Benchmark Performance based on loaded NN", "Benchmark Perfomance Classifier")
    itemBenchmarkAcc  = toolsMenu.Append(wx.ID_UNINDENT, "&Benchmark Accuracy based on loaded NN", "Benchmark Accuracy Classifier")
    toolsMenu.AppendSeparator()
    itemMakeVideo     = toolsMenu.Append(wx.ID_ANY, "&Make Video", "Render all frames to a video file")
    itemImageSnapshot = toolsMenu.Append(wx.ID_ANY, "&Image Snapshot", "Save 4 PNGs of the current frame (left/right, with and without overlays)")
    self.Bind(wx.EVT_MENU, self.onOpenMagnifier,itemMagnify)
    self.Bind(wx.EVT_MENU, self.onRecordDataset,itemRecordDataset)
    self.Bind(wx.EVT_MENU, self.onCreateDataset,itemCreateDataset)
    self.Bind(wx.EVT_MENU, self.onTileExplorer,itemTileExplorer)
    self.Bind(wx.EVT_MENU, self.onStreamer,itemStreamer)
    self.Bind(wx.EVT_MENU, self.onBenchmarkPerf,itemBenchmarkPerf)
    self.Bind(wx.EVT_MENU, self.onBenchmarkAcc,itemBenchmarkAcc)
    self.Bind(wx.EVT_MENU, self.onMakeVideo, itemMakeVideo)
    self.Bind(wx.EVT_MENU, self.onImageSnapshot, itemImageSnapshot)
    menuBar.Append(toolsMenu, "&Tools")

    helpMenu = wx.Menu()
    itemAbout = helpMenu.Append(wx.ID_ABOUT, "&About", "Information about this application")
    self.Bind(wx.EVT_MENU, self.onAbout, itemAbout)
    menuBar.Append(helpMenu, "&Help")

    self.frame.SetMenuBar(menuBar)

    # ----- Main image views ---------------------------------------------------
    img = wx.Image(self.PhotoMaxSizeWidth,self.PhotoMaxSizeHeight)
    self.imageCtrl = wx.StaticBitmap(self.panel, wx.ID_ANY, wx.Bitmap(img))
    self.secondaryImageCtrl = wx.StaticBitmap(self.panel, wx.ID_ANY, wx.Bitmap(img))

    self.instructLbl = wx.StaticText(self.panel, label='Magician Annotator')
    self.photoTxt = wx.TextCtrl(self.panel, size=(200, -1),style=wx.TE_PROCESS_ENTER)
    self.photoTxt.Bind(wx.EVT_TEXT_ENTER, self.onPhotoTxtEnter)

    browseBtn = wx.Button(self.panel, label='Browse')
    browseBtn.Bind(wx.EVT_BUTTON, self.onBrowse)
    self.rescanBtn = wx.Button(self.panel, label='Finalize')
    self.rescanBtn.Bind(wx.EVT_BUTTON, self.onFinalize)

    # Horizontal “timeline” slider
    self.scrollBar = wx.Slider(self.panel, value=0, minValue=0, maxValue=1000, size=(400, -1), style=wx.SL_HORIZONTAL)
    self.scrollBar.SetTickFreq(50)
    self.scrollBar.Bind(wx.EVT_SLIDER, self.onScroll)

    # Brightness / contrast: one compact slider each (integer range 0..5)
    self.brightnessLabel  = wx.StaticText(self.panel, label="Br")
    self.brightnessSlider = wx.Slider(self.panel, value=0, minValue=0, maxValue=60,
                                      size=(90, -1), style=wx.SL_HORIZONTAL)
    self.brightnessSlider.SetToolTip("Brightness offset (0-60; adds 1x per step to pixel values)")
    self.brightnessSlider.Bind(wx.EVT_SCROLL_CHANGED, self.on_brightness_slider)
    self.contrastLabel  = wx.StaticText(self.panel, label="Co")
    self.contrastSlider = wx.Slider(self.panel, value=0, minValue=0, maxValue=5,
                                    size=(90, -1), style=wx.SL_HORIZONTAL)
    self.contrastSlider.SetToolTip("Contrast offset (0-5)")
    self.contrastSlider.Bind(wx.EVT_SCROLL_CHANGED, self.on_contrast_slider)

    # View size: scales BOTH image panels, up to the native 1224x1024 resolution
    self.viewSizeLabel  = wx.StaticText(self.panel, label="View")
    _vmax = max(41, int(100 * getattr(self, '_viewScaleMax', 1.0)))
    self.viewSizeSlider = wx.Slider(self.panel,
                                    value=max(40, min(_vmax, int(100 * self.PhotoMaxSizeWidth / 1224))),
                                    minValue=40, maxValue=_vmax,
                                    size=(110, -1), style=wx.SL_HORIZONTAL)
    self.viewSizeSlider.SetToolTip(f"Size of the two image panels as % of the native "
                                   f"1224x1024 image resolution (max {_vmax}% fits this desktop)")
    self.viewSizeSlider.Bind(wx.EVT_SCROLL_CHANGED, self.on_view_size_slider)

    # Under-image navigation
    self.prevBtn = wx.Button(self.panel, label='<')
    self.prevBtn.Bind(wx.EVT_BUTTON, self.onPrevious)
    self.playBtn = wx.Button(self.panel, label='Play')
    self.playBtn.Bind(wx.EVT_BUTTON, self.onTogglePlay)
    self.nextBtn = wx.Button(self.panel, label='>')
    self.nextBtn.Bind(wx.EVT_BUTTON, self.onNext)


    self.isPlaying = False
    self.playIntervalMs = 100  # adjust speed here
    self.playTimer = wx.Timer(self)
    self.Bind(wx.EVT_TIMER, self.onPlayTimer, self.playTimer)

    global processors
    self.ProcessorComboBox = wx.ComboBox(self.panel, choices=processors, style=wx.CB_DROPDOWN)
    self.ProcessorComboBox.Bind(wx.EVT_COMBOBOX, self.onProcessorComboBoxSelect)
    self.ProcessorComboBox.SetValue("DoLP")

    # Canonical light view: photometrically remap whichever of the 6 strobed
    # lights lit this frame so every frame "shows as" light #0 (see
    # _canonicalizeLighting). Experimental visualization aid.
    self.canonicalLightCheckbox = wx.CheckBox(self.panel, label="Canonical light")
    self.canonicalLightCheckbox.SetToolTip(
        "Resolve which of the strobed lights illuminates this frame (ActiveLighting "
        "signatures) and rescale the 4 polarization channels so it renders as light #0")
    self.canonicalLightCheckbox.Bind(wx.EVT_CHECKBOX,
                                     lambda e: self.onProcessNewImageSample(self.filepath))

    # ----- Layout roots -------------------------------------------------------
    self.mainSizer  = wx.BoxSizer(wx.VERTICAL)
    self.sizer      = wx.BoxSizer(wx.HORIZONTAL)  # holds (left images) + (right tabs)
    self.underImage = wx.BoxSizer(wx.HORIZONTAL)

    self.mainSizer.Add(wx.StaticLine(self.panel, wx.ID_ANY), 0, wx.ALL | wx.EXPAND, 5)
    self.mainSizer.Add(self.instructLbl, 0, wx.ALL, 5)

    # Left: two image panes
    imagesSizer = wx.BoxSizer(wx.HORIZONTAL)
    imagesSizer.Add(self.imageCtrl, 0, wx.ALL, 5)
    imagesSizer.Add(self.secondaryImageCtrl, 0, wx.ALL, 5)
    self.sizer.Add(imagesSizer, 0, wx.ALL, 5)

    # Right: Notebook with two tabs
    self.rightBook = wx.Notebook(self.panel, style=wx.NB_TOP)

    # --- Annotator tab (contains everything up to "Guess lighting direction") ---
    annotatorPanel = wx.Panel(self.rightBook)
    self._buildAnnotatorTab(annotatorPanel)
    self.rightBook.AddPage(annotatorPanel, "Annotator")

    # --- Classifier tab (model, threshold, majority voting, tile size, two-stage) ---
    classifierPanel = wx.Panel(self.rightBook)
    self._buildClassifierTab(classifierPanel)
    self.rightBook.AddPage(classifierPanel, "Classifier")

    # --- Sensor / Controls tab (model, threshold, majority voting, tile size, two-stage) ---
    self.controlsPanel = wx.Panel(self.rightBook)
    self._buildControlsTab(self.controlsPanel)
    self.rightBook.AddPage(self.controlsPanel, "Sensors")



    # Add notebook to the right side
    self.sizer.Add(self.rightBook, 1, wx.ALL | wx.EXPAND, 5)

    # Add top row to main
    self.mainSizer.Add(self.sizer, 1, wx.ALL | wx.EXPAND, 5)

    # Under-image controls row
    self.underImage.Add(self.prevBtn, 0, wx.ALL, 5)
    self.underImage.Add(self.playBtn, 0, wx.ALL, 5) 
    self.underImage.Add(self.nextBtn, 0, wx.ALL, 5)
    self.underImage.Add(self.photoTxt, 0, wx.ALL, 5)
    self.underImage.Add(browseBtn, 0, wx.ALL, 5)
    self.underImage.Add(self.rescanBtn, 0, wx.ALL, 5)
    self.underImage.Add(self.scrollBar, 1, wx.ALL | wx.EXPAND, 5)
    self.underImage.Add(self.ProcessorComboBox, 0, wx.ALL, 5)
    self.underImage.Add(self.canonicalLightCheckbox, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)

    self.underImage.Add(self.brightnessLabel, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
    self.underImage.Add(self.brightnessSlider, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
    self.underImage.Add(self.contrastLabel, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
    self.underImage.Add(self.contrastSlider, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
    self.underImage.Add(self.viewSizeLabel, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
    self.underImage.Add(self.viewSizeSlider, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)

    self.mainSizer.Add(self.underImage, 0, wx.ALL | wx.EXPAND, 5)

    # Finalize
    self.panel.SetSizer(self.mainSizer)
    self.mainSizer.Fit(self.frame)

    def _onFrameSize(evt):
        fw, fh = evt.GetSize()
        dw, dh = wx.DisplaySize()
        clamped = " (CLAMPED to display!)" if fw >= dw or fh >= dh else ""
        print(f"[Resize] frame -> {fw}x{fh}{clamped}")
        # Keep the View slider ceiling in sync with the frame the WM actually
        # grants (manual drags can exceed what programmatic resizes get) —
        # cheap: only the slider max moves; the user pulls it up to enlarge.
        try:
            wx.CallAfter(self._applyViewCeiling)  # client size settles after the event
        except Exception:
            pass
        evt.Skip()
    self.frame.Bind(wx.EVT_SIZE, _onFrameSize)

    # Mouse + keys bindings on the images/panel
    self.imageCtrl.Bind(wx.EVT_LEFT_DOWN, self.onLeftDown)
    self.secondaryImageCtrl.Bind(wx.EVT_LEFT_DOWN, self.onLeftDown)
    self.imageCtrl.Bind(wx.EVT_MIDDLE_DOWN, self.onMiddleDown)
    self.secondaryImageCtrl.Bind(wx.EVT_MIDDLE_DOWN, self.onMiddleDown)
    self.imageCtrl.Bind(wx.EVT_RIGHT_DOWN, self.onRightDown)
    self.secondaryImageCtrl.Bind(wx.EVT_RIGHT_DOWN, self.onRightDown)
    self.panel.Bind(wx.EVT_MOUSEWHEEL, self.onMouseWheel)
    self.frame.Bind(wx.EVT_CHAR_HOOK, self.onKeyPress)
    self.panel.Layout()
#===============================================================================
#===============================================================================
#===============================================================================
   def _buildAnnotatorTab(self, parent):
    """Builds the right-side Annotator tab with the original right panel controls
       up to and including 'Guess lighting direction'."""
    s = wx.BoxSizer(wx.VERTICAL)

    # Processor (kept in top bar previously, but we leave it there; not duplicated here)

    # Dataset Information
    self.datasetLabel = wx.StaticText(parent, label="Dataset Information")
    datasetListSize = wx.Size(-1, 80)
    self.datasetList  = wx.ListBox(parent, size=datasetListSize, choices=[], style=wx.LB_SINGLE)

    # Image Regions: widgets kept alive (regionList is used by restore/update
    # code paths) but hidden and not added to the sizer — the tab was too crammed
    self.regionLabel = wx.StaticText(parent, label="Image Regions")
    regionListSize = wx.Size(-1, 24)
    self.regionList = wx.ListBox(parent, size=regionListSize, choices=[], style=wx.LB_SINGLE)
    self.regionList.Bind(wx.EVT_LISTBOX, self.onSelectPoint)
    self.removeRegionBtn = wx.Button(parent, label='Remove Selected Point')
    self.removeRegionBtn.Bind(wx.EVT_BUTTON, self.onRemovePoint)
    self.regionLabel.Hide()
    self.regionList.Hide()
    self.removeRegionBtn.Hide()

    # Classification + Severity (combo row)
    self.defectLabel = wx.StaticText(parent, label="Defect Classification")
    global options, severities, directions
    self.defectComboBox = wx.ComboBox(parent, choices=options, style=wx.CB_DROPDOWN)
    self.defectComboBox.Append("Add Custom Option")
    self.defectComboBox.Bind(wx.EVT_COMBOBOX, self.onDefectComboBoxSelect)
    self.defectComboBox.SetValue(options[0])

    self.severityComboBox = wx.ComboBox(parent, choices=severities, style=wx.CB_DROPDOWN)

    comboClass = wx.BoxSizer(wx.HORIZONTAL)
    comboClass.Add(self.defectComboBox, 1, wx.ALL | wx.EXPAND, 5)
    comboClass.Add(self.severityComboBox, 1, wx.ALL | wx.EXPAND, 5)

    # Light Direction
    self.lightLabel = wx.StaticText(parent, label="Light Direction")
    self.lightComboBox = wx.ComboBox(parent, choices=directions, style=wx.CB_DROPDOWN)

    # Points
    self.pointLabel = wx.StaticText(parent, label="Image Points")
    # Real minimum height: without it the surrounding fixed-size widgets squeeze
    # this (the only stretchable item) to almost nothing; proportion 1 in the
    # sizer still lets it absorb any extra space.
    self.pointList = wx.ListBox(parent, size=wx.Size(-1, 200), choices=[], style=wx.LB_SINGLE)
    self.pointList.Bind(wx.EVT_LISTBOX, self.onSelectPoint)
    self.removePointBtn = wx.Button(parent, label='Remove Selected Point')
    self.removePointBtn.Bind(wx.EVT_BUTTON, self.onRemovePoint)

    # Copy points from previous frame
    self.copyPrevPointsBtn = wx.Button(parent, label='Copy Previous Points')
    self.copyPrevPointsBtn.Bind(wx.EVT_BUTTON, self.onCopyPreviousPoints)

    # Action buttons
    self.autoBtn = wx.Button(parent, label='Auto')
    self.autoBtn.Bind(wx.EVT_BUTTON, self.onAuto)
    self.fullAutoBtn = wx.Button(parent, label='Full Auto')
    self.fullAutoBtn.Bind(wx.EVT_BUTTON, self.onFullAuto)
    self.trackBtn = wx.Button(parent, label='Track >')
    self.trackBtn.Bind(wx.EVT_BUTTON, self.onTrack)
    self.trackBackBtn = wx.Button(parent, label='< Track')
    self.trackBackBtn.Bind(wx.EVT_BUTTON, self.onTrackBack)
    self.saveBtn = wx.Button(parent, label='Save')
    self.saveBtn.Bind(wx.EVT_BUTTON, self.onSave)
    self.deleteMetadataBtn = wx.Button(parent, label='Delete')
    self.deleteMetadataBtn.Bind(wx.EVT_BUTTON, self.ondeleteMetadata)

    self.fillTrackingBtn = wx.Button(parent, label='Fill Tracking')
    self.fillTrackingBtn.Bind(wx.EVT_BUTTON, self.onFillTracking)
    self.nudgeBtn = wx.Button(parent, label='Nudge')
    self.nudgeBtn.Bind(wx.EVT_BUTTON, self.onNudgeTracking)

    # WrapSizer: the row folds to extra lines instead of clipping Save/Delete
    # when the pane is narrower than the buttons.
    comboButtons = wx.WrapSizer(wx.HORIZONTAL)
    comboButtons.Add(self.autoBtn, 0, wx.ALL, 5)
    comboButtons.Add(self.fullAutoBtn, 0, wx.ALL, 5)
    comboButtons.Add(self.trackBtn, 0, wx.ALL, 5)
    comboButtons.Add(self.trackBackBtn, 0, wx.ALL, 5)
    comboButtons.Add(self.saveBtn, 0, wx.ALL, 5)
    comboButtons.Add(self.deleteMetadataBtn, 0, wx.ALL, 5)

    trackButtons = wx.WrapSizer(wx.HORIZONTAL)
    trackButtons.Add(self.fillTrackingBtn, 0, wx.ALL, 5)
    trackButtons.Add(self.nudgeBtn, 0, wx.ALL, 5)

    # Checkboxes (up to Guess lighting direction)
    self.incrementFrameAfterAnAdditionCheckbox = wx.CheckBox(parent, label="Increment frame after defect annotation")        
    self.incrementFrameAfterAnAddition=True
    self.incrementFrameAfterAnAdditionCheckbox.SetValue(self.incrementFrameAfterAnAddition)
    self.calcFocusLightCheckbox = wx.CheckBox(parent, label="Calculate Focus & Light Direction")
    self.calcFocusLightCheckbox.SetValue(False)

    self.autoUseClassifierCheckbox = wx.CheckBox(parent, label="Use classifier in Auto annotation")
    self.autoUseClassifierCheckbox.SetToolTip(
        "When checked, Auto and Full Auto also run the ACTIVE classifier (current "
        "threshold/step/vote settings) and merge its detections into the annotations "
        "(source 'classifier', severity 'AI')")
    self.autoUseClassifierCheckbox.SetValue(False)

    # Layout stack for Annotator tab
    s.Add(self.datasetLabel, 0, wx.ALL | wx.EXPAND, 5)
    s.Add(self.datasetList, 0, wx.ALL | wx.EXPAND, 5)

    s.Add(wx.StaticLine(parent), 0, wx.ALL | wx.EXPAND, 5)

    s.Add(self.defectLabel, 0, wx.ALL, 5)
    s.Add(comboClass, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM | wx.EXPAND, 5)

    s.Add(self.lightLabel, 0, wx.ALL, 5)
    s.Add(self.lightComboBox, 0, wx.ALL | wx.EXPAND, 5)

    s.Add(self.pointLabel, 0, wx.ALL | wx.EXPAND, 5)
    s.Add(self.pointList, 1, wx.ALL | wx.EXPAND, 5)

    pointButtons = wx.BoxSizer(wx.HORIZONTAL)
    pointButtons.Add(self.removePointBtn, 1, wx.ALL | wx.EXPAND, 5)
    pointButtons.Add(self.copyPrevPointsBtn, 1, wx.ALL | wx.EXPAND, 5)
    s.Add(pointButtons, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM | wx.EXPAND, 0)

    s.Add(comboButtons, 0, wx.ALL | wx.EXPAND, 5)
    s.Add(trackButtons, 0, wx.ALL | wx.EXPAND, 5)
    s.Add(self.autoUseClassifierCheckbox, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 5)
    s.Add(self.incrementFrameAfterAnAdditionCheckbox, 0, wx.ALL, 5)
    s.Add(self.calcFocusLightCheckbox, 0, wx.ALL, 5)

    parent.SetSizer(s)
#===============================================================================
#===============================================================================
#===============================================================================
   def _buildClassifierTab(self, parent):
    # Moved to mga/core/classifier_tab.py as ClassifierTabMixin (Stage 2 of this
    # file's refactor); PhotoCtrl mixes it in and the widgets live on self as before.
    ClassifierTabMixin._buildClassifierTab(self, parent)
   def _buildControlsTab(self, parent):
    """Builds the Controls tab showing real-time sensor data from CSV."""
    s = wx.BoxSizer(wx.VERTICAL)

    # --- Titles ---
    self.controlsLabel = wx.StaticText(parent, label="Sensor & Control Status")
    s.Add(self.controlsLabel, 0, wx.ALL | wx.EXPAND, 5)

    grid = wx.FlexGridSizer(rows=0, cols=4, vgap=5, hgap=10)
    grid.AddGrowableCol(1, 1)
    grid.AddGrowableCol(3, 1)

    # Helper to make static text pairs
    def label_pair(label_text):
        label = wx.StaticText(parent, label=label_text)
        value = wx.TextCtrl(parent, value="", style=wx.TE_READONLY, size=(30,-1))
        return label, value

    # --- Create all fields ---
    labels = [
        "timestamp", "dev_timestamp", "Button1",
        "Distance1", "Distance2", "Distance3",
        "Light1", "Light2", "Light3", "Light4", "Light5", "Light6"
    ]

    self.controlsFields = {}

    for lbl in labels:
        l, v = label_pair(lbl)
        self.controlsFields[lbl] = v
        grid.Add(l, 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 5)
        grid.Add(v, 1, wx.EXPAND | wx.RIGHT, 5)

    s.Add(grid, 0, wx.ALL | wx.EXPAND, 10)

    # --- Measuring Tool (calibration + 2-click distance) ---
    measure_box = wx.StaticBoxSizer(wx.StaticBox(parent, label="Measuring Tool"), wx.VERTICAL)

    calibGrid = wx.FlexGridSizer(rows=2, cols=2, vgap=5, hgap=10)
    calibGrid.AddGrowableCol(1, 1)

    calibGrid.Add(wx.StaticText(parent, label="Pixels per mm"), 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 5)
    self.calibPxPerMm = wx.TextCtrl(parent, value="7.95", size=(70, -1))
    calibGrid.Add(self.calibPxPerMm, 1, wx.EXPAND | wx.RIGHT, 5)

    calibGrid.Add(wx.StaticText(parent, label="at height (mm)"), 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 5)
    self.calibHeightMm = wx.TextCtrl(parent, value="107.66", size=(70, -1))
    calibGrid.Add(self.calibHeightMm, 1, wx.EXPAND | wx.RIGHT, 5)

    measure_box.Add(calibGrid, 0, wx.ALL | wx.EXPAND, 5)

    self.measureBtn = wx.Button(parent, label="Measure (2 clicks)")
    self.measureBtn.Bind(wx.EVT_BUTTON, self.onToggleMeasure)
    measure_box.Add(self.measureBtn, 0, wx.ALL | wx.EXPAND, 5)

    self.measureResult = wx.StaticText(parent, label="Idle.")
    measure_box.Add(self.measureResult, 0, wx.ALL | wx.EXPAND, 5)

    s.Add(measure_box, 0, wx.ALL | wx.EXPAND, 5)

    # --- CSV Info ---
    self.csvInfo = wx.StaticText(parent, label="No CSV loaded.")
    s.Add(self.csvInfo, 0, wx.ALL | wx.EXPAND, 5)

    # --- Small Sensor Plots Section ---
    plot_box   = wx.StaticBoxSizer(wx.StaticBox(parent, label="Tactile Sensor Plots"), wx.HORIZONTAL)
    grid_plots = wx.GridSizer(rows=2, cols=3, vgap=5, hgap=5)

    # Create placeholders for the 6 CSV plot images
    self.sensorPlotImages = {}
    plot_names = [
        "acceleration_psd", "acceleration_spikeness", "accelerometer",
        "force", "force_psd", "friction"
    ]

    for name in plot_names:
        bmp = wx.StaticBitmap(parent, bitmap=wx.Bitmap(100, 100))
        self.sensorPlotImages[name] = bmp
        grid_plots.Add(bmp, 0, wx.ALL | wx.ALIGN_CENTER, 5)

    plot_box.Add(grid_plots, 1, wx.ALL | wx.EXPAND, 5)
    s.Add(plot_box, 0, wx.ALL | wx.EXPAND, 5)

    parent.SetSizer(s)

    # --- Load the CSV plots initially ---
    self._initializeSensorPlotPlaceholders(parent=parent)
#===============================================================================
#===============================================================================
#===============================================================================
   def _initializeSensorPlotPlaceholders(self, parent, width=100, height=100):
    """Fills existing wx.StaticBitmap controls with 'Not Loaded' placeholder images."""
    plot_names = [
        "acceleration_psd",
        "acceleration_spikeness",
        "accelerometer",
        "force",
        "force_psd",
        "friction",
    ]

    for name in plot_names:
        img = np.zeros((height, width, 3), dtype=np.uint8)
        text = "No Data"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.2
        color = (180, 180, 180)
        thickness = 1
        tsize = cv2.getTextSize(text, font, font_scale, thickness)[0]
        tx, ty = (width - tsize[0]) // 2, (height + tsize[1]) // 2
        cv2.putText(img, text, (tx, ty), font, font_scale, color, thickness, cv2.LINE_AA)

        bmp = wx.Bitmap.FromBuffer(width, height, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # ✅ Only update bitmap of already existing control
        if name in self.sensorPlotImages:
            self.sensorPlotImages[name].SetBitmap(bmp)

    parent.Layout()

#===============================================================================
#===============================================================================
#===============================================================================

   def _loadSensorPlotsNewDataset(self, directory = "./"):
    """Load CSVs, render small plots, and update existing wx.StaticBitmap controls."""
    from mga.core.tactile_plotter import SensorVisualizer, load_csv_with_headers, load_csv_without_headers

    
    self.vis = SensorVisualizer()
    
    self.vis.add_dataset("acceleration_psd",       load_csv_without_headers(os.path.join(directory, "acceleration_psd.csv"), "freq", "power"))
    self.vis.add_dataset("acceleration_spikeness", load_csv_without_headers(os.path.join(directory, "acceleration_spikeness.csv"), "time", "spike"))
    self.vis.add_dataset("force_psd",              load_csv_without_headers(os.path.join(directory, "force_psd.csv"), "freq", "power"))
    self.vis.add_dataset("friction",               load_csv_without_headers(os.path.join(directory, "friction.csv"), "time", "value"))
    self.vis.add_dataset("accelerometer",          load_csv_with_headers(os.path.join(directory, "accelerometer.csv")))
    self.vis.add_dataset("force",                  load_csv_with_headers(os.path.join(directory, "force.csv")))

    #Make plots less spam
    self.vis.drop_column("acceleration_psd","freq")
    self.vis.drop_column("acceleration_spikeness","time")
    self.vis.drop_column("force_psd","freq")
    self.vis.drop_column("friction","time")
    self.vis.drop_column("accelerometer","timestamp")
    self.vis.drop_column("accelerometer","dev_timestamp")
    self.vis.drop_column("force","timestamp")
    self.vis.drop_column("force","tX")
    self.vis.drop_column("force","tY")
    self.vis.drop_column("force","tZ")



   def _loadSensorPlotsNewSample(self, sample_number=100):
    """Render small plots, and update existing wx.StaticBitmap controls."""
    try:
      # Small plots for UI
      images = self.vis.plot_window(sample_number=sample_number, window_size=100, width=100, height=100)

      for name, img in images.items():
        if name in self.sensorPlotImages:
            h, w = img.shape[:2]
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            bmp = wx.Bitmap.FromBuffer(w, h, img_rgb)
            # 🔧 Update existing control instead of creating new one
            self.sensorPlotImages[name].SetBitmap(bmp)

      self.csvInfo.SetLabel(f"CSV plots loaded for sample {sample_number}.")
      self.controlsLabel.GetParent().Layout()  # ensure refresh in grid

    except Exception as e:
      print("_loadSensorPlotsNewSample failed")

#===============================================================================
#===============================================================================
#===============================================================================

    # Add this method to your PhotoCtrl class
   def restoreFromJSON(self, filepath):
      if checkIfFileExists(filepath):
        with open(filepath, 'r') as json_file:
            data = json.load(json_file)

            if 'md5hash' in data:
                self.filehash = data['md5hash']

            if "tenengradFocusMeasure" in data:
                self.tenengrad_focus_measure = data['tenengradFocusMeasure']


            # Restore self.pointList
            if 'pointClicks' in data:
                self.points_of_interest = data['pointClicks']
            if 'pointClasses' in data:
                self.points_classes = data['pointClasses']
            if 'pointSeverities' in data:
                self.points_severities = data['pointSeverities']
            # Origin of each point. Legacy JSONs have no 'pointSources' field -> assume manual.
            if 'pointSources' in data:
                self.points_sources = align_sources(self.points_of_interest,
                                                    data['pointSources'])
            else:
                self.points_sources = align_sources(self.points_of_interest, [])
            if 'regionClicks' in data:
                self.regions_of_interest = data['regionClicks']

            if 'lightDirection' in data:
                   self.lightComboBox.SetValue(data['lightDirection'])
            else:
                   self.lightComboBox.SetValue("Unknown")

            # Inter-frame transform records from the Track button (see onTrack).
            # A list of records; a bare dict (early format) is wrapped for compatibility.
            self.tracking = normalize_tracking(data)

            self.updatePointList()
            self.updateRegionList()

   def onGenerateJSON(self,event):
        print("on generate called!")
        if (self.filePathIsDirectory):
               #self.onSave(None)
               self.folderStreamer.select(0)
               #self.directoryListIndex = 0
               #for i in range(len(self.directoryListIndex)):
               for i in range(self.folderStreamer.max()):
                 print("NEXT")   
                 self.onNext(event) 

   def onDebug(self, event):
        print("Debug")
        import wx.lib.inspection
        wx.lib.inspection.InspectionTool().Show()

   def ondeleteMetadata(self, event):
        print("Deleting metadata for active image")
        self.cleanThisFrameMetaData()
        jsonFile = "%s.json" % self.filepath
        print("Will now delete ",jsonFile)
        os.system("rm %s"%jsonFile)
        self.onRedrawData(event)

   def _ensureAutoAnnotator(self):
      """Lazily create the AutoAnnotator (connects to the SAM3 server on first use)."""
      if getattr(self, "autoAnnotator", None) is not None:
          return True
      if AutoAnnotator is None:
          wx.MessageBox(
              "AutoAnnotator unavailable — is gradio_client installed?\n\n"
              f"Import error: {_autoImportError}",
              "Auto Annotate", wx.OK | wx.ICON_ERROR)
          return False
      try:
          self.autoAnnotator = AutoAnnotator()
      except Exception as e:
          wx.MessageBox(f"Could not initialise AutoAnnotator:\n{e}",
                        "Auto Annotate", wx.OK | wx.ICON_ERROR)
          return False
      return True

   def _applyGateSettings(self, classifier):
      """Push the GUI's decision-gate settings onto a classifier before a forward().

      Keeps the single- and two-stage paths on one definition. See
      mvc.inference.classifier_pnm.gate_tiles() for what the knobs do; note the threshold
      cuts on the selected mode's score, so it is not portable between modes."""
      if classifier is None:
          return
      classifier.maxProbabilityThreshold = float(self.classifierThreshold.GetValue() / 100.0)
      classifier.gateMode                = self.classifierGateMode.GetValue()
      classifier.assignBestDefectClass   = self.classifierBestDefectClass.GetValue()

   def _classifierDetectionsForImage(self, raw, min_dist=48):
      """Run the ACTIVE classifier on a raw frame (cv2.imread UNCHANGED output) and
      return (points, classNames) for non-clean detections in FULL mosaic coords
      (classifier works on the half-res demosaic -> points are scaled x2, the same
      convention as user clicks). Detections are greedily thinned to min_dist px.
      Uses the current GUI threshold/step/vote settings."""
      if (not useClassifier) or self.ClassifierPnm is None or raw is None:
          return [], []
      img = raw
      if img.ndim == 3 and img.shape[2] == 4:
          img = repackPolarToMosaic(img[:, :, 0], img[:, :, 1], img[:, :, 2], img[:, :, 3])
      if img.ndim != 2:
          return [], []
      self.ClassifierPnm.step = self.classifierTileSize.GetValue()
      self._applyGateSettings(self.ClassifierPnm)
      _hm, _occ, responses = self.ClassifierPnm.forward(
          img,
          majorityVote=self.classifierMajorityVoting.GetValue(),
          legend=False,
          erosion_kernel=self.erodeKernelSize.GetValue(),
          erosion_threshold=self.erodeThreshold.GetValue())
      label_map = {"NegativeDent": "Negative Dent", "PositiveDent": "Positive Dent",
                   "MaterialDefect": "Material Defect", "Deformation": "Deformation",
                   "Seal": "Seal", "Welding": "Welding", "Suspicious": "Suspicious"}
      pts, cls = [], []
      for (x, y), c in zip(responses.get("points", []), responses.get("classes", [])):
          base = c[len("class_"):] if c.startswith("class_") else c
          for suf in ("ClassA", "ClassB", "ClassC"):
              if base.endswith(suf):
                  base = base[:-len(suf)]
          name = label_map.get(base)
          if name is None:
              continue
          fx, fy = 2 * x, 2 * y
          if is_near_any(fx, fy, pts, min_dist):
              continue  # thin dense activation clusters to one point per min_dist
          pts.append((fx, fy)); cls.append(name)
      return pts, cls

   def _mergeClassifierIntoCurrentFrame(self):
      """Append the active classifier's detections for the CURRENT frame to the
      annotation points (skipping any near an existing point)."""
      raw = cv2.imread(self.filepath, cv2.IMREAD_UNCHANGED)
      pts, cls = self._classifierDetectionsForImage(raw)
      added = 0
      tileFull = 96  # classifier tile is 48 at half-res
      for (x, y), name in zip(pts, cls):
          if is_near_any(x, y, self.points_of_interest, tileFull):
              continue
          add_point(self.points_of_interest, self.points_classes,
                    self.points_severities, self.points_sources,
                    x, y, name, "AI", "classifier")
          added += 1
      if added:
          self._stat_points_added += added
          self.updatePointList()
          self.onView()
      self.instructLbl.SetLabel(f"Auto: merged {added} classifier detection(s) into this frame.")
      return added

   def onAuto(self, event):
      """Auto annotation: SAM3 pen-mark flow, plus the active classifier's
      detections when 'Use classifier in Auto annotation' is checked."""
      use_cls = self.autoUseClassifierCheckbox.GetValue()
      # classifier-only mode shouldn't error-popup about a missing SAM3 client
      sam_possible = (getattr(self, "autoAnnotator", None) is not None) or (AutoAnnotator is not None)
      if sam_possible or not use_cls:
          self._onAutoSAM(event)
      if use_cls:
          self._mergeClassifierIntoCurrentFrame()

   def _onAutoSAM(self, event):
      """SAM3 pen-mark assisted annotation (see AutoAnnotator.py):
        - blank frame that has a visible mark -> detect it and place a candidate point;
        - already-annotated frame             -> propagate the points to the NEXT frame.
      """
      if not self._ensureAutoAnnotator():
          return

      # --- Blank frame: detect mark(s) on THIS frame from scratch ---
      if not self.points_of_interest:
          prev_raw = cv2.imread(self.filepath, cv2.IMREAD_UNCHANGED)
          if prev_raw is None:
              wx.MessageBox("Could not load the current frame image.",
                            "Auto Annotate", wx.OK | wx.ICON_ERROR)
              return
          polarity = "high" if "positive" in self.defectComboBox.GetValue().lower() else "low"
          wx.BeginBusyCursor()
          try:
              dets = self.autoAnnotator.detect(prev_raw, polarity=polarity)
          except Exception as e:
              wx.EndBusyCursor()
              wx.MessageBox(f"Detection failed:\n{e}", "Auto Annotate",
                            wx.OK | wx.ICON_ERROR)
              return
          wx.EndBusyCursor()
          if not dets:
              self.instructLbl.SetLabel("Auto: no pen mark detected on this frame.")
              return
          for x, y, _area in dets:
              add_point(self.points_of_interest, self.points_classes,
                        self.points_severities, self.points_sources,
                        x, y,
                        self.defectComboBox.GetValue() or options[0],
                        self.severityComboBox.GetValue() or severities[0],
                        "auto")
          self._stat_points_added += len(dets)
          self.updatePointList()
          self.onView()
          self.instructLbl.SetLabel(
              "Auto: detected %u mark(s) on this frame — verify, then Save." % len(dets))
          return

      cur = self.folderStreamer.current()
      nxt = cur + 1
      if nxt > self.folderStreamer.max():
          wx.MessageBox("Already on the last frame; nothing to propagate to.",
                        "Auto Annotate", wx.OK | wx.ICON_INFORMATION)
          return

      prev_raw = cv2.imread(self.filepath, cv2.IMREAD_UNCHANGED)

      # Resolve the next frame's image path without disturbing the current selection.
      self.folderStreamer.select(nxt)
      next_path = self.folderStreamer.getImage()
      self.folderStreamer.select(cur)
      next_raw = cv2.imread(next_path, cv2.IMREAD_UNCHANGED)
      if prev_raw is None or next_raw is None:
          wx.MessageBox("Could not load the current or next frame image.",
                        "Auto Annotate", wx.OK | wx.ICON_ERROR)
          return

      # Snapshot annotations to carry class/severity across the association.
      prev_points     = list(self.points_of_interest)
      prev_classes    = list(self.points_classes)
      prev_severities = list(self.points_severities)

      wx.BeginBusyCursor()
      try:
          preds = self.autoAnnotator.propagate(prev_raw, prev_points, next_raw)
      except Exception as e:
          wx.EndBusyCursor()
          wx.MessageBox(f"Propagation failed:\n{e}", "Auto Annotate",
                        wx.OK | wx.ICON_ERROR)
          return
      wx.EndBusyCursor()

      # Advance to the next frame (this saves the current frame and loads N+1's JSON).
      self.gotoFrameUI(self._ui_from_stream(nxt))

      matched = 0
      for i, pred in enumerate(preds):
          if pred is None:
              continue
          add_point(self.points_of_interest, self.points_classes,
                    self.points_severities, self.points_sources,
                    pred[0], pred[1],
                    prev_classes[i] if i < len(prev_classes) else options[0],
                    prev_severities[i] if i < len(prev_severities) else severities[0],
                    "auto")
          matched += 1

      self._stat_points_added += matched
      self.updatePointList()
      self.onView()
      self.instructLbl.SetLabel(
          "Auto: propagated %u/%u annotation(s) to frame %u — verify, then Save."
          % (matched, len(prev_points), nxt))

   def _lightDirectionForFrame(self, idx):
      """Return the lightDirection label for stream frame idx. Uses the label stored
      in the frame's JSON when present; otherwise estimates it exactly like Finalize's
      batch pass (determine_intensity_region) and persists it into an existing JSON so
      it is not recomputed on later calls."""
      cur = self.folderStreamer.current()
      try:
          self.folderStreamer.select(idx)
          img_path = self.folderStreamer.getImage()
      finally:
          self.folderStreamer.select(cur)

      jp = annotation_json_path(img_path)
      data = read_annotation_json(jp)
      light = data.get("lightDirection", "Unknown")
      if light not in ("", "Unknown"):
          return light

      raw = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
      if raw is None:
          return "Unknown"
      light = determine_intensity_region(raw, threshold=0.1)
      if data:
          data["lightDirection"] = light
          write_annotation_json(jp, data, tag="Track: light direction")
      return light

   def _streamIndexOfFrame(self, name):
      """Stream index of the frame with this basename, or None.
      FolderStreamer keeps its frames in directoryList, HTTPFolderStreamer in
      file_list; extensions may differ between the remote listing (.pnm) and
      the local cache (.png), so frames are matched on the stem."""
      frames = (getattr(self.folderStreamer, "directoryList", None)
                or getattr(self.folderStreamer, "file_list", None) or [])
      stem = os.path.splitext(name)[0]
      for i, p in enumerate(frames):
          if os.path.splitext(os.path.basename(p))[0] == stem:
              return i
      return None

   def _findSameLightFrame(self, nxt, next_path, direction=1):
      """Return (path, similarity) of the frame lit most like frame nxt among the
      SAME_LIGHT_SEARCH_MAX frames on the source side of it (behind when tracking
      forward, ahead when tracking backward), or (None, 0.0) when nothing scores
      above SAME_LIGHT_MIN_SIMILARITY. The adjacent source frame is excluded — it
      is already tracking record [0]."""
      fp_next = lighting_fingerprint_cached(next_path, self._light_fp_cache)
      if fp_next is None:
          return None, 0.0
      cur = self.folderStreamer.current()
      n_total = self.folderStreamer.max()
      best_path, best_sim = None, 0.0
      try:
          for k in range(2, 2 + SAME_LIGHT_SEARCH_MAX):
              idx = nxt - direction * k
              if not (0 <= idx < n_total):
                  break
              self.folderStreamer.select(idx)
              path = self.folderStreamer.getImage()
              fp = lighting_fingerprint_cached(path, self._light_fp_cache)
              if fp is None:
                  continue
              sim = float(fp_next @ fp)
              if sim > best_sim:
                  best_path, best_sim = path, sim
      finally:
          self.folderStreamer.select(cur)
      if best_sim < SAME_LIGHT_MIN_SIMILARITY:
          return None, 0.0
      return best_path, best_sim

   def onTrack(self, event):
      """Track →: propagate the current frame's points to the next frame."""
      self._track(direction=1)

   def onTrackBack(self, event):
      """Track ←: propagate the current frame's points to the previous frame."""
      self._track(direction=-1)

   def _track(self, direction):
      """Propagate the current frame's points to the adjacent frame in the given
      direction (+1 next / -1 previous) using a locally estimated similarity
      transform (blockwise phase correlation — no SAM3 server needed), then move
      to it. The estimated inter-frame transforms are stored in the destination
      frame's JSON under 'tracking': a list of records, one per reference frame —
      the source frame plus (when found) the best same-lighting frame."""
      if not self.points_of_interest:
          self.instructLbl.SetLabel("Track: no points on this frame to propagate.")
          return

      cur = self.folderStreamer.current()
      nxt = cur + direction
      if not (0 <= nxt < self.folderStreamer.max()):
          wx.MessageBox("Already on the %s frame; nothing to track to."
                        % ("last" if direction > 0 else "first"),
                        "Track", wx.OK | wx.ICON_INFORMATION)
          return

      prev_path = self.filepath
      # Resolve the destination frame's image path without disturbing the current
      # selection (this also forces the download when streaming over HTTP).
      self.folderStreamer.select(nxt)
      next_path = self.folderStreamer.getImage()
      self.folderStreamer.select(cur)

      # Estimate (and later persist) the destination's light direction label so
      # tracking never runs ahead of the Finalize light pass.
      target_light = self._lightDirectionForFrame(nxt)
      # Same-lighting reference: the frame lit most like the destination, by
      # lighting fingerprint, on the source side of it.
      same_light_path, same_light_sim = self._findSameLightFrame(nxt, next_path, direction)

      # Smooth-motion prior: the adjacent-frame record already on THIS frame,
      # sign-corrected for our direction of travel.
      prior_shift = None
      if self.tracking:
          prior_shift = prior_shift_from_record(
              self.tracking[0], cur, direction,
              self._streamIndexOfFrame(self.tracking[0].get("fromFrame", "")))

      wx.BeginBusyCursor()
      try:
          M, (dx, dy), response, inliers = estimateFrameAffine(prev_path, next_path)
      except Exception as e:
          wx.EndBusyCursor()
          wx.MessageBox(f"Transform estimation failed:\n{e}", "Track",
                        wx.OK | wx.ICON_ERROR)
          return
      records = []
      if same_light_path:
          try:
              sM, (sdx, sdy), sresp, sinl = estimateFrameAffine(same_light_path, next_path)
              records.append(tracking_record(same_light_path, sM, sdx, sdy, sresp, sinl,
                                             light_similarity=same_light_sim))
          except Exception as e:
              print("Track: same-lighting transform estimation failed:", e)
      wx.EndBusyCursor()

      # Trust the block consensus when it exists; otherwise a weak global response
      # falls back to the smooth-motion prior.
      fallback = False
      if response < 0.05 and not inliers and prior_shift:
          dx, dy = prior_shift
          M = np.float64([[1, 0, dx], [0, 1, dy]])
          fallback = True

      # Snapshot annotations to carry class/severity across the shift.
      prev_points     = list(self.points_of_interest)
      prev_classes    = list(self.points_classes)
      prev_severities = list(self.points_severities)

      # Move to the destination frame (this saves the current frame and loads its JSON).
      self.gotoFrameUI(self._ui_from_stream(nxt))

      # Persist the light estimate computed above if the frame didn't have one yet.
      if target_light not in ("", "Unknown", "No Light") and \
         self.lightComboBox.GetValue() in ("", "Unknown"):
          self.lightComboBox.SetValue(target_light)

      W, H = self.viewedImageFullWidth, self.viewedImageFullHeight
      carried_pts, carried_cls, carried_sev = propagate_points(
          prev_points, prev_classes, prev_severities, M, W, H, 30.0,
          options[0], severities[0], self.points_of_interest)
      for (x, y), cls, sev in zip(carried_pts, carried_cls, carried_sev):
          add_point(self.points_of_interest, self.points_classes,
                    self.points_severities, self.points_sources,
                    x, y, cls, sev, "auto")
      carried = len(carried_pts)

      self.tracking = [tracking_record(prev_path, M, dx, dy, response, inliers,
                                       fallback=fallback)] + records

      self._stat_points_added += carried
      self.updatePointList()
      self.onView()
      self.onSave(None)
      rot = np.degrees(np.arctan2(M[1, 0], M[0, 0]))
      self.instructLbl.SetLabel(
          "Track: carried %u/%u point(s) to frame %u (shift %+.0f,%+.0f rot %+.2f° "
          "resp %.2f, %u blocks%s) — verify."
          % (carried, len(prev_points), nxt, dx, dy, rot, response, inliers,
             ", fallback" if fallback else ""))

   def onFillTracking(self, event):
      """Fill Tracking button: confirm, then run the batch tracking pass and report."""
      images = dataset_images(self.folderStreamer, self.filepath)
      if len(images) < 2:
          wx.MessageBox("Need at least two frames in a dataset.", "Fill Tracking",
                        wx.OK | wx.ICON_WARNING)
          return

      ask = wx.MessageDialog(
          self.frame,
          f"Measure inter-frame tracking for {len(images)} frames?\n\n"
          f"Frames that already have tracking records are skipped, then a global\n"
          f"least-squares pass reconciles all measurements. Annotations are untouched.",
          "Fill Tracking", wx.YES_NO | wx.ICON_QUESTION)
      if ask.ShowModal() != wx.ID_YES:
          ask.Destroy()
          return
      ask.Destroy()

      res = self._fillTracking()
      if res is None:
          return
      filled, skipped, failed, solved, aborted = res
      wx.MessageBox(
          f"Fill Tracking complete.\n\n"
          f"Measured {filled} frame(s), {skipped} already had tracking, {failed} failed.\n"
          f"Least-squares positions stored for {solved} frame(s)."
          + ("\n(Aborted before completion.)" if aborted else ""),
          "Fill Tracking", wx.OK | wx.ICON_INFORMATION)

   def _fillTracking(self):
      """Batch tracking pass over the whole dataset (also run by Finalize):
        PASS 1 — every frame without 'tracking' records gets the measured transform
                 from its previous frame plus the best same-lighting link.
        PASS 2 — all measurements are reconciled with a weighted least-squares solve
                 of the resulting pose graph, and each frame's optimized global
                 position (relative to the first frame) is stored as an extra
                 'leastSquaresGlobal' record.
      Annotation JSONs are read-modified-written, so points/classes are untouched.
      The pass itself lives in mga.core.dataset_finalize.fill_tracking (Stage 3c of
      this file's refactor); this wrapper owns the streamer, the progress dialog
      and the reload.
      Returns (filled, skipped, failed, solved, aborted), or None with <2 frames."""
      images = dataset_images(self.folderStreamer, self.filepath)
      if len(images) < 2:
          return None

      # Flush the currently-open frame so its JSON is up to date before the pass.
      self.onSave(None)

      prog = wx.ProgressDialog(
          "Fill Tracking", "Measuring inter-frame shifts…", maximum=len(images),
          parent=self.frame,
          style=wx.PD_APP_MODAL | wx.PD_AUTO_HIDE | wx.PD_CAN_ABORT
                | wx.PD_ELAPSED_TIME | wx.PD_REMAINING_TIME)

      def progress(i, msg):
          cont, _ = prog.Update(i, msg)
          wx.GetApp().Yield(True)
          return bool(cont)

      res = fill_tracking(images, self._light_fp_cache, progress=progress,
                          frame_size=(self.width, self.height))
      prog.Destroy()

      # The pass may have rewritten the open frame's JSON — reload it.
      self.onProcessNewImageSample(self.filepath)
      self.onView()
      return res

   def onNudgeTracking(self, event):
      """Manual tracking fix-up: shift every auto-sourced point on the current frame
      as one block with arrow buttons. The correction is folded into tracking
      record [0] and marked method 'manual', which the Fill Tracking optimizer
      treats as ground truth."""
      if not any(s == "auto" for s in self.points_sources):
          self.instructLbl.SetLabel("Nudge: no auto points on this frame to adjust.")
          return

      dlg = wx.Dialog(self.frame, title="Nudge Tracking")
      panel = wx.Panel(dlg)
      step = wx.SpinCtrl(panel, min=1, max=200, initial=10)

      def nudge(ddx, ddy):
          nudge_auto_points(self.points_of_interest, self.points_sources, ddx, ddy)
          if self.tracking:
              nudge_tracking_record(self.tracking[0], ddx, ddy)
          self.updatePointList()
          self.onView()

      def rotate(deg):
          # +deg turns clockwise on screen (y points down)
          c = rotate_auto_points(self.points_of_interest, self.points_sources, deg)
          if c is None:
              return
          if self.tracking:
              rotate_tracking_record(self.tracking[0], deg, c[0], c[1])
          self.updatePointList()
          self.onView()

      up    = wx.Button(panel, label="▲");  up.Bind(wx.EVT_BUTTON,    lambda e: nudge(0, -step.GetValue()))
      down  = wx.Button(panel, label="▼");  down.Bind(wx.EVT_BUTTON,  lambda e: nudge(0,  step.GetValue()))
      left  = wx.Button(panel, label="◀");  left.Bind(wx.EVT_BUTTON,  lambda e: nudge(-step.GetValue(), 0))
      right = wx.Button(panel, label="▶");  right.Bind(wx.EVT_BUTTON, lambda e: nudge( step.GetValue(), 0))
      rotL  = wx.Button(panel, label="↺ Left 3°");  rotL.Bind(wx.EVT_BUTTON,  lambda e: rotate(-3))
      rotR  = wx.Button(panel, label="↻ Right 3°"); rotR.Bind(wx.EVT_BUTTON,  lambda e: rotate( 3))
      done  = wx.Button(panel, label="Done"); done.Bind(wx.EVT_BUTTON, lambda e: dlg.EndModal(wx.ID_OK))

      grid = wx.GridSizer(3, 3, 2, 2)
      for item in (wx.StaticText(panel), up, wx.StaticText(panel),
                   left, done, right,
                   wx.StaticText(panel), down, wx.StaticText(panel)):
          grid.Add(item, 0, wx.EXPAND)
      col = wx.BoxSizer(wx.VERTICAL)
      rotRow = wx.BoxSizer(wx.HORIZONTAL)
      rotRow.Add(rotL, 0, wx.RIGHT, 4)
      rotRow.Add(rotR, 0)
      row = wx.BoxSizer(wx.HORIZONTAL)
      row.Add(wx.StaticText(panel, label="Step (mosaic px):"), 0,
              wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 4)
      row.Add(step, 0)
      col.Add(grid, 0, wx.ALL | wx.ALIGN_CENTER, 8)
      col.Add(rotRow, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM | wx.ALIGN_CENTER, 8)
      col.Add(row, 0, wx.ALL | wx.ALIGN_CENTER, 8)
      panel.SetSizerAndFit(col)
      dlg.Fit()
      dlg.ShowModal()
      dlg.Destroy()
      self.onSave(None)

   def onFullAuto(self, event):
      """Run SAM3 pen-mark detection across EVERY frame in the open dataset and write
      candidate annotations where a circle is found. Frames that already have
      annotations are skipped so manual work is never clobbered."""
      if not self._ensureAutoAnnotator():
          return

      # The frame list is already loaded in the streamer (local_dir is not always set,
      # e.g. when started via --from); fall back to the current frame's directory.
      images = dataset_images(self.folderStreamer, self.filepath)
      if not images:
          wx.MessageBox("No dataset frames are loaded.\nOpen a dataset directory first.",
                        "Full Auto", wx.OK | wx.ICON_WARNING)
          return

      defect   = self.defectComboBox.GetValue() or options[0]
      severity = self.severityComboBox.GetValue() or severities[0]
      polarity = "high" if "positive" in defect.lower() else "low"

      ask = wx.MessageDialog(
          self.frame,
          f"Run SAM3 auto-detection on {len(images)} frames?\n\n"
          f"Detected marks will be labelled '{defect}' / '{severity}'.\n"
          f"Frames that already have annotations are SKIPPED.\n"
          f"This queries the SAM3 server once per frame and may take a while.",
          "Full Auto", wx.YES_NO | wx.ICON_QUESTION)
      if ask.ShowModal() != wx.ID_YES:
          ask.Destroy()
          return
      ask.Destroy()

      prog = wx.ProgressDialog(
          "Full Auto", "Starting…", maximum=len(images), parent=self.frame,
          style=wx.PD_APP_MODAL | wx.PD_AUTO_HIDE | wx.PD_CAN_ABORT
                | wx.PD_ELAPSED_TIME | wx.PD_REMAINING_TIME)

      # PASS 1 — detect on every eligible frame (the slow, SAM3-bound part).
      annotated = marks = skipped = failed = 0
      frame_cands = []       # (scan_index, detect_ex candidates, json_path)
      use_classifier = self.autoUseClassifierCheckbox.GetValue()
      classifier_dets = {}   # json_path -> (points_fullres, classNames)
      for i, img in enumerate(images):
          cont, _ = prog.Update(
              i, f"{i+1}/{len(images)} — detecting… {skipped} skipped")
          if not cont:
              break
          wx.GetApp().Yield(True)

          json_path = annotation_json_path(img)

          # Skip frames that already carry annotations.
          if read_annotation_json(json_path).get("pointClicks"):
              skipped += 1
              continue

          raw = cv2.imread(img, cv2.IMREAD_UNCHANGED)
          if raw is None:
              failed += 1
              continue
          try:
              cands = self.autoAnnotator.detect_ex(raw, polarity=polarity)
          except Exception as e:
              prog.Destroy()
              wx.MessageBox(f"Detection failed on:\n{img}\n\n{e}",
                            "Full Auto", wx.OK | wx.ICON_ERROR)
              return
          if use_classifier:
              c_pts, c_cls = self._classifierDetectionsForImage(raw)
              if c_pts:
                  classifier_dets[json_path] = (c_pts, c_cls)
          if cands:
              frame_cands.append((i, cands, json_path))

      # PASS 2 — temporal offset consensus across the scan (pen ring and defect are
      # physical, so the ring->defect offset is constant: the median over a ring track
      # fixes frames whose own DoLP anomaly is weak or missing), then write the JSONs.
      final = temporal_consensus([(i, c) for i, c, _jp in frame_cands])
      classifier_marks = 0
      for i, _c, json_path in frame_cands:
          dets = final.get(i, [])
          if not dets and json_path not in classifier_dets:
              continue
          points     = [[x, y] for x, y, _a in dets]
          classes    = [defect] * len(dets)
          sevs       = [severity] * len(dets)
          sources    = ["auto"] * len(dets)
          # merge classifier detections that don't overlap a SAM point
          c_pts, c_cls = classifier_dets.pop(json_path, ([], []))
          for (cx, cy), cname in zip(c_pts, c_cls):
              if is_near_any(cx, cy, points, 96):
                  continue
              points.append([cx, cy]); classes.append(cname)
              sevs.append("AI"); sources.append("classifier")
              classifier_marks += 1
          data = empty_annotation(self.width, self.height,
                                  points=points, classes=classes,
                                  severities=sevs, sources=sources)
          if write_annotation_json(json_path, data, tag="[Full Auto]"):
              annotated += 1
              marks     += len(dets)
          else:
              failed += 1

      # Frames where ONLY the classifier found something (no pen mark)
      for json_path, (c_pts, c_cls) in classifier_dets.items():
          data = empty_annotation(self.width, self.height,
                                  points=[[x, y] for x, y in c_pts],
                                  classes=list(c_cls),
                                  severities=["AI"] * len(c_pts),
                                  sources=["classifier"] * len(c_pts))
          if write_annotation_json(json_path, data, tag="[Full Auto]"):
              annotated += 1
              classifier_marks += len(c_pts)
          else:
              failed += 1

      prog.Destroy()
      wx.MessageBox(
          f"Full Auto complete.\n\n"
          f"Annotated {marks} pen mark(s) + {classifier_marks} classifier "
          f"detection(s) across {annotated} frame(s).\n"
          f"Skipped {skipped} already-annotated frame(s).\n"
          f"{failed} frame(s) failed.",
          "Full Auto", wx.OK | wx.ICON_INFORMATION)

      # Refresh the current frame in case it was (re)annotated.
      self.onProcessNewImageSample(self.filepath)
      self.onView()

   def onSave(self, event):
        print("Save")
        if not self.filepath or not os.path.isfile(self.filepath):
            print("onSave: skipping — filepath is not a valid file:", self.filepath)
            return

        allData = annotation_to_dict(
            self.points_of_interest, self.points_classes, self.points_severities,
            self.points_sources, self.regions_of_interest,
            self.width, self.height, self.filehash,
            tenengrad=self.tenengrad_focus_measure,
            light_direction=self.lightComboBox.GetValue(),
            tracking=self.tracking)

        # <-- colorFrame_0_00047.json when no historical scheme exists yet
        primary_json = annotation_json_path(self.filepath)

        if write_annotation_json(primary_json, allData):
            self.folderStreamer.saveJSON()



   def cleanThisFrameMetaData(self):
               self.pointList.Clear()
               self.regionList.Clear()
               self.points_classes      = []
               self.points_severities   = []
               self.points_sources      = []
               self.regions_of_interest = []
               self.points_of_interest  = []
               self.lightComboBox.SetValue("Unknown")
               self.tenengrad_focus_measure = 0.0  # restored from JSON below if the frame has a saved value
               self.tracking            = None     # restored from JSON below if the frame has a saved value



   def sensibleDefaults(self,loadDatasetCase):
               #Small check (this will need to  be updated if defects change)..
               for widget, value in dataset_defaults(loadDatasetCase):
                   getattr(self, widget).SetValue(value)

   def openDataset(self, base_dir, streamer, is_directory=True):
    """
    base_dir: local path where info.json/controller.csv/tactile live
              (for network: the cache dir, e.g. selectedDirectory)
    streamer: FolderStreamer or HTTPFolderStreamer
    is_directory: whether we should run directory-mode behaviors
    """
    self.folderStreamer = streamer
    self.filePathIsDirectory = is_directory
    self.lastFrameFile = os.path.join(base_dir, "last.frame")  # per-dataset, next to the JSONs
    self._resetSessionStats()   # effort statistics accumulate per dataset session

    # Load metadata/controls/sensors
    self.populateMetaData(f"{base_dir}/info.json")
    self.loadControlsCSV(f"{base_dir}/controller.csv")

    # If tactile plots exist under base_dir/tactile/
    self._loadSensorPlotsNewDataset(directory=f"{base_dir}/tactile/")

    # Apply startFrame/endFrame
    self._applyDatasetRangeFromMetadata()

    # Configure slider to reflect range length (relative)
    self.scrollBar.SetMin(0)
    self.scrollBar.SetMax(self._ui_max())

    self.sensibleDefaults(base_dir)

    # Jump to the last frame the user was on (last.frame), or the first in range
    self.gotoFrameUI(self._restoreLastFrameUI())

    # Optional: reset placeholders
    self._initializeSensorPlotPlaceholders(parent=self.controlsPanel)

   def gotoFrameUI(self, ui_idx):
    ui_idx = max(0, min(ui_idx, self._ui_max()))
    self.scrollBar.SetValue(ui_idx)

    stream_idx = self._stream_from_ui(ui_idx)
    self.folderStreamer.select(stream_idx)

    if self.filePathIsDirectory:
        self.onSave(None)


    newFilepath = self.folderStreamer.getImage()
    if not newFilepath:
        # Network stall/failure: the streamer could not provide the frame.
        # Stay on the current frame rather than crashing on a None path.
        print(f"gotoFrameUI: could not fetch image for frame {stream_idx}; staying put")
        return

    self.filepath = newFilepath
    self._rememberLastFrame(self.filepath)
    self.onProcessNewImageSample(self.filepath)
    self.updateMinMaxSlider()
    #self.onView() # redundant: onProcessNewImageSample already calls onView()

   def _rememberLastFrame(self, filepath):
    """Persist the current frame's basename to the dataset's last.frame so
    reopening the editor can restore where the user left off (see
    _restoreLastFrameUI). Stored per-dataset: frame names are generic
    (colorFrame_0_XXXXX.png), so a shared file would leak across datasets."""
    if not filepath or not self.lastFrameFile:
        return
    try:
        with open(self.lastFrameFile, "w") as f:
            f.write(os.path.basename(filepath))
    except Exception as e:
        print("Could not write %s:" % self.lastFrameFile, e)

   def _restoreLastFrameUI(self):
    """UI index of the frame recorded in the dataset's last.frame, clamped to
    the current range, or 0 when the file is missing or the frame is not in
    this dataset."""
    if not self.lastFrameFile:
        return 0
    try:
        with open(self.lastFrameFile) as f:
            name = f.read().strip()
    except Exception:
        return 0
    stream_idx = self._streamIndexOfFrame(name)
    if stream_idx is None:
        return 0
    return max(0, min(self._ui_from_stream(stream_idx), self._ui_max()))



   def _renderFrame(self, filepath, way, brightness, contrast):
       """Pure decode + render for the classifier-off path: returns
       {'foreground': <way/DoLP render>, 'processed': <RGB base render>} or None.
       Touches no UI and no shared state, so it is safe to run in a worker thread
       (only reads self.PhotoMaxSize* via rescaleCVMAT, which is stable during navigation)."""
       canonical = self._canonCache if self.canonicalLightCheckbox.GetValue() else None
       imgPNM, imgCV = loadFrameMosaic(filepath, canonical_cache=canonical)
       if imgPNM is None:
           return None
       src, w = imgCV, way
       if w == 3:                       # Sobel pre-step, mirrors onProcessNewImageSample
           src = detect_sobel_edges(src)
           w = 0
       foreground = self.rescaleCVMAT(convertPolarCVMATToRGB(src, way=w, brightness=brightness, contrast=contrast))
       rgba = readPolarPNMToRGBALive(imgPNM)
       base = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
       processed = self.rescaleCVMAT(convertRGBCVMATToRGB(base, brightness=brightness, contrast=contrast))
       return {'foreground': foreground, 'processed': processed}

   def _takePrefetch(self, filepath, way, brightness, contrast):
       """Consume the background-rendered result for this exact frame+params, or None."""
       key = (filepath, way, brightness, contrast)
       with self._prefetch_lock:
           if self._prefetch is not None and self._prefetch[0] == key and self._prefetch[1] is not None:
               data = self._prefetch[1]
               self._prefetch = None
               return data
       return None

   def _schedulePrefetch(self, way, brightness, contrast):
       """Render the NEXT frame in a background thread so the forward transition is instant.
       One worker at a time; the result is keyed so stale params are simply ignored on use."""
       # Don't prefetch during auto-play: frames advance faster than a render finishes, so the
       # worker would just compete with the main thread for CPU/GIL and freeze the UI.
       if getattr(self, "isPlaying", False):
           return
       if self._prefetch_thread is not None and self._prefetch_thread.is_alive():
           return
       try:
           cur = self.folderStreamer.current()
           nxt = cur + 1
           if nxt >= self.folderStreamer.max():   # valid indices are 0..max()-1
               return
           self.folderStreamer.select(nxt)
           next_path = self.folderStreamer.getImage()
           self.folderStreamer.select(cur)
       except Exception:
           return
       if not next_path:
           return
       key = (next_path, way, brightness, contrast)
       with self._prefetch_lock:
           if self._prefetch is not None and self._prefetch[0] == key:
               return
       def work():
           data = None
           try:
               data = self._renderFrame(next_path, way, brightness, contrast)
           except Exception as e:
               print("prefetch render failed:", e)
           with self._prefetch_lock:
               self._prefetch = (key, data)
       self._prefetch_thread = threading.Thread(target=work, daemon=True)
       self._prefetch_thread.start()

   def _restoreFrameAnnotations(self, filepath):
       """Load this frame's saved annotations: trust the streamer's JSON first
       (the HTTP streamer downloads stem.json), then the local legacy resolver."""
       jsonPath = self.folderStreamer.getJSON()
       print(" self.folderStreamer.getJSON() = ", jsonPath, " ")

       # 1) Trust the streamer's answer first (HTTP streamer downloads stem.json)
       if jsonPath is not None and checkIfFileExists(jsonPath):
           print("There are saved data that need to be restored here (", jsonPath, ")")
           self.restoreFromJSON(jsonPath)
       else:
           # 2) Fallback to resolver (local legacy compatibility)
           resolved = resolve_annotation_json_path(filepath, prefer_existing=True)
           if resolved is not None and checkIfFileExists(resolved):
               print("There are saved data that need to be restored here (", resolved, ")")
               self.restoreFromJSON(resolved)
           else:
               print("No annotations found for ", filepath, " / ", resolved)

   def _renderForeground(self, imgCV, cached):
       """Right-panel visualization: the polarization/DoLP/... render of the frame
       (with the Sobel pre-step), or the prefetched render when it matches."""
       if cached is not None:
           return cached['foreground']
       if (self.processingWay==3):
           imgCV = detect_sobel_edges(imgCV)
           self.processingWay=0
       return self.rescaleCVMAT(convertPolarCVMATToRGB(imgCV,way=self.processingWay,brightness=self.brightness_offset, contrast=self.contrast_offset))

   def _runClassifierOnFrame(self, imgPNM):
       """Run the ACTIVE classifier (1-stage or 2-stage ensemble, per the GUI
       switches) on the frame, update the correlation stats against the user's
       annotations, and return the rescaled classifier visualization for the
       left panel — or None when the classifier is disabled or suppressed (the
       boot-logo "default" dataset), so the caller shows the plain base render."""
       if not (useClassifier and not self.classifierDisabledCheckbox.GetValue()): #<- Only use classifier when classifier is on
           return None

       imgRGBFromClassifier = None
       if self.photoTxt.GetValue() != "default": #<- Don't trigger classification in logo "default dataset" when application boots
         self.AIAnnotations=None
         if self.classifierTwoStage.GetValue() and getattr(self, 'EnsembleClassifierPnm', None) is None:
            print("2-stage ensemble requested but no allclass_* models are loaded — using single classifier")
            self.classifierTwoStage.SetValue(False)
         if self.classifierTwoStage.GetValue():
            print("Image classification done through 2-stage ensemble classifier")
            self.EnsembleClassifierPnm.step = self.classifierTileSize.GetValue()
            self._applyGateSettings(self.EnsembleClassifierPnm) #parallel=True	Re-tiles the full image per model (selected-tile optimization is lost); Python GIL + shared CUDA queue limits real overlap.
            imgRGBFromClassifier, occupancy, self.AIAnnotations = self.EnsembleClassifierPnm.forward(imgPNM, majorityVote=self.classifierMajorityVoting.GetValue(), parallel=False, multimodel=self.parallellTwoStage.GetValue())
            imgRGBFromClassifier = self.rescaleCVMAT(convertRGBCVMATToRGB(imgRGBFromClassifier,brightness=self.brightness_offset, contrast=self.contrast_offset))
            self.classifierInfo.SetLabel("2-stage: %0.2f Hz" % self.EnsembleClassifierPnm.hz)
         else:
            print("Image classification done through regular 1-stage classifier")
            self.ClassifierPnm.step = self.classifierTileSize.GetValue()
            self._applyGateSettings(self.ClassifierPnm)
            imgRGBFromClassifier,occupancy, self.AIAnnotations = self.ClassifierPnm.forward(imgPNM, majorityVote=self.classifierMajorityVoting.GetValue(), erosion_kernel=self.erodeKernelSize.GetValue(),erosion_threshold=self.erodeThreshold.GetValue())
            imgRGBFromClassifier = self.rescaleCVMAT(convertRGBCVMATToRGB(imgRGBFromClassifier,brightness=self.brightness_offset, contrast=self.contrast_offset))
            self.classifierInfo.SetLabel("1-stage: %0.2f Hz" % self.ClassifierPnm.hz)

       # Correlation stats against the user's annotations (matches the original,
       # which also ran this on the boot logo with AIAnnotations=None). The
       # classifier runs on the DEMOSAICED half-res image, so AI points scale x2
       # to the user-click (full mosaic) coords.
       current_hz = (self.EnsembleClassifierPnm.hz
                     if self.classifierTwoStage.GetValue()
                     else self.ClassifierPnm.hz)
       ai_ann_scaled = self.AIAnnotations
       if ai_ann_scaled and ai_ann_scaled.get("points"):
           ai_ann_scaled = dict(ai_ann_scaled)
           ai_ann_scaled["points"] = [(2 * x, 2 * y) for (x, y) in ai_ann_scaled["points"]]
       self.stats.update(
                        frame_id=self.filepath,
                        user_ann={
                                  "points":     self.points_of_interest,
                                  "classes":    self.points_classes,
                                  "severities": self.points_severities,
                                 },
                        ai_ann=ai_ann_scaled,
                        hz=current_hz
                      )
       self.classifierInfo.SetLabel(self.stats.get_summary_string())
       return imgRGBFromClassifier

   def _classifierBaseImage(self, imgPNM, cached):
       """Plain RGBA base render for the left panel when the classifier is off:
       the prefetched render when it matches, otherwise a fresh decode."""
       if cached is not None:
           return cached['processed']
       rgba = readPolarPNMToRGBALive(imgPNM)
       classifierBaseImg = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
       return self.rescaleCVMAT(convertRGBCVMATToRGB(classifierBaseImg, brightness=self.brightness_offset, contrast=self.contrast_offset))

   def onProcessNewImageSample(self,filepath):
           # Always start from a clean frame; we may restore JSON or apply carried points below
           self.cleanThisFrameMetaData()
           self._pf_next = None  # set below if this frame is prefetch-eligible


           print("onProcessNewImageSample (", filepath, ") ")
           self._restoreFrameAnnotations(filepath)

           
           ui_idx = self.scrollBar.GetValue()
           stream_idx = self._stream_from_ui(ui_idx)

           if hasattr(self, 'controlsData'):
               if 0 <= stream_idx < len(self.controlsData):
                   self.updateControlsTab(self.controlsData[stream_idx], sample_number=stream_idx)



           # Render the polarization image for both view panels
           global combineChannels
           #imgCV  = cv2.imread(filepath) #,cv2.IMREAD_UNCHANGED  # wasteful: this decode is overwritten below for 4-ch polar PNGs; imgCV is derived from imgPNM instead
           # if we got a 4-channel PNG (p0,p45,p90,p135), repack to the original 2x2
           # mosaic; canonical-light remap when the checkbox is on (see loadFrameMosaic)
           canonical = self._canonCache if self.canonicalLightCheckbox.GetValue() else None
           imgPNM, imgCV = loadFrameMosaic(filepath, canonical_cache=canonical)

           if (imgPNM is None):
                  print("Could not load ",filepath)
                  return


           print("Raw image dims for ",filepath," ",imgCV.shape)
           self.viewedImageFullWidth  = imgCV.shape[1]
           self.viewedImageFullHeight = imgCV.shape[0] 

           if self.calcFocusLightCheckbox.GetValue():
               self.tenengrad_focus_measure = tenengrad_focus_measure(imgCV)
               print("Focus : ",self.tenengrad_focus_measure)
           # else: leave focus as-is — it's 0.0 for a fresh frame, or the value restored from JSON
           
           processingString = self.ProcessorComboBox.GetValue()
           if self.photoTxt.GetValue() == "default":
               processingString = processors[0]
           self.processingWay = PROCESSOR_WAYS.get(processingString, self.processingWay)

           if combineChannels:
              print("Image CV Combining all channels to one")
              # Prefetch fast path: when the classifier is off, the heavy renders below are a
              # pure function of (filepath, way, brightness, contrast) and may already be
              # computed by the background worker for this frame.
              fast_eligible = (not (useClassifier and not self.classifierDisabledCheckbox.GetValue())
                               and self.photoTxt.GetValue() != "default")
              cached = None
              if fast_eligible:
                  key_way = self.processingWay  # capture before the Sobel branch mutates it
                  cached = self._takePrefetch(filepath, key_way, self.brightness_offset, self.contrast_offset)
                  self._pf_next = (key_way, self.brightness_offset, self.contrast_offset)

              # Right panel: the polarization/DoLP/... visualization (prefetched when it matches)
              imgCV = self._renderForeground(imgCV, cached)

              # Left panel: the classifier visualization when it ran, the plain
              # RGBA base render otherwise.
              processed_img = self._runClassifierOnFrame(imgPNM)
              if processed_img is None:
                  processed_img = self._classifierBaseImage(imgPNM, cached)
              self.leftViewImage = processed_img

              if (self.lightComboBox.GetValue()=="Unknown"): #If we don't have a light orientation set
               print("We don't know Light Direction")
               if (self.calcFocusLightCheckbox.GetValue()):   #If we are ok with guessing 
                 print("We will try to guess light direction")
                 self.lightComboBox.SetValue(determine_intensity_region(imgCV, threshold=0.1))

              self.rightViewImage = imgCV
           else:
              processed_img                      = imgCV
              self.leftViewImage           = imgCV
              self.rightViewImage = imgCV
           self.width  = processed_img.shape[1]
           self.height = processed_img.shape[0]
           self.viewedImageViewWidth  = self.width
           self.viewedImageViewHeight = self.height
           print("Cast to WxBitmap width / height for ",filepath," ",self.width,"x",self.height)

           self.clickRatioX = self.viewedImageFullWidth / self.viewedImageViewWidth 
           self.clickRatioY = self.viewedImageFullHeight / self.viewedImageViewHeight 

           self.imageCtrl.SetBitmap(wx.Bitmap.FromBuffer(self.width, self.height, processed_img))


           self.onView()

           bmp = self.imageCtrl.GetBitmap()
           if hasattr(self, 'magnifier') and self.magnifier and bmp.IsOk():
               img = bmp.ConvertToImage()
               self.magnifier.setImage(img)
               self.magnifier.refreshZoom()

           # Kick off the background render of the NEXT frame so the forward transition is instant.
           if self._pf_next is not None:
               self._schedulePrefetch(*self._pf_next)


   def onAbout(self, event):
        wx.MessageBox("Written by Ammar Qammaz a.k.a. AmmarkoV\nhttp://ammar.gr/\nVersion %s\nhttps://github.com/magician-project/magician_grabber_annotator\nPsalm 32:8"%version, "About", wx.OK | wx.ICON_INFORMATION)

   def onRescan(self, newPath):
        """Refresh / re-render the current frame (triggered by the 'R' key)."""
        self.onProcessNewImageSample(self.filepath)

   # ---- annotation-effort statistics -------------------------------------------------
   def _recordInteraction(self):
        """Accumulate active annotation time between consecutive interactions, ignoring
        long idle gaps so the total reflects real effort (for man-month estimates)."""
        now = time.time()
        last = self._stat_last_interaction
        if last is not None:
            gap = now - last
            if gap < self._STAT_IDLE_CAP:
                self._stat_active_seconds += gap
        self._stat_last_interaction = now

   def _resetSessionStats(self):
        self._stat_clicks = 0
        self._stat_keystrokes = 0
        self._stat_points_added = 0
        self._stat_points_deleted = 0
        self._stat_active_seconds = 0.0
        self._stat_last_interaction = None

   def _datasetDefectTotals(self, local_dir):
        """Scan every frame JSON in the dataset and tally defect classes and
        severities — see mga.core.dataset_finalize.defect_totals (Stage 3c)."""
        return defect_totals(local_dir)

   def _batchComputeFocusLight(self, local_dir):
        """Compute Tenengrad focus + a latency-corrected light direction for every
        frame and store them in each frame's JSON (light decoded sequentially by
        lightDecoder, 'No Light' flag preserved). The pass itself lives in
        mga.core.dataset_finalize.compute_focus_light (Stage 3c of this file's
        refactor); this wrapper owns the streamer and the progress dialog."""
        images = dataset_images(self.folderStreamer, self.filepath)
        if not images:
            return 0

        prog = wx.ProgressDialog(
            "Finalize — focus & light", "Computing focus and light direction…",
            maximum=len(images), parent=self.frame,
            style=wx.PD_APP_MODAL | wx.PD_AUTO_HIDE | wx.PD_CAN_ABORT
                  | wx.PD_ELAPSED_TIME | wx.PD_REMAINING_TIME)

        def progress(i, msg):
            cont, _ = prog.Update(i, msg)
            wx.GetApp().Yield(True)
            return bool(cont)

        updated = compute_focus_light(images, progress=progress,
                                      frame_size=(self.width, self.height))
        prog.Destroy()
        return updated

   def _detectLeadingDarkFrames(self, local_dir):
        """Count the consecutive dark ('No Light') frames at the very start of the
        dataset — see mga.core.dataset_finalize.leading_dark_frames (Stage 3c)."""
        images = dataset_images(self.folderStreamer, self.filepath)
        return leading_dark_frames(images)

   def _finalizeInfoJSON(self, local_dir):
        """Read, update and write the dataset's info.json with certification info,
        the accumulated annotation-effort statistics, and the dataset-wide
        defect/severity totals. The read/modify/write lives in
        mga.core.dataset_finalize.update_info_json (Stage 3c of this file's
        refactor); this wrapper commits the session stats and shows the failure
        dialog. Returns the written info dict, or None when the write failed."""
        info_path = os.path.join(local_dir, "info.json")

        # Commit this session's active time before reading the counters.
        self._recordInteraction()

        stats = {"active_seconds": self._stat_active_seconds,
                 "clicks": self._stat_clicks,
                 "keystrokes": self._stat_keystrokes,
                 "points_added": self._stat_points_added,
                 "points_deleted": self._stat_points_deleted}

        info = update_info_json(
            info_path,
            *self._datasetDefectTotals(local_dir),
            stats,
            leading_dark_fn=lambda: self._detectLeadingDarkFrames(local_dir))
        if info is None:
            wx.MessageBox(f"Failed to write {info_path}", "Finalize", wx.OK | wx.ICON_ERROR)
        return info

   def onFinalize(self, event):
        """Finalize the dataset: backfill focus/light and inter-frame tracking,
        then write/augment info.json with certification info, effort statistics
        and the dataset-wide defect/severity totals."""
        local_dir = getattr(self.folderStreamer, "local_dir", None)
        if not local_dir or not os.path.isdir(local_dir):
            wx.MessageBox("No local dataset directory is loaded — cannot finalize.",
                          "Finalize", wx.OK | wx.ICON_ERROR)
            return

        # If focus/light were not computed live (checkbox off), backfill them for every frame.
        if not self.calcFocusLightCheckbox.GetValue():
            self._batchComputeFocusLight(local_dir)

        # Backfill inter-frame tracking for frames the Track button never visited,
        # then reconcile everything with the least-squares pass.
        res = self._fillTracking()
        if res:
            print("Finalize tracking: %u measured, %u already tracked, %u failed, "
                  "%u least-squares positions" % res[:4])

        info = self._finalizeInfoJSON(local_dir)
        if info is None:
            return

        # Reset so a second Finalize this session doesn't double-count the same effort.
        self._resetSessionStats()
        self.populateMetaData(os.path.join(local_dir, "info.json"))
        wx.MessageBox(
            f"Finalized {os.path.basename(local_dir)}.\n\n"
            f"Total defects: {info['total_defects']}\n"
            f"Defect types: {info['defect_counts']}\n"
            f"Severities: {info['severity_counts']}\n"
            f"(info.json updated with effort statistics.)\n\n"
            f"The Upload Annotations dialog will open next (Cancel there to skip).",
            "Finalize", wx.OK | wx.ICON_INFORMATION)

        # A finalized dataset is ready for the server: kick off File->Upload Annotations
        # (the dialog still lets the user cancel).
        self.onUploadAnnotations(None)

   def populateMetaData(self,path):
         self.metadata = None
         if (checkIfFileExists(path)):
              try:
                   with open(path) as json_data:
                        self.metadata = json.load(json_data)
              except Exception as e:
                   # A hand-edited info.json (e.g. a stray trailing comma) must not crash startup.
                   print("Warning: could not parse metadata from", path, ":", e)
                   self.datasetList.Set([f"info.json unreadable: {e}"])
                   return self.metadata

              metadata = list()
              for k in self.metadata.keys():
                  metadata.append("%s: %s"%(k,self.metadata[k]))

              print("Dataset metadata is : ",metadata)
              self.datasetList.Set(metadata)
         else:
              print("Failed opening meta data from ",path)

         return self.metadata


   def onNewInputPath(self, newPath):
    print("\n\n\n\nNew Input Path Received : ", newPath)
    self.filepath = newPath
    if self.filepath != "":
        self.filePathIsDirectory = checkIfPathIsDirectory(self.filepath)
        if self.filePathIsDirectory:
            self.folderStreamer.loadNewDataset(self.filepath)  # FolderStreamer
            self.openDataset(
                base_dir=self.filepath,
                streamer=self.folderStreamer,
                is_directory=True
            )
        else:
            self.onProcessNewImageSample(self.filepath)

   def loadControlsCSV(self, path):
    """Load control/sensor CSV file."""
    try:
        with open(path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            self.controlsData = list(reader)
        self.csvInfo.SetLabel(f"Loaded {len(self.controlsData)} entries from {path}")
    except Exception as e:
        self.csvInfo.SetLabel(f"Failed to load CSV ({path}): {e}")
        self.controlsData = []

   def onPhotoTxtEnter(self, event):
        self.onNewInputPath(self.photoTxt.GetValue())

   def onExit(self, event):
        sys.exit(0)

   def onProcessorComboBoxSelect(self, event):
        print("Combo box select changed")
        self.onRedrawData(event)

   def onDefectComboBoxSelect(self, event):
      selected_option = self.defectComboBox.GetValue()
      self.severityComboBox.SetValue("") #<- Cause Severity to be erased to make sure user picks it correctly 
      if selected_option == "Add Custom Option":
        # Handle custom option logic here
        custom_option = wx.GetTextFromUser("Enter custom option:")
        if custom_option:
            self.defectComboBox.Append(custom_option)
            self.defectComboBox.SetValue(custom_option)
        else:
            # Handle case where user cancels the input
            pass
   def onOpenDirectory(self, event):
        dialog = wx.DirDialog(None, "Choose a directory:", style=wx.DD_DEFAULT_STYLE | wx.DD_NEW_DIR_BUTTON)

        if dialog.ShowModal() == wx.ID_OK:
            directory_path = dialog.GetPath()
            self.photoTxt.SetValue(directory_path)
            self._loadSensorPlotsNewDataset(directory = "%s/tactile/" %  self.folderStreamer.local_dir)
            self.onNewInputPath(directory_path)

        dialog.Destroy()

   def onOpenNetwork(self, event):
        from mga.core.dataset_selector import DatasetSelector
        dlg = DatasetSelector(local_base_path=self.local_base_path)
        if dlg.ShowModal() == wx.ID_OK:
            selectedDirectory = self.local_base_path + "/" + dlg.selectedDirectory 
            print("Selected Dataset:",  dlg.selectedDataset)
            print("Caching Directory:", dlg.selectedDirectory)
            # You can pass this to your HTTPFolderStreamer
            #self.onNewInputPath(dlg.selectedDataset)
            from mga.core.http_stream import HTTPFolderStreamer
            self.folderStreamer = HTTPFolderStreamer(provider=dlg.selectedProvider, dataset=dlg.selectedDataset, local_dir=selectedDirectory, retrieve_zip=dlg.replaceAnnotations)

            # Must be set BEFORE openDataset: the classifier trigger checks
            # photoTxt != "default" while processing the FIRST frame
            self.photoTxt.SetValue(dlg.selectedDirectory)

            self.openDataset(
                             base_dir=selectedDirectory,   # cache dir where info.json/controller.csv live
                             streamer=self.folderStreamer,
                             is_directory=True
                            )
        dlg.Destroy()



   def onBrowse(self, event):
        wildcard = "JPEG files (*.jpg)|*.jpg"
        dialog   = wx.FileDialog(None, "Choose a file",wildcard=wildcard,style=wx.FD_OPEN)
        if dialog.ShowModal() == wx.ID_OK:
            self.photoTxt.SetValue(dialog.GetPath())
        dialog.Destroy()
        self.onNewInputPath(self.photoTxt.GetValue())

   def rescaleAnything(self,width,height):
        W = width
        H = height
        NewW  = self.PhotoMaxSizeWidth
        NewH  = self.PhotoMaxSizeWidth * H / W
        #print("Width based calculation ",NewW,"x",NewH)
        heightNewH = self.PhotoMaxSizeHeight
        heightNewW = self.PhotoMaxSizeHeight * W / H
        #print("Height based calculation ",heightNewW,"x",heightNewH)
        if (heightNewW<=self.PhotoMaxSizeWidth) and (heightNewH<=self.PhotoMaxSizeHeight):
            NewW = heightNewW
            NewH = heightNewH
        #print("Rescaled ",W,"x",H," to ",NewW,"x",NewH)
        return NewW,NewH

   def rescaleCVMAT(self,img):
        NewW,NewH = self.rescaleAnything(img.shape[1],img.shape[0])
        return cv2.resize(img, dsize=(int(NewW),int(NewH)), interpolation=cv2.INTER_CUBIC)
 

   def _annotate_bitmap_with_hz(self, base_bmp: wx.Bitmap) -> wx.Bitmap:
    """Return a NEW bitmap with the classifier's rate drawn top-right (does not
    modify base_bmp -- it is the cached base from _baseBitmapsForView()).

    The left panel IS the classifier visualization, so its refresh rate belongs
    on it. Returns base_bmp untouched when the classifier is off or has not
    timed a forward() yet."""
    if not (useClassifier and not self.classifierDisabledCheckbox.GetValue()):
        return base_bmp
    twoStage   = getattr(self, "classifierTwoStage", None)
    classifier = (getattr(self, "EnsembleClassifierPnm", None)
                  if (twoStage is not None and twoStage.GetValue())
                  else getattr(self, "ClassifierPnm", None))
    hz = getattr(classifier, "hz", 0.0) or 0.0
    if hz <= 0.0:
        return base_bmp

    temp_bmp = wx.Bitmap(wx.Image(base_bmp.ConvertToImage()))
    dc = wx.MemoryDC()
    dc.SelectObject(temp_bmp)
    dc.SetFont(wx.Font(11, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))

    text   = "%0.2f Hz" % hz
    tw, th = dc.GetTextExtent(text)
    x      = temp_bmp.GetWidth() - tw - 10
    # dark plate first: the visualization underneath can be any colour
    dc.SetPen(wx.Pen(wx.Colour(0, 0, 0)))
    dc.SetBrush(wx.Brush(wx.Colour(0, 0, 0)))
    dc.DrawRectangle(x - 5, 3, tw + 10, th + 4)
    dc.SetTextForeground(wx.Colour(0, 255, 255))
    dc.DrawText(text, x, 5)

    dc.SelectObject(wx.NullBitmap)
    return temp_bmp

   def _annotate_bitmap_with_points(self, base_bmp: wx.Bitmap, ratioX: float, ratioY: float,
                                     checkmarks: bool = False) -> wx.Bitmap:
    """Return a NEW bitmap with annotations drawn on it (does not modify base_bmp).

    checkmarks=True  → left image: defects shown as bold green ✓
    checkmarks=False → right image: defects shown as coloured circles (default)
    """
    img_copy = wx.Image(base_bmp.ConvertToImage())
    temp_bmp = wx.Bitmap(img_copy)

    dc = wx.MemoryDC()
    dc.SelectObject(temp_bmp)
    dc.SetBrush(wx.TRANSPARENT_BRUSH)

    expectedTileSize = 40
    r = expectedTileSize // 2

    selected_index = self.pointList.GetSelection()

    for pointID in range(len(self.points_of_interest)):
        x = self.points_of_interest[pointID][0]
        y = self.points_of_interest[pointID][1]
        pClass = self.points_classes[pointID]
        pSever = self.points_severities[pointID]

        cx = int(x / ratioX)
        cy = int(y / ratioY)

        if pClass == "RLClean":
            dc.SetPen(wx.Pen(wx.RED, 1))
            dc.SetTextForeground(wx.RED)
            dc.SetFont(wx.Font(10, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
            tw, th = dc.GetTextExtent("?")
            dc.DrawText("?", cx - tw // 2, cy - th // 2)
        elif pClass in ("Suspicious", "Clean"):
            dc.SetPen(wx.Pen(wx.GREEN, 2))
            dc.DrawCircle(cx, cy, r)
        elif checkmarks:
            # Left image: draw a bold green checkmark for all defect classes
            dc.SetTextForeground(wx.GREEN)
            dc.SetFont(wx.Font(14, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
            tw, th = dc.GetTextExtent("\u2713")
            dc.DrawText("\u2713", cx - tw // 2, cy - th // 2)
        else:
            if "Class A" in pSever:
                dc.SetPen(wx.Pen(wx.YELLOW, 2))
            elif "Class B" in pSever:
                dc.SetPen(wx.Pen(wx.NamedColour("orange"), 2))
            elif "Class C" in pSever:
                dc.SetPen(wx.Pen(wx.BLACK, 2))
            elif "AI" in pSever:
                dc.SetPen(wx.Pen(wx.WHITE, 2))
            else:
                print("Weird severity encountered (", pSever, ")")
                dc.SetPen(wx.Pen(wx.BLUE, 2))

            dc.DrawCircle(cx, cy, r + 2)

            dc.SetPen(wx.Pen(wx.RED, 2))
            dc.DrawCircle(cx, cy, r)
            dc.DrawCircle(cx, cy, r + 4)

        # Highlight the point selected in the list with a larger dotted green circle.
        if pointID == selected_index:
            dc.SetPen(wx.Pen(wx.GREEN, 3, wx.PENSTYLE_DOT))
            dc.DrawCircle(cx, cy, r + 10)

        # Badge machine-generated points with a small cyan "A" so reviewers can spot them.
        src = self.points_sources[pointID] if pointID < len(self.points_sources) else "manual"
        if src == "auto":
            dc.SetTextForeground(wx.Colour(0, 255, 255))
            dc.SetFont(wx.Font(9, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
            dc.DrawText("A", cx + r + 2, cy - r - 2)

    dc.SelectObject(wx.NullBitmap)
    return temp_bmp

   def _baseBitmapsForView(self):
    """Return (left_bmp, right_bmp, left_ok): the annotation-free base bitmaps for the
    current frame. Cached by the identity of the processed arrays so repeated onView()
    calls (clicks, clear, remove) skip the resize + buffer rebuild. The cache holds
    references to the source arrays, so a new render (new array object) misses and
    rebuilds, while annotation-only edits (same arrays) hit."""
    img = self.leftViewImage
    fg  = self.rightViewImage
    c = self._base_cache
    if c is not None and c[0] is img and c[1] is fg:
        return c[2], c[3], c[4]

    right_img = self.rescaleCVMAT(fg)
    right_bmp = wx.Bitmap.FromBuffer(right_img.shape[1], right_img.shape[0], right_img)

    if img is not None:
        left_img = self.rescaleCVMAT(img)
        left_bmp = wx.Bitmap.FromBuffer(left_img.shape[1], left_img.shape[0], left_img)
        left_ok  = left_bmp.IsOk()
    else:
        left_bmp = self.imageCtrl.GetBitmap()
        left_ok  = bool(left_bmp and left_bmp.IsOk())

    self._base_cache = (img, fg, left_bmp, right_bmp, left_ok)
    return left_bmp, right_bmp, left_ok

   def onView(self):
    # Annotation-free base bitmaps (cached). Annotations are drawn on fresh copies below,
    # so the cached bases never accumulate markers across repeated onView() calls.
    left_bmp, right_bmp, left_ok = self._baseBitmapsForView()
    rw, rh = right_bmp.GetWidth(), right_bmp.GetHeight()
    if left_ok:
        lw, lh = left_bmp.GetWidth(), left_bmp.GetHeight()

    # If no points, refresh both panels with their clean base images
    if len(self.points_of_interest) == 0:
        if self.DRAW_TARGET & self.DRAW_TARGET_LEFT and left_ok:
            self.imageCtrl.SetBitmap(self._annotate_bitmap_with_hz(left_bmp))
        if self.DRAW_TARGET & self.DRAW_TARGET_RIGHT:
            self.secondaryImageCtrl.SetBitmap(right_bmp)
        else:
            self.secondaryImageCtrl.SetBitmap(right_bmp)
        self._drawMeasureOverlay()
        self.panel.Refresh()
        return

    # ---- Annotate per-target using per-target ratios ----
    # IMPORTANT: ratios must match the bitmap you're drawing on
    # Full dims are in: self.viewedImageFullWidth / self.viewedImageFullHeight
    # (set in onProcessNewImageSample)
    if self.DRAW_TARGET & self.DRAW_TARGET_LEFT and left_ok:
        left_ratioX = self.viewedImageFullWidth / lw
        left_ratioY = self.viewedImageFullHeight / lh
        left_overlay = self._annotate_bitmap_with_points(left_bmp, left_ratioX, left_ratioY, checkmarks=True)
        self.imageCtrl.SetBitmap(self._annotate_bitmap_with_hz(left_overlay))

    if self.DRAW_TARGET & self.DRAW_TARGET_RIGHT:
        right_ratioX = self.viewedImageFullWidth / rw
        right_ratioY = self.viewedImageFullHeight / rh
        right_overlay = self._annotate_bitmap_with_points(right_bmp, right_ratioX, right_ratioY, checkmarks=False)
        self.secondaryImageCtrl.SetBitmap(right_overlay)
    else:
        # if not drawing on right, still show the right base image
        self.secondaryImageCtrl.SetBitmap(right_bmp)

    self._drawMeasureOverlay()
    self.panel.Refresh()


   def onSelectPoint(self, event):
        # Redraw so the selected point gets its dotted green highlight circle.
        self.onView()
 
   def updateMinMaxSlider(self):
    stream_cur = self.folderStreamer.current()
    ui_cur     = self._ui_from_stream(stream_cur)
    ui_cur     = max(0, min(ui_cur, self._ui_max()))

    ui_max = self._ui_max()
    percent = 0.0 if ui_max == 0 else 100.0 * (ui_cur / ui_max)

    self.scrollBar.SetValue(ui_cur)
    self.scrollBar.SetMax(ui_max)

    # Show absolute frame too (useful!)
    abs_frame = self._stream_from_ui(ui_cur)
    label = "Sample %u/%u (abs %u) - %0.2f%%" % (ui_cur, ui_max, abs_frame, percent)
    if self.tenengrad_focus_measure != 0.0:
        label += "  - Focus %0.2f" % self.tenengrad_focus_measure
    self.instructLbl.SetLabel(label)


   def onScroll(self, event):
    ui_idx = self.scrollBar.GetValue()
    print("Scroll Position:", ui_idx, "/", self.scrollBar.GetMax())
    self.gotoFrameUI(ui_idx)




   def openJumpToFrameDialog(self):
    dlg = wx.TextEntryDialog(
                             self.frame,
                             message="Enter frame number (Cur %u Max %u):" %(self.scrollBar.GetValue(),self.scrollBar.GetMax()),
                             caption="Jump to Frame"
                           )
    if dlg.ShowModal() == wx.ID_OK:
        value = dlg.GetValue()
        try:
            frame = int(value)

            # Clamp within scrollbar limits
            frame = max(0, min(frame, self.scrollBar.GetMax()))

            # Update scrollbar
            self.scrollBar.SetValue(frame)

            # Trigger your normal scroll logic
            self.onScroll(None)

        except ValueError:
            wx.MessageBox("Please enter a valid number.", "Error", wx.OK | wx.ICON_ERROR)
    dlg.Destroy()


   def on_brightness_slider(self, event):
        self.brightness_offset = self.brightnessSlider.GetValue()
        self.onRedrawData(event)

   def on_contrast_slider(self, event):
        self.contrast_offset = self.contrastSlider.GetValue()
        self.onRedrawData(event)

   def _computeViewCeiling(self):
        """Max view scale that fits the CURRENT client area. Horizontal reserve
        is the tab pane + margins; vertical reserve is MEASURED from the fixed-
        height widgets above/below the image (instruction label + under-image
        control strip + borders) — never subtract the image itself, that is
        circular and freezes the ceiling at the current size."""
        cw, ch = self.frame.GetClientSize()
        try:
            top_h = self.instructLbl.GetSize().height
            bot_h = self.underImage.GetSize().height
            chrome_h = top_h + bot_h + 70   # StaticLine + ~6 sizer borders of 5px + safety
        except Exception:
            chrome_h = 130
        pane_w = 290   # tab pane (250) + inter-column margins
        avail_w = max(300, (cw - pane_w) // 2)
        avail_h = max(260, ch - chrome_h)
        smax = min(avail_w / 1224.0, avail_h / 1024.0, 1.0)
        print(f"[ViewCeiling] client {cw}x{ch} chrome_h {chrome_h} -> "
              f"avail {avail_w}x{avail_h} -> {int(100*smax)}%")
        return smax

   def _applyViewCeiling(self, set_to_max=False):
        smax = self._computeViewCeiling()
        self._viewScaleMax = smax
        vmax = max(41, int(100 * smax))
        if vmax != self.viewSizeSlider.GetMax():
            self.viewSizeSlider.SetMax(vmax)
            self.viewSizeSlider.SetToolTip(f"Size of the two image panels as % of native "
                                           f"1224x1024 (max {vmax}% fits the current window)")
        if set_to_max:
            self.viewSizeSlider.SetValue(vmax)
        return smax

   def _calibrateWindowToUsableDesktop(self):
        """One-shot post-startup calibration. The WM (LxQt) may clamp
        programmatic resizes to the current monitor, so: request the usable
        budget once, then size the image panels from the frame size ACTUALLY
        granted, reserving a fixed width for the right tab pane (its
        GetBestSize lies upward because WrapSizer rows report unwrapped)."""
        try:
            usable_w, usable_h = self._usableDesktop
            PANE_W = 250   # right tab pane reservation (controls fit; images get the rest)
            self.rightBook.SetMinSize((PANE_W, -1))
            self.frame.SetSize(wx.Size(usable_w, usable_h))

            def _finish():
                print(f"[Calibrate] frame now {tuple(self.frame.GetSize())} "
                      f"(asked {usable_w}x{usable_h})")
                smax = self._applyViewCeiling(set_to_max=True)
                newW, newH = int(1224 * smax), int(1024 * smax)
                if abs(newW - self.PhotoMaxSizeWidth) > 4 or abs(newH - self.PhotoMaxSizeHeight) > 4:
                    self.PhotoMaxSizeWidth, self.PhotoMaxSizeHeight = newW, newH
                    with self._prefetch_lock:
                        self._prefetch = None
                    self.onRedrawData(None)
                # Layout INSIDE the granted frame; Fit would re-fight the WM
                self.panel.Layout()
            wx.CallLater(600, _finish)
        except Exception as e:
            print(f"[Calibrate] failed (leaving startup sizing as-is): {e}")

   def on_view_size_slider(self, event):
        """Resize both image panels; 100% = native 1224x1024 demosaic resolution."""
        self._applyViewCeiling()   # keep the max honest w.r.t. the current window
        scale = min(self.viewSizeSlider.GetValue() / 100.0, self._viewScaleMax)
        self.PhotoMaxSizeWidth  = int(1224 * scale)
        self.PhotoMaxSizeHeight = int(1024 * scale)
        # prefetched renders were produced at the old size — drop them
        with self._prefetch_lock:
            self._prefetch = None
        print(f"[ViewSize] slider={self.viewSizeSlider.GetValue()}% -> "
              f"PhotoMaxSize {self.PhotoMaxSizeWidth}x{self.PhotoMaxSizeHeight}")
        self.onRedrawData(event)
        self.panel.Layout()
        def _report():
            iw, ih = self.imageCtrl.GetSize()
            bmp = self.imageCtrl.GetBitmap()
            bw, bh = (bmp.GetWidth(), bmp.GetHeight()) if bmp.IsOk() else (-1, -1)
            fw, fh = self.frame.GetSize()
            dw, dh = wx.DisplaySize()
            print(f"[ViewSize] after Fit: frame {fw}x{fh} (display {dw}x{dh}) | "
                  f"left imageCtrl widget {iw}x{ih} | bitmap {bw}x{bh}"
                  + ("  <-- WIDGET SMALLER THAN BITMAP: clicks/legend will be wrong!"
                     if (iw < bw or ih < bh) else ""))
        wx.CallAfter(_report)

   def onRedrawData(self, event):
        print("Asked to redraw data")
        # Re-render the CURRENT frame once (used by brightness/contrast/processor changes).
        # gotoFrameUI(current) saves in-memory edits then re-processes — half the work of the
        # old onNext()+onPrevious() double-reload, with the same data-preservation behaviour.
        self.gotoFrameUI(self.scrollBar.GetValue())

   def updateControlsTab(self, data_row,sample_number = 0):
    """
    Update the Controls tab UI fields with a row dict from the CSV.
    Example row:
      {"timestamp": 308239, "dev_timestamp": 4, "Button1": 0, "Distance1": "F", ...}
    """
    print("Controller : ",data_row)
    for key, ctrl in self.controlsFields.items():
        if key in data_row:
            value = data_row[key]
            if isinstance(value, float):
                ctrl.SetValue("%0.1f" % value)
            else:
                ctrl.SetValue(str(value))

    self._loadSensorPlotsNewSample(sample_number=sample_number)

   def _slider_max(self):
    try:
        return int(self.scrollBar.GetMax())
    except Exception:
        return 0

   def _stopPlayback(self):
    if getattr(self, "playTimer", None) is not None and self.playTimer.IsRunning():
        self.playTimer.Stop()
    self.isPlaying = False
    if getattr(self, "playBtn", None) is not None:
        self.playBtn.SetLabel("Play")

   def onTogglePlay(self, event):
    print("Play pressed. ui=", self.scrollBar.GetValue(), " max=", self.scrollBar.GetMax())
    if self.isPlaying:
        self._stopPlayback()
        return

    self.isPlaying = True
    self.playBtn.SetLabel("Pause")
    self.playTimer.Start(self.playIntervalMs)

   def onPlayTimer(self, event):
    # Advance one frame; stop at end (do NOT wrap).
    try:
        ui = int(self.scrollBar.GetValue())
    except Exception:
        self._stopPlayback()
        return

    ui_max = self._slider_max()
    if ui >= ui_max:
        self._stopPlayback()
        return

    # IMPORTANT: schedule on UI loop and yield paint events
    self.gotoFrameUI(ui + 1)

    # Force redraw so you actually see frames change
    try:
        self.panel.Refresh(False)
        self.panel.Update()
        wx.YieldIfNeeded()
    except Exception:
        pass

   def onNext(self, event):
    if getattr(self, "isPlaying", False):
        self._stopPlayback()
    ui = self.scrollBar.GetValue()
    ui = 0 if ui >= self._slider_max() else (ui + 1)
    self.gotoFrameUI(ui)

   def onPrevious(self, event):
    if getattr(self, "isPlaying", False):
        self._stopPlayback()
    ui = self.scrollBar.GetValue()
    ui = self._slider_max() if ui <= 0 else (ui - 1)
    self.gotoFrameUI(ui)

   def onRemovePoint(self, event):
        selected_index = self.pointList.GetSelection()
        if selected_index != -1:
            remove_point(self.points_of_interest, self.points_classes,
                         self.points_severities, self.points_sources,
                         selected_index)
            self._stat_points_deleted += 1
            self.updatePointList()
            self.onView()   # fast redraw so the removed marker disappears immediately

   def onCopyPreviousPoints(self, event):
        """Copy points/classes/severities from the previous frame's JSON (if it exists) into the current frame."""
        try:
            cur_idx = self.folderStreamer.current()
        except Exception:
            wx.MessageBox("No dataset loaded.", "Copy Previous Points", wx.OK | wx.ICON_INFORMATION)
            return

        prev_idx = cur_idx - 1
        if prev_idx < 0:
            wx.MessageBox("You are already on the first frame; there is no previous frame to copy from.",
                         "Copy Previous Points", wx.OK | wx.ICON_INFORMATION)
            return

        # Temporarily jump to previous to compute JSON path, then restore.
        try:
            self.folderStreamer.select(prev_idx)
            prev_img = self.folderStreamer.getImage()
            prev_json = resolve_annotation_json_path(prev_img, prefer_existing=True)
        finally:
            self.folderStreamer.select(cur_idx)

        if not checkIfFileExists(prev_json):
            wx.MessageBox("Previous frame has no saved JSON annotations to copy from.",
                         "Copy Previous Points", wx.OK | wx.ICON_INFORMATION)
            return

        try:
            with open(prev_json, 'r') as f:
                data = json.load(f)
        except Exception as e:
            wx.MessageBox(f"Failed to read previous JSON: {e}",
                         "Copy Previous Points", wx.OK | wx.ICON_ERROR)
            return

        state = annotation_from_dict(data)

        # Normalize lengths
        pts, cls, sev, src = normalize_parallel(
            state["points"], state["classes"], state["severities"], state["sources"],
            options[0], severities[0], "manual")

        self.points_of_interest = pts
        self.points_classes = cls
        self.points_severities = sev
        self.points_sources = src
        self._stat_points_added += len(pts)
        self.updatePointList()
        self.onNext(event)

   def onClearAllAnnotations(self, event):
        """Wipe every point/region annotation on the current frame (cleanup shortcut: 'D')."""
        self._stat_points_deleted += len(self.points_of_interest)
        self.cleanThisFrameMetaData()
        self.onSave(event)   # persist the now-empty annotations
        self.onView()        # fast redraw on the cached base — no image re-decode/re-render

   def formatPoints(self):
        result = list()
        if len(self.points_of_interest) != len(self.points_severities):
            print("Points without severities, this should never happen!")
            print(len(self.points_of_interest)," vs ",len(self.points_severities))
        elif len(self.points_of_interest) != len(self.points_classes):
            print("Points without classes, this should never happen!")
            print(len(self.points_of_interest)," vs ",len(self.points_classes))
        else:
            for i in range(0,len(self.points_of_interest)):            
               result.append("%u,%u - %s / %s" % (self.points_of_interest[i][0],
                                                  self.points_of_interest[i][1],
                                                  self.points_classes[i],
                                                  self.points_severities[i])    )
        return result

   def updatePointList(self):
        self.pointList.Set(self.formatPoints())

   def formatRegions(self):
        result = list()
        for i in range(0,len(self.regions_of_interest)):            
               result.append("%u,%u" % (self.regions_of_interest[i][0],self.regions_of_interest[i][1]))
        return result

   def updateRegionList(self):
        self.regionList.Set(self.formatRegions())

   def onLeftDown(self, event):
       ex, ey = event.GetPosition()
       ctrl = event.GetEventObject()
       cw, ch = ctrl.GetSize()
       bmp = ctrl.GetBitmap() if hasattr(ctrl, "GetBitmap") else None
       bw, bh = (bmp.GetWidth(), bmp.GetHeight()) if (bmp and bmp.IsOk()) else (-1, -1)
       print(f"[Click] pos=({ex},{ey}) widget={cw}x{ch} bitmap={bw}x{bh} "
             f"viewed(view {self.viewedImageViewWidth}x{self.viewedImageViewHeight} / "
             f"full {self.viewedImageFullWidth}x{self.viewedImageFullHeight}) "
             f"ratio=({self.clickRatioX:.3f},{self.clickRatioY:.3f}) -> "
             f"full=({ex*self.clickRatioX:.0f},{ey*self.clickRatioY:.0f})"
             + ("  <-- widget/bitmap size MISMATCH" if (bw > 0 and (cw != bw or ch != bh)) else ""))
       if self.measureMode:
           mx, my = event.GetPosition()
           fullPt = (mx * self.clickRatioX, my * self.clickRatioY)
           if len(self.measurePoints) >= 2:   # start a fresh measurement
               self.measurePoints = []
           self.measurePoints.append(fullPt)
           if len(self.measurePoints) == 2:
               self._computeMeasurement()
           else:
               self.measureResult.SetLabel("First point set — click the second point.")
           self.onView()
           return
       if self.photoTxt.GetValue() != "default": #<- Don't trigger in logo on boot
        self._stat_clicks += 1
        self._stat_points_added += 1
        self._recordInteraction()
        self.x, self.y = event.GetPosition()
        add_point(self.points_of_interest, self.points_classes,
                  self.points_severities, self.points_sources,
                  self.x * self.clickRatioX, self.y * self.clickRatioY,
                  self.defectComboBox.GetValue(),
                  self.severityComboBox.GetValue(),
                  "manual")

        self.updatePointList()

        if (self.incrementFrameAfterAnAdditionCheckbox.GetValue()):
               print("Auto Incrementing")
               self.onNext(event)
        else:
               self.onSave(event)   # persist the new point
               self.onView()        # fast redraw on the cached base — no image re-decode/re-render

   def onToggleMeasure(self, event):
       self.measureMode   = not self.measureMode
       self.measurePoints = []
       if self.measureMode:
           self.measureBtn.SetLabel("Measuring… (click 2 points)")
           self.measureResult.SetLabel("Click the first point on the image.")
       else:
           self.measureBtn.SetLabel("Measure (2 clicks)")
       self.onView()

   def _currentTOFHeightMm(self):
       """Average of the valid TOF Distance1/2/3 sensor readings, or None."""
       vals = []
       for k in ("Distance1", "Distance2", "Distance3"):
           ctrl = self.controlsFields.get(k)
           if ctrl is None:
               continue
           try:
               vals.append(float(ctrl.GetValue()))
           except (ValueError, TypeError):
               pass  # non-numeric readings such as "F" (out of range)
       if not vals:
           return None
       return sum(vals) / len(vals)

   def _computeMeasurement(self):
       """Distance between the two measure points, reported in raw debayered pixels and mm.
       The pinhole px-per-mm math (with TOF height correction) lives in
       mga.core.frame_processing.pixels_to_mm (Stage 3d of this file's refactor);
       this keeps the widget reads and the labels."""
       try:
           px_per_mm_ref = float(self.calibPxPerMm.GetValue())
       except ValueError:
           px_per_mm_ref = 0.0
       try:
           ref_h = float(self.calibHeightMm.GetValue())
       except ValueError:
           ref_h = 0.0

       cur_h = self._currentTOFHeightMm()
       if cur_h is None:
           cur_h = ref_h

       debayer_dist, mm, _px_per_mm = pixels_to_mm(
           self.measurePoints[0], self.measurePoints[1],
           px_per_mm_ref, ref_h, cur_h)

       if mm is not None:
           self.measureResult.SetLabel(
               "%.1f px (debayered)  ≈  %.2f mm   @ %.1f mm height"
               % (debayer_dist, mm, cur_h))
           self.measureLabel = "%.1f px / %.2f mm" % (debayer_dist, mm)
       else:
           self.measureResult.SetLabel(
               "%.1f px (debayered)  — set Pixels per mm" % debayer_dist)
           self.measureLabel = "%.1f px" % debayer_dist
       print("Measurement:", self.measureResult.GetLabel())

   def _drawMeasureOverlay(self):
       """Draw the measurement crosses + connecting line on the current left bitmap."""
       if not self.measurePoints:
           return
       bmp = self.imageCtrl.GetBitmap()
       if not (bmp and bmp.IsOk()):
           return
       lw, lh = bmp.GetWidth(), bmp.GetHeight()
       ratioX = self.viewedImageFullWidth  / lw if lw else 1.0
       ratioY = self.viewedImageFullHeight / lh if lh else 1.0

       temp_bmp = wx.Bitmap(wx.Image(bmp.ConvertToImage()))
       dc = wx.MemoryDC()
       dc.SelectObject(temp_bmp)
       dc.SetPen(wx.Pen(wx.Colour(0, 255, 255), 2))
       pts = [(int(px / ratioX), int(py / ratioY)) for px, py in self.measurePoints]
       for (x, y) in pts:
           dc.DrawLine(x - 6, y, x + 6, y)
           dc.DrawLine(x, y - 6, x, y + 6)
       if len(pts) == 2:
           dc.DrawLine(pts[0][0], pts[0][1], pts[1][0], pts[1][1])
           label = getattr(self, "measureLabel", "")
           if label:
               mx = (pts[0][0] + pts[1][0]) // 2
               my = (pts[0][1] + pts[1][1]) // 2
               dc.SetTextForeground(wx.Colour(0, 255, 255))
               dc.SetFont(wx.Font(10, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
               dc.DrawText(label, mx + 8, my + 8)
       dc.SelectObject(wx.NullBitmap)
       self.imageCtrl.SetBitmap(temp_bmp)

   def onRightDown(self, event):
      # Right-click removes the annotation point nearest to the cursor.
      if self.photoTxt.GetValue() == "default":  # ignore in the boot logo
          return
      if not self.points_of_interest:
          return
      self._stat_clicks += 1
      self._recordInteraction()
      mx, my = event.GetPosition()
      fx, fy = mx * self.clickRatioX, my * self.clickRatioY  # full-res coords, as points are stored
      idx, d2 = nearest_point_sq(fx, fy, self.points_of_interest)
      # guard against deleting a far-away point on an empty-space click (~1.5 tiles)
      if d2 > (144 * self.clickRatioX) ** 2:
          print("Right-click: no point near (%d,%d)" % (int(fx), int(fy)))
          return
      print("Right-click removing point %d at (%d,%d)" %
            (idx, int(self.points_of_interest[idx][0]), int(self.points_of_interest[idx][1])))
      remove_point(self.points_of_interest, self.points_classes,
                   self.points_severities, self.points_sources, idx)
      self._stat_points_deleted += 1
      self.updatePointList()
      self.onView()

   def onMiddleDown(self, event):
        self.onNext(event)

   def onMouseWheel(self, event):
        """Handle mouse wheel events."""
        rotation = event.GetWheelRotation()  # Positive for up, negative for down
        if rotation > 0:
            print("Mouse wheel moved up")
            self.onPrevious(event)
        else:
            print("Mouse wheel moved down")
            self.onNext(event)

   def onKeyPress(self, event):
        keycode = event.GetKeyCode()
        self._stat_keystrokes += 1
        self._recordInteraction()
        if keycode == wx.WXK_LEFT:
            self.onPrevious(event)
        elif keycode == wx.WXK_RIGHT:
            self.onNext(event)
        elif keycode == wx.WXK_ESCAPE:
            self.onExit(event)
        elif keycode == ord('J') or keycode == ord('j'):
            self.openJumpToFrameDialog()
        elif keycode == ord('R') or keycode == ord('r'):
            focus = wx.Window.FindFocus()
            if isinstance(focus, (wx.TextCtrl, wx.ComboBox)):
                event.Skip()  # user is typing 'r' into a field, don't refresh
            else:
                self.onRescan(event)
        elif keycode == ord('D') or keycode == ord('d'):
            focus = wx.Window.FindFocus()
            if isinstance(focus, (wx.TextCtrl, wx.ComboBox)):
                event.Skip()  # user is typing 'd' into a field, don't wipe
            else:
                self.onClearAllAnnotations(event)
        elif keycode == ord('T') or keycode == ord('t'):
            focus = wx.Window.FindFocus()
            if isinstance(focus, (wx.TextCtrl, wx.ComboBox)):
                event.Skip()  # user is typing 't' into a field, don't track
            elif event.ShiftDown():
                self.onTrackBack(event)   # Shift+T tracks backward
            else:
                self.onTrack(event)
        elif keycode == wx.WXK_TAB:
            if self.magnifier and self.magnifier.IsShown():
                self.magnifier_source = "right" if self.magnifier_source == "left" else "left"
                self._updateMagnifierImage()
                return
        else:
            event.Skip()

   def onUploadAnnotations(self, event):
        """Upload this dataset's annotation JSONs to the server (zip build and
        dialog moved to mga.core.upload_annotations.upload_dataset_annotations)."""
        from mga.core.upload_annotations import upload_dataset_annotations
        upload_dataset_annotations(self.frame, self.local_base_path,
                                  self.folderStreamer.local_dir)

   def onRunBatch(self, event):
        dlg = BatchProcessDialog(self.frame, self.folderStreamer)
        dlg.ShowModal()
        dlg.Destroy()

   def _updateMagnifierImage(self):
    if not self.magnifier:
        return

    if self.magnifier_source == "left":
        bmp = self.imageCtrl.GetBitmap()
    else:
        bmp = self.secondaryImageCtrl.GetBitmap()

    self.magnifier.updateImage(bmp)

   def onOpenMagnifier(self, event):
    # If already open, just raise it
    if self.magnifier and self.magnifier.IsShown():
        self.magnifier.Raise()
        return

    self.magnifier = MagnifierFrame(self.frame)
    self.magnifier.Show()

    # Choose initial source
    src_ctrl = self.imageCtrl if self.magnifier_source == "left" else self.secondaryImageCtrl
    self._magnifier_src = src_ctrl  # track current source control

    # Set initial image (wx.Image!)
    bmp = src_ctrl.GetBitmap()
    if bmp and bmp.IsOk():
        self.magnifier.setImage(bmp.ConvertToImage())

    # IMPORTANT: bind motion so updates happen
    self.imageCtrl.Bind(wx.EVT_MOTION, self.onMouseMoveMagnifier)
    self.secondaryImageCtrl.Bind(wx.EVT_MOTION, self.onMouseMoveMagnifier)


   def onRecordDataset(self,event):
       os.system("python3 -m mga.grabber_frontend %s" % self.local_base_path) #<- Lazy

   def onCreateDataset(self,event):
       os.system("python3 -m mga.dataset_creator %s" % self.local_base_path) #<- Lazy

   def onTileExplorer(self,event):
       os.system("python3 -m analysis.tile_explorer %s" % self.local_base_path) #<- Lazy

   def onStreamer(self,event):
       try:
          selectedDirectory = self.folderStreamer.local_dir
          print("Streamer set directory : ",selectedDirectory)
          os.system("python3 -m mga.stream_dataset %s" % selectedDirectory) #<- Lazy
       except AttributeError:
          wx.MessageBox("Please open a network database before attempting to stream something", "Error", wx.OK | wx.ICON_ERROR)


   def onBenchmarkGeneral(self,event,alterStep=False):
        dlg = wx.MessageDialog(
            self.frame,
            f"Make sure you have a correct NN configuration\n\n"
            "The benchmark will take some time and the UI will become unresponsive",
            "Are you sure you want to continue?",
            wx.YES_NO | wx.ICON_QUESTION
        )
        res = dlg.ShowModal()
        dlg.Destroy()

        if res == wx.ID_YES:
           print("Doing Perfomance Benchmark")
           self.scrollBar.SetValue(0) #Go To Start
           self.onScroll(None)
           totalFrames = self.scrollBar.GetMax()
           stepSizeMinimumBenchmark = 14
           stepSizeMaximumBenchmark = 32
           stepSize = stepSizeMinimumBenchmark

           self.stats.reset()
           for frameNumber in range(totalFrames):
               if (alterStep):
                 print("Perfomance Benchmark %u/%u" % (frameNumber,totalFrames))
                 stepSize = stepSize + 1 
                 if (stepSize>stepSizeMaximumBenchmark):
                      stepSize = stepSizeMinimumBenchmark
               else:
                 print("Accuracy Benchmark %u/%u" % (frameNumber,totalFrames))

               self.classifierTileSize.SetValue(stepSize)
               self.onNext(event)
               wx.Yield()
           thr = self.classifierThreshold.GetValue() / 100.0
           if alterStep:
               step_info = f"step={stepSizeMinimumBenchmark}..{stepSizeMaximumBenchmark} (cycled)"
           else:
               step_info = f"step={self.classifierTileSize.GetValue()}"
           # gate mode is part of the operating point: a threshold means a
           # different thing per mode, so benchmarks without it are uncomparable
           self.stats.run_info = (f"gate={self.classifierGateMode.GetValue()}  "
                                  f"threshold={thr:.2f}  {step_info}  "
                                  f"tile={self.ClassifierPnm.tile_size}  hit_radius={self.stats.hit_radius}  "
                                  f"erode_kernel={self.erodeKernelSize.GetValue()}  "
                                  f"min_votes={self.erodeThreshold.GetValue()}  "
                                  f"majority_vote={self.classifierMajorityVoting.GetValue()}")
           self.stats.print_stats()

        else:
           print("Doing Nothing")

   def onBenchmarkPerf(self,event):
        self.onBenchmarkGeneral(event,alterStep=True)

   def onBenchmarkAcc(self,event):
        self.onBenchmarkGeneral(event,alterStep=False)

   # ---------------------------------------------------------------------------
   def onMakeVideo(self, event):
        """Render every frame (left + right side-by-side) to JPEGs then
        encode with ffmpeg (see mga.core.video_export.export_dataset_video)."""
        from mga.core.video_export import export_dataset_video
        export_dataset_video(self)

   def onImageSnapshot(self, event):
        """Save 4 PNGs of the current frame into snapshots/: snap_<serial>_L.png / _LO.png
        (left, plain / with overlays) and _R.png / _RO.png (right), where <serial> is the
        next incremental snapshot number. Plain images come from the cached annotation-free
        bases, overlays straight from the displayed bitmaps -- all at panel resolution."""
        if not self.filepath or self.rightViewImage is None:
            wx.MessageBox("No frame loaded.", "Image Snapshot", wx.OK | wx.ICON_INFORMATION)
            return

        left_bmp, right_bmp, left_ok = self._baseBitmapsForView()
        if not left_ok:
            wx.MessageBox("Left image is not available for the current frame.",
                          "Image Snapshot", wx.OK | wx.ICON_WARNING)
            return
        if not (right_bmp and right_bmp.IsOk()):
            wx.MessageBox("Right image is not available for the current frame.",
                          "Image Snapshot", wx.OK | wx.ICON_WARNING)
            return

        os.makedirs("snapshots", exist_ok=True)
        serial = self._nextSnapshotSerial()

        saved = []
        for suffix, bmp in (("_L.png",  left_bmp),
                            ("_LO.png", self.imageCtrl.GetBitmap()),
                            ("_R.png",  right_bmp),
                            ("_RO.png", self.secondaryImageCtrl.GetBitmap())):
            if not (bmp and bmp.IsOk()):
                continue
            path = os.path.join("snapshots", "snap_%05d%s" % (serial, suffix))
            if bmp.ConvertToImage().SaveFile(path, wx.BITMAP_TYPE_PNG):
                saved.append(path)
            else:
                print("Image Snapshot: failed to write", path)

        #wx.MessageBox("Saved %d of 4 snapshots:\n%s" % (len(saved), "\n".join(saved)),
        #              "Image Snapshot", wx.OK | wx.ICON_INFORMATION)

   def _nextSnapshotSerial(self):
        """Next snapshot serial: one past the highest snap_NNNNN_*.png already in
        snapshots/ (0 when the folder holds no snapshots)."""
        serial = -1
        for name in os.listdir("snapshots"):
            parts = name.split("_")
            if name.startswith("snap_") and len(parts) > 1 and parts[1].isdigit():
                serial = max(serial, int(parts[1]))
        return serial + 1

   def onMouseMoveMagnifier(self, event):
    if not (hasattr(self, 'magnifier') and self.magnifier and self.magnifier.IsShown()):
        event.Skip()
        return

    src = event.GetEventObject()  # either imageCtrl or secondaryImageCtrl

    bmp = src.GetBitmap()
    if not (bmp and bmp.IsOk()):
        event.Skip()
        return

    # Switch magnifier source only when the mouse moves over the other image
    if getattr(self, "_magnifier_src", None) is not src:
        self._magnifier_src = src
        try:
            self.magnifier.setImage(bmp.ConvertToImage())
        except Exception:
            pass

    # Map cursor from control-client coordinates to bitmap coordinates.
    # The control may retain its original allocated size (e.g. PhotoMaxSizeWidth x
    # PhotoMaxSizeHeight) even after a smaller bitmap is set, so a direct pixel
    # pass-through would land in the wrong image position.
    x, y = event.GetX(), event.GetY()
    ctrl_w, ctrl_h = src.GetClientSize()
    bmp_w, bmp_h = bmp.GetWidth(), bmp.GetHeight()
    if ctrl_w > 0 and ctrl_h > 0:
        x = int(x * bmp_w / ctrl_w)
        y = int(y * bmp_h / ctrl_h)

    self.magnifier.updateMagnifier(x, y)

    event.Skip()


if __name__ == '__main__':
    print("Annotator App Starting..")
    app        = PhotoCtrl()
    inputIsSet = False
    if (len(sys.argv)>1):
       #print('Argument List:', str(sys.argv))
       for i in range(0, len(sys.argv)):
           if (sys.argv[i]=="--debug"):
               app.onDebug(None)
           if (sys.argv[i]=="--classifier"):
               #global useClassifier
               useClassifier = True
               app.classifierDisabledCheckbox.SetValue(False)
           if (sys.argv[i]=="--db"): 
               app.local_base_path = sys.argv[i+1] 
               print("Using ",app.local_base_path," as dataset base path")
           if (sys.argv[i]=="--from"):
               loadDataset=sys.argv[i+1]
               print("Loading from ",loadDataset," dataset")

               #Small check (this will need to  be updated if defects change)..
               for widget, value in dataset_defaults(loadDataset):
                   getattr(app, widget).SetValue(value)

               app.photoTxt.SetValue(loadDataset)
               app.onNewInputPath(loadDataset)
               inputIsSet = True
 

    if not inputIsSet:
               app.photoTxt.SetValue("default")
               app.onNewInputPath("default")


    app.MainLoop()

