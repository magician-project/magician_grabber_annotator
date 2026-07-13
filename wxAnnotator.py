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
python3 wxAnnotator.py --from /path/to/dataset/here/

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
import getpass
import glob
from datetime import datetime


"""
Configurations in one central place
"""

version         = "0.70"
useClassifier   = True #<- Master switch classifier off if you have hw/sw limitations
benchmark       = False #<- Set to True to run a forward-pass timing test on each model at startup
combineChannels = True
options         = ["Unknown", "Material Defect", "Positive Dent", "Negative Dent", "Deformation", "Seal", "Welding", "Suspicious", "Clean", "RLClean"]
severities      = ["Class A","Class B","Class C"]
directions      = ["Unknown","No Light","Bottom Left","Top Left","Top","Top Right", "Bottom Right", "Bottom"]
processors      = ["PolarizationRGB1","PolarizationRGB2","PolarizationRGB3", "Polarization_0_degree","Polarization_45_degree","Polarization_90_degree", "Polarization_135_degree", "AoLP", "DoLP", "Normals", "Intensity", "s0", "s1", "s2", "s3", "AoLP (light)", "AoLP (dark)", "DoP", "DoCP", "ToP", "CoP", "RetardationMag", "MaxMinAvgRGB", "Sobel","Visible"]


#classifier_relative_directory = "../classifier" #Old Name
classifier_online_repository         = "http://ammar.gr/magician/ckpts2/"
classifier_relative_directory = "../magician_vision_classifier"
classifier_model_path         = None
classifier_cfg_path           = None


"""
from wxAcquisition import CameraSettingsDialog
"""

# Import the wxScrollBar module
import wx.lib.newevent
import sys
import os

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

from folderStream import FolderStreamer
from classifierGrading import AnnotationCorrelationStats
from downloadAllFrames import BatchProcessDialog
from magnifier import MagnifierFrame
from modelUpdater import ModelUpdaterDialog
from rlAnnotator import RLAnnotatorDialog

# AutoAnnotator needs gradio_client (optional). Import lazily-safe so the GUI still
# launches if the dependency / servers are unavailable; onAuto reports the error.
try:
    from AutoAnnotator import AutoAnnotator, temporal_consensus
except Exception as _autoErr:
    AutoAnnotator = None
    _autoImportError = _autoErr


# Add this line at the beginning of the file to define a new event
ScrollEvent, EVT_SCROLL_EVENT = wx.lib.newevent.NewCommandEvent()

from readData import debayerPolarImage,repackPolarToMosaic,readPolarPNMToRGBALive

"""
def debayerPolarImage(image): 
 # Split the A, B, C, and D values into separate monochrome images
 polarization_90_deg   = image[0::2, 0::2]
 polarization_45_deg   = image[0::2, 1::2]
 polarization_135_deg  = image[1::2, 0::2]
 polarization_0_deg    = image[1::2, 1::2]
 return polarization_0_deg,polarization_45_deg,polarization_90_deg,polarization_135_deg      
"""
#-------------------------------------------------------------------------------
# Make Classifier completely seperatable from the rest of the codebase
#-------------------------------------------------------------------------------
if useClassifier:
  parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), classifier_relative_directory))
  sys.path.append(parent_path)
  try:
    from liveClassifierTorch import ClassifierPnm
    from EnsembleClassifier  import EnsembleClassifierPnm
  except Exception as e:
    print("Can't seem to be able to access the magician_vision_classifier, consider setting useClassifier=False in wxAnnotator.py")
    print("Classifier Path : ",parent_path)
    print("If you want the classifier but don't have it get it @ https://github.com/magician-project/magician_vision_classifier")
    print(f"Exact error was : {e}")
    sys.exit(1)
else:
  class ClassifierPnm:
    def __init__(self, model_path='foo', cfg_path='foo', tile_classes=['foo'],tile_size=64, step=16):
        print("Classifier PNM is disabled, please start with --classifier or change the useClassifier variable in wxAnnotator to use it!")
        pass
    def load_model(self):
        return None
    def model_scan(directoryPath):
        return ['Disabled']
    def reload_model(self, directoryPath, name):
        return False
    def forward(self, image, majorityVote=True):
        return None
#-------------------------------------------------------------------------------



from readData import resolve_annotation_json_path, list_image_files, checkIfFileExists, checkIfPathExists, checkIfPathIsDirectory, get_md5
from visualizeData import convertPolarCVMATToRGB, convertRGBCVMATToRGB, tenengrad_focus_measure, determine_intensity_region, detect_sobel_edges
from uploadAnnotations import UploadDialog


# The illumination cycles through the scene lights during acquisition, so nearby
# frames repeat the same lighting. The Track button records a direct transform to
# the earlier frame whose lighting fingerprint best matches the destination's —
# scanning back at most this many frames (fingerprints, not a fixed period, so
# this stays correct across framerates).
SAME_LIGHT_SEARCH_MAX = 12
# Minimum fingerprint cosine similarity to accept a frame as "same lighting"
# (same-light pairs score >0.95 on FORTH_DoorCase_weld_650, differently-lit
# neighbours <0.7).
SAME_LIGHT_MIN_SIMILARITY = 0.90


def lightingFingerprint(path, grid=4):
    """Compact lighting signature of a frame: the per-channel × grid×grid cell mean
    intensities, mean-subtracted and L2-normalized. Cosine similarity between two
    fingerprints separates the scene-light cycle far better than the coarse 6-region
    lightDirection label, which cannot when the part occupies one image corner.
    Returns None when the image cannot be read."""
    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        return None
    raw = raw.astype(np.float32)
    if raw.ndim == 2:
        raw = raw[:, :, None]
    H, W, C = raw.shape
    cells = []
    for c in range(C):
        for gy in range(grid):
            for gx in range(grid):
                cells.append(raw[gy*H//grid:(gy+1)*H//grid,
                                 gx*W//grid:(gx+1)*W//grid, c].mean())
    v = np.array(cells, np.float32)
    v -= v.mean()
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def estimateFrameAffine(prev_path, next_path, block=256, step=128, min_block_resp=0.08):
    """Estimate the transform between two consecutive frames as a similarity
    (rotation + scale + translation): the camera motion is not purely 2D, so a
    global translation drifts (~0.4 deg and ~0.3% scale per frame on
    FORTH_DoorCase_weld_650). Phase correlation is the only primitive robust to
    the frame-to-frame lighting cycle, so it is applied per block on a grid and a
    RANSAC similarity is fitted through the block motions; when fewer than 4
    inlier blocks survive, falls back to the global translation.
    Returns (M, (cdx, cdy), response, inliers) with the 2x3 matrix M and the
    displacement (cdx, cdy) of the image centre both in full-mosaic coordinates,
    response the global phase-correlation confidence."""
    shift_scale = 1.0
    imgs = []
    for path in (prev_path, next_path):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise IOError("Could not load %s" % path)
        if img.ndim == 3:
            if img.shape[2] == 4:
                shift_scale = 2.0
            img = img.mean(axis=2)
        imgs.append(img.astype(np.float32))
    ia, ib = imgs
    H, W = ia.shape
    win = cv2.createHanningWindow((W, H), cv2.CV_32F)
    (dx, dy), response = cv2.phaseCorrelate(ia * win, ib * win)

    src, dst = [], []
    bwin = cv2.createHanningWindow((block, block), cv2.CV_32F)
    for y0 in range(0, H - block + 1, step):
        for x0 in range(0, W - block + 1, step):
            blk = ia[y0:y0 + block, x0:x0 + block]
            if blk.std() < 8:
                continue  # flat/dark block carries no alignment signal
            xs, ys = int(round(x0 + dx)), int(round(y0 + dy))
            if xs < 0 or ys < 0 or xs + block > W or ys + block > H:
                continue
            (bx, by), r = cv2.phaseCorrelate(
                blk * bwin, ib[ys:ys + block, xs:xs + block] * bwin)
            if r < min_block_resp:
                continue
            c = block / 2.0
            src.append([x0 + c, y0 + c])
            dst.append([xs + c + bx, ys + c + by])

    M, inliers = None, 0
    if len(src) >= 4:
        M, inl = cv2.estimateAffinePartial2D(np.float32(src), np.float32(dst),
                                             ransacReprojThreshold=4.0)
        inliers = 0 if inl is None else int(inl.sum())
    if M is None or inliers < 4:
        M, inliers = np.float64([[1, 0, dx], [0, 1, dy]]), 0

    # To mosaic coordinates: the linear part is scale-invariant, translation scales.
    Mm = np.float64(M).copy()
    Mm[:, 2] *= shift_scale
    cx, cy = W * shift_scale / 2.0, H * shift_scale / 2.0
    cdx = Mm[0, 0] * cx + Mm[0, 1] * cy + Mm[0, 2] - cx
    cdy = Mm[1, 0] * cx + Mm[1, 1] * cy + Mm[1, 2] - cy
    return Mm, (float(cdx), float(cdy)), float(response), inliers


"""
def loadMoreClasses(filename,classes_dict):
    with open("%s.json"%filename) as json_data:
        data          = json.load(json_data)
        point_clicks  = data.get("pointClicks", [])
        point_classes = data.get("pointClasses", [])
        for cl in point_classes:
           #print("Add `",cl,"` class ")
           classes_dict[cl]=True 
    return classes_dict 
"""





class PhotoCtrl(wx.App):
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

        """
['ID_ABORT', 'ID_ABOUT', 'ID_ADD', 'ID_ANY', 'ID_APPLY', 'ID_BACKWARD', 'ID_BOLD
', 'ID_CANCEL', 'ID_CLEAR', 'ID_CLOSE', 'ID_CLOSE_ALL', 'ID_CONTEXT_HELP', 'ID_C
OPY', 'ID_CUT', 'ID_DEFAULT', 'ID_DELETE', 'ID_DOWN', 'ID_DUPLICATE', 'ID_EDIT',
 'ID_EXIT', 'ID_FILE', 'ID_FILE1', 'ID_FILE2', 'ID_FILE3', 'ID_FILE4', 'ID_FILE5
', 'ID_FILE6', 'ID_FILE7', 'ID_FILE8', 'ID_FILE9', 'ID_FIND', 'ID_FORWARD', 'ID_
HELP', 'ID_HELP_COMMANDS', 'ID_HELP_CONTENTS', 'ID_HELP_CONTEXT', 'ID_HELP_INDEX
', 'ID_HELP_PROCEDURES', 'ID_HELP_SEARCH', 'ID_HIGHEST', 'ID_HOME', 'ID_IGNORE',
 'ID_INDENT', 'ID_INDEX', 'ID_ITALIC', 'ID_JUSTIFY_CENTER', 'ID_JUSTIFY_FILL', '
ID_JUSTIFY_LEFT', 'ID_JUSTIFY_RIGHT', 'ID_LOWEST', 'ID_MORE', 'ID_NEW', 'ID_NO',
 'ID_NONE', 'ID_NOTOALL', 'ID_OK', 'ID_OPEN', 'ID_PAGE_SETUP', 'ID_PASTE', 'ID_P
REFERENCES', 'ID_PREVIEW', 'ID_PREVIEW_CLOSE', 'ID_PREVIEW_FIRST', 'ID_PREVIEW_G
OTO', 'ID_PREVIEW_LAST', 'ID_PREVIEW_NEXT', 'ID_PREVIEW_PREVIOUS', 'ID_PREVIEW_P
RINT', 'ID_PREVIEW_ZOOM', 'ID_PRINT', 'ID_PRINT_SETUP', 'ID_PROPERTIES', 'ID_RED
O', 'ID_REFRESH', 'ID_REMOVE', 'ID_REPLACE', 'ID_REPLACE_ALL', 'ID_RESET', 'ID_R
ETRY', 'ID_REVERT', 'ID_REVERT_TO_SAVED', 'ID_SAVE', 'ID_SAVEAS', 'ID_SELECTALL'
, 'ID_SEPARATOR', 'ID_SETUP', 'ID_STATIC', 'ID_STOP', 'ID_UNDELETE', 'ID_UNDERLI
NE', 'ID_UNDO', 'ID_UNINDENT', 'ID_UP', 'ID_VIEW_DETAILS', 'ID_VIEW_LARGEICONS',
 'ID_VIEW_LIST', 'ID_VIEW_SMALLICONS', 'ID_VIEW_SORTDATE', 'ID_VIEW_SORTNAME', '
ID_VIEW_SORTSIZE', 'ID_VIEW_SORTTYPE', 'ID_YES', 'ID_YESTOALL', 'ID_ZOOM_100', '
ID_ZOOM_FIT', 'ID_ZOOM_IN', 'ID_ZOOM_OUT']"""


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
        """Return a list of (pth, json) pairs for all valid allclass_* models in directory."""
        import zipfile
        result = []
        if not os.path.isdir(directory):
            return result
        for name in sorted(os.listdir(directory)):
            if not name.startswith("allclass_") or not name.endswith(".pth"):
                continue
            base     = name[:-4]
            pth_path = os.path.join(directory, f"{base}.pth")
            cfg_path = os.path.join(directory, f"{base}.json")
            if not os.path.isfile(cfg_path):
                continue
            if not zipfile.is_zipfile(pth_path):
                print(f"[Ensemble] Skipping corrupted/incomplete: {pth_path}")
                continue
            print(f"[Ensemble] Adding model: {base}")
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
           _ensemble_initial = ("../magician_vision_classifier/allclass_verysmall_cnn.pth","../magician_vision_classifier/allclass_verysmall_cnn.json")
           _ensemble_models  = self._scan_allclass_models(classifier_relative_directory)
           if (not _ensemble_models) or (not os.path.isfile(_ensemble_initial[0])) or (not os.path.isfile(_ensemble_initial[1])):
              wx.MessageBox(
                  "Ensemble classifier disabled: no usable models found.\n\n"
                  f"The ensemble scans {classifier_relative_directory} for pairs named\n"
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
    self.Bind(wx.EVT_MENU, self.onOpenMagnifier,itemMagnify)
    self.Bind(wx.EVT_MENU, self.onRecordDataset,itemRecordDataset)
    self.Bind(wx.EVT_MENU, self.onCreateDataset,itemCreateDataset)
    self.Bind(wx.EVT_MENU, self.onTileExplorer,itemTileExplorer)
    self.Bind(wx.EVT_MENU, self.onStreamer,itemStreamer)
    self.Bind(wx.EVT_MENU, self.onBenchmarkPerf,itemBenchmarkPerf)
    self.Bind(wx.EVT_MENU, self.onBenchmarkAcc,itemBenchmarkAcc)
    self.Bind(wx.EVT_MENU, self.onMakeVideo, itemMakeVideo)
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
    self.brightnessSlider = wx.Slider(self.panel, value=0, minValue=0, maxValue=5,
                                      size=(90, -1), style=wx.SL_HORIZONTAL)
    self.brightnessSlider.SetToolTip("Brightness offset (0-5)")
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
    self.cameraSettingsBtn = wx.Button(self.panel, label='Camera')
    self.cameraSettingsBtn.Bind(wx.EVT_BUTTON, self.onCameraSettings)


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
    self.underImage.Add(self.cameraSettingsBtn, 0, wx.ALL, 5)
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
    """Builds the Classifier tab with model select, threshold, majority voting,
       tile size (4..128), and two-stage classification toggle."""
    s = wx.BoxSizer(wx.VERTICAL)

    # --- 1. Get available models from directory ---
    model_dir = classifier_relative_directory
    available_models = ClassifierPnm.model_scan(model_dir)
    if available_models:
        global classifier_model_path
        classifier_model_path = "%s/%s.pth"  % (classifier_relative_directory, available_models[0])
        global classifier_cfg_path
        classifier_cfg_path   = "%s/%s.json" % (classifier_relative_directory, available_models[0])
    else:
        available_models = ["(none)"]

    # Models on the online zip repository (CameraV2Models). Already-local names
    # are listed too — downloading them fetches the server's newest archive.
    def _list_remote(local_models):
        try:
            from ModelDownload import remote_model_names
            local = set(local_models)
            return [n + (" [have local copy]" if n in local else "")
                    for n in remote_model_names(timeout=5)]
        except Exception as e:
            print(f"[Models] Online repository unavailable: {e}")
            return []
    self._list_remote = _list_remote

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
    LOCAL_MARK = " [have local copy]"
    remoteRow = wx.BoxSizer(wx.HORIZONTAL)
    self.remoteModelsLbl = wx.StaticText(parent, label="Online")
    remote_models = self._list_remote(available_models)

    def _remote_summary(remote):
        if not remote:
            return "Online (offline)"
        have = sum(1 for m in remote if m.endswith(LOCAL_MARK))
        return f"Online ({have}/{len(remote)} local)"

    self.remoteModelsLbl.SetLabel(_remote_summary(remote_models))
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

    def _refresh_model_lists(select=None):
        local = ClassifierPnm.model_scan(model_dir)
        self.classifierModelCombo.Clear()
        for m in local:
            self.classifierModelCombo.Append(m)
        if select and select in local:
            self.classifierModelCombo.SetValue(select)
        elif local:
            self.classifierModelCombo.SetValue(local[0])
        remote = self._list_remote(local)
        self.remoteModelsLbl.SetLabel(_remote_summary(remote))
        self.classifierRemoteCombo.Clear()
        for m in (remote if remote else ["(repository unreachable)"]):
            self.classifierRemoteCombo.Append(m)
        self.classifierRemoteCombo.SetValue((remote or ["(repository unreachable)"])[0])
        remoteRow.Layout()
    self._refresh_model_lists = _refresh_model_lists

    def onDownloadModel(_evt):
        name = self.classifierRemoteCombo.GetValue()
        if not name or name.startswith("("):
            return
        if name.endswith(LOCAL_MARK):
            name = name[:-len(LOCAL_MARK)]
        self.classifierInfo.SetLabel(f"Downloading '{name}' from the model repository...")
        busy = wx.BusyCursor()
        wx.Yield()
        try:
            from ModelDownload import download_model
            download_model(name, model_dir)
        except Exception as e:
            wx.MessageBox(f"Failed to download '{name}':\n{e}",
                          "Model Download Error", wx.OK | wx.ICON_ERROR)
            return
        finally:
            del busy
        _refresh_model_lists(select=name)
        # Load the freshly downloaded model right away
        if useClassifier and self.ClassifierPnm is not None:
            if self.ClassifierPnm.reload_model(model_dir, name):
                self.stats.classifier_name = name.lower()
                self.stats.reset()
                self.classifierInfo.SetLabel(f"Downloaded and switched to '{name}'.")
            else:
                wx.MessageBox(f"Downloaded '{name}' but failed to load it.",
                              "Model Load Error", wx.OK | wx.ICON_ERROR)
    self.downloadModelBtn.Bind(wx.EVT_BUTTON, onDownloadModel)

    # --- 3. Callback to reload model when changed ---
    def onClassifierModelChanged(evt):
        model_name = self.classifierModelCombo.GetValue()
        print(f"[INFO] Changing classifier model to: {model_name}")
        if useClassifier and self.ClassifierPnm is not None:
            success = self.ClassifierPnm.reload_model(model_dir, model_name)
            if success:
                print(f"Successfully reloaded model: {model_name}")
                self.stats.classifier_name = model_name.lower()
                self.stats.reset()
                self.classifierInfo.SetLabel(f"Model changed to '{model_name}' — statistics reset.")
            else:
                pth = os.path.join(model_dir, f"{model_name}.pth")
                answer = wx.MessageBox(
                    f"Failed to load '{model_name}'.\n\n"
                    f"The file may be corrupted or incomplete:\n{pth}\n\n"
                    f"Re-download it now?",
                    "Model Load Error", wx.YES_NO | wx.ICON_ERROR
                )
                if answer == wx.YES:
                    from modelUpdater import ModelUpdaterDialog
                    dlg = ModelUpdaterDialog(self.frame, classifier_online_repository, model_dir)
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

    self.classifierModelCombo.Bind(wx.EVT_COMBOBOX, onClassifierModelChanged)

    # --- 4. Threshold slider ---
    thrRow = wx.BoxSizer(wx.HORIZONTAL)
    thrLbl = wx.StaticText(parent, label="Threshold")
    self.classifierThreshold = wx.Slider(parent, value=85, minValue=0, maxValue=100, style=wx.SL_HORIZONTAL)
    self.classifierThresholdValue = wx.StaticText(parent, label="0.85")
    def _on_thr(evt):
        self.classifierThresholdValue.SetLabel(f"{self.classifierThreshold.GetValue()/100.0:.2f}")
        evt.Skip()
    self.classifierThreshold.Bind(wx.EVT_SLIDER, _on_thr)
    thrRow.Add(thrLbl, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 8)
    thrRow.Add(self.classifierThreshold, 1, wx.RIGHT, 8)
    thrRow.Add(self.classifierThresholdValue, 0, wx.ALIGN_CENTER_VERTICAL)

    # --- 5. Majority voting checkbox ---
    self.classifierMajorityVoting = wx.CheckBox(parent, label="Use majority voting")
    self.classifierMajorityVoting.SetValue(True)

    # --- 6. Tile size slider ---
    tileRow = wx.BoxSizer(wx.HORIZONTAL)
    tileLbl = wx.StaticText(parent, label="Step size")
    self.classifierTileSize = wx.Slider(parent, value=16, minValue=4, maxValue=128, style=wx.SL_HORIZONTAL)
    self.classifierTileSizeValue = wx.StaticText(parent, label="16")
    def _on_tile(evt):
        self.classifierTileSizeValue.SetLabel(str(self.classifierTileSize.GetValue()))
        evt.Skip()
    self.classifierTileSize.Bind(wx.EVT_SLIDER, _on_tile)
    tileRow.Add(tileLbl, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 8)
    tileRow.Add(self.classifierTileSize, 1, wx.RIGHT, 8)
    tileRow.Add(self.classifierTileSizeValue, 0, wx.ALIGN_CENTER_VERTICAL)


    # --- Erode Kernel Size  ---
    erodeKernelRow        = wx.BoxSizer(wx.HORIZONTAL)
    erodeKernelLbl        = wx.StaticText(parent, label="Vote Neighborhood (kernel)")
    erodeKernelLbl.SetToolTip("Radius k of the tile-voting neighborhood: votes are counted "
                              "over the (2k+1)x(2k+1) tiles around each activation")
    self.erodeKernelSize  = wx.Slider(parent, value=1, minValue=0, maxValue=8, style=wx.SL_HORIZONTAL)
    self.erodeKernelValue = wx.StaticText(parent, label="1")
    def _on_erodkrnthr(evt):
        self.erodeKernelValue.SetLabel(f"{self.erodeKernelSize.GetValue()}")
        evt.Skip()
    self.erodeKernelSize.Bind(wx.EVT_SLIDER, _on_erodkrnthr)
    erodeKernelRow.Add(erodeKernelLbl, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 8)
    erodeKernelRow.Add(self.erodeKernelSize, 1, wx.RIGHT, 8)
    erodeKernelRow.Add(self.erodeKernelValue, 0, wx.ALIGN_CENTER_VERTICAL)


    # --- Erode Threshold Value ---
    erodeThresholdRow        = wx.BoxSizer(wx.HORIZONTAL)
    erodeThresholdLbl        = wx.StaticText(parent, label="Min Votes to Keep Tile")
    erodeThresholdLbl.SetToolTip("Activated tiles (including the tile itself) required inside the "
                                 "vote neighborhood for an activation to be accepted; 0/1 = voting off. "
                                 "Same setting as the ROS set_min_votes service.")
    self.erodeThreshold      = wx.Slider(parent, value=2, minValue=0, maxValue=8, style=wx.SL_HORIZONTAL)
    self.erodeThresholdValue = wx.StaticText(parent, label="2")
    def _on_erodthr(evt):
        self.erodeThresholdValue.SetLabel(f"{self.erodeThreshold.GetValue()}")
        evt.Skip()
    self.erodeThreshold.Bind(wx.EVT_SLIDER, _on_erodthr)
    erodeThresholdRow.Add(erodeThresholdLbl, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 8)
    erodeThresholdRow.Add(self.erodeThreshold, 1, wx.RIGHT, 8)
    erodeThresholdRow.Add(self.erodeThresholdValue, 0, wx.ALIGN_CENTER_VERTICAL)


    # --- 7. Two-stage classification checkbox ---
    self.classifierTwoStage = wx.CheckBox(parent, label="Enable two-stage classification")
    self.parallellTwoStage  = wx.CheckBox(parent, label="Two-stage parallelism (VRAM intensive)")
    self.parallellTwoStage.SetValue(True)

    def _on_two_stage_toggled(evt):
        if self.classifierTwoStage.GetValue():
            self.stats.classifier_name = "allclass_ensemble"
        else:
            self.stats.classifier_name = self.classifierModelCombo.GetValue().lower()
        self.stats.reset()
        self.classifierInfo.SetLabel(
            f"Switched to {'two-stage ensemble' if self.classifierTwoStage.GetValue() else self.classifierModelCombo.GetValue()} — statistics reset.")
        evt.Skip()

    self.classifierTwoStage.Bind(wx.EVT_CHECKBOX, _on_two_stage_toggled)

    # --- 7b. Min Hz filter for ensemble (applied at init time) ---
    minHzRow = wx.BoxSizer(wx.HORIZONTAL)
    minHzLbl = wx.StaticText(parent, label="Ensemble min Hz filter:")
    self.ensembleMinHz = wx.TextCtrl(parent, value="10", size=(55, -1), style=wx.TE_PROCESS_ENTER)
    self.ensembleMinHz.SetToolTip(
        "Drop ensemble models slower than this Hz.\n"
        "0 = keep all models.  Press Enter or click away to apply immediately.")
    minHzRow.Add(minHzLbl, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 6)
    minHzRow.Add(self.ensembleMinHz, 0, wx.ALIGN_CENTER_VERTICAL)

    def _apply_min_hz(_evt=None):
        if not useClassifier:
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

    self.ensembleMinHz.Bind(wx.EVT_TEXT_ENTER, _apply_min_hz)
    self.ensembleMinHz.Bind(wx.EVT_KILL_FOCUS,  _apply_min_hz)

    # --- 8. "Disabled Model" checkbox (active by default — NN off until user enables it) ---
    self.classifierDisabledCheckbox = wx.CheckBox(parent, label="Disable Neural Network Model (For Speed)")
    self.classifierDisabledCheckbox.SetValue(True)

    # --- 9. Layout ---
    s.Add(modelRow, 0, wx.ALL | wx.EXPAND, 10)
    s.Add(remoteRow, 0, wx.LEFT | wx.RIGHT | wx.TOP | wx.EXPAND, 10)
    s.Add(self.downloadModelBtn, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM | wx.EXPAND, 10)
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

    def _on_check_updates(_evt):
        dlg = ModelUpdaterDialog(self.frame, classifier_online_repository, classifier_relative_directory)
        dlg.ShowModal()
        # Refresh model list after dialog closes
        updated_models = ClassifierPnm.model_scan(classifier_relative_directory)
        if updated_models:
            self.classifierModelCombo.Clear()
            for m in updated_models:
                self.classifierModelCombo.Append(m)
            self.classifierModelCombo.SetValue(updated_models[0])
        dlg.Destroy()

    self.checkUpdatesBtn.Bind(wx.EVT_BUTTON, _on_check_updates)

    # --- "Check Model Statistics" button ---
    self.checkStatsBtn = wx.Button(parent, label="Check Model Statistics")
    s.Add(self.checkStatsBtn, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

    def _on_check_stats(_evt):
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

    self.checkStatsBtn.Bind(wx.EVT_BUTTON, _on_check_stats)

    # --- "Reinforcement Learning" button + pixel-distance textbox ---
    rl_row = wx.BoxSizer(wx.HORIZONTAL)
    self.rlBtn = wx.Button(parent, label="Reinforcement Learning")
    rl_row.Add(self.rlBtn, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 8)
    rl_row.Add(wx.StaticText(parent, label="Radius (px):"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 4)
    self.rlRadiusCtrl = wx.TextCtrl(parent, value="120", size=(55, -1))
    rl_row.Add(self.rlRadiusCtrl, 0, wx.ALIGN_CENTER_VERTICAL)
    s.Add(rl_row, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

    def _on_rl(_evt):
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

    self.rlBtn.Bind(wx.EVT_BUTTON, _on_rl)

    # --- "Purge R/L Labels" button ---
    self.purgeRLBtn = wx.Button(parent, label="Purge R/L Labels")
    s.Add(self.purgeRLBtn, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

    def _on_purge_rl(_evt):
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

        from readData import list_image_files
        from rlAnnotator import _resolve_json
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

    self.purgeRLBtn.Bind(wx.EVT_BUTTON, _on_purge_rl)

    self.classifierInfo = wx.StaticText(parent, label="No classifier run yet.")
    s.Add(self.classifierInfo, 0, wx.ALL | wx.EXPAND, 5)


    s.AddStretchSpacer(1)

    parent.SetSizer(s)
#===============================================================================
#===============================================================================
#===============================================================================
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
    from tactilePlotter import SensorVisualizer, load_csv_with_headers, load_csv_without_headers

    
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
                self.points_sources = list(data['pointSources'])
            else:
                self.points_sources = ["manual"] * len(self.points_of_interest)
            # Keep parallel array aligned with the points in case of malformed input.
            if len(self.points_sources) < len(self.points_of_interest):
                self.points_sources += ["manual"] * (len(self.points_of_interest) - len(self.points_sources))
            else:
                self.points_sources = self.points_sources[:len(self.points_of_interest)]
            if 'regionClicks' in data:
                self.regions_of_interest = data['regionClicks']

            if 'lightDirection' in data:
                   self.lightComboBox.SetValue(data['lightDirection'])
            else:
                   self.lightComboBox.SetValue("Unknown")

            # Inter-frame transform records from the Track button (see onTrack).
            # A list of records; a bare dict (early format) is wrapped for compatibility.
            tr = data.get('tracking', None)
            self.tracking = [tr] if isinstance(tr, dict) else tr

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
      self.ClassifierPnm.maxProbabilityThreshold = float(self.classifierThreshold.GetValue() / 100.0)
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
          if any((fx - px) ** 2 + (fy - py) ** 2 < min_dist ** 2 for px, py in pts):
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
          if any((x - px) ** 2 + (y - py) ** 2 < tileFull ** 2
                 for px, py in self.points_of_interest):
              continue
          self.points_of_interest.append((x, y))
          self.points_classes.append(name)
          self.points_severities.append("AI")
          self.points_sources.append("classifier")
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
              self.points_of_interest.append((x, y))
              self.points_classes.append(self.defectComboBox.GetValue() or options[0])
              self.points_severities.append(self.severityComboBox.GetValue() or severities[0])
              self.points_sources.append("auto")
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
          self.points_of_interest.append((pred[0], pred[1]))
          self.points_classes.append(prev_classes[i] if i < len(prev_classes) else options[0])
          self.points_severities.append(prev_severities[i] if i < len(prev_severities) else severities[0])
          self.points_sources.append("auto")
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

      jp = resolve_annotation_json_path(img_path, prefer_existing=True)
      if not jp or not checkIfFileExists(jp):
          jp = os.path.splitext(img_path)[0] + ".json"
      data = {}
      if checkIfFileExists(jp):
          try:
              with open(jp) as f:
                  data = json.load(f)
          except Exception:
              data = {}
      light = data.get("lightDirection", "Unknown")
      if light not in ("", "Unknown"):
          return light

      raw = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
      if raw is None:
          return "Unknown"
      light = determine_intensity_region(raw, threshold=0.1)
      if data:
          data["lightDirection"] = light
          try:
              with open(jp, "w") as f:
                  json.dump(data, f, sort_keys=False)
          except Exception as e:
              print("Track: light direction write failed", jp, e)
      return light

   def _lightFingerprintCached(self, path):
      fp = self._light_fp_cache.get(path)
      if fp is None:
          fp = lightingFingerprint(path)
          if fp is not None:
              self._light_fp_cache[path] = fp
      return fp

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
      fp_next = self._lightFingerprintCached(next_path)
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
              fp = self._lightFingerprintCached(path)
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
          rec = self.tracking[0]
          s = rec.get("shift")
          a = self._streamIndexOfFrame(rec.get("fromFrame", ""))
          if s and a is not None:
              if a == cur - direction:
                  prior_shift = (s[0], s[1])
              elif a == cur + direction:
                  prior_shift = (-s[0], -s[1])

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
              records.append({"fromFrame": os.path.basename(same_light_path),
                              "shift": [sdx, sdy],
                              "affine": sM.tolist(),
                              "response": sresp,
                              "inliers": sinl,
                              "method": "phaseCorrelateAffine" if sinl else "phaseCorrelate",
                              "fallback": False,
                              "lightSimilarity": round(same_light_sim, 3)})
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
      carried = 0
      for i, (x, y) in enumerate(prev_points):
          tx = M[0, 0] * x + M[0, 1] * y + M[0, 2]
          ty = M[1, 0] * x + M[1, 1] * y + M[1, 2]
          if not (0 <= tx < W and 0 <= ty < H):
              continue
          # Skip predictions landing on a point the frame already has (re-pressing
          # Track or tracking onto a partially annotated frame must not double up).
          if any((tx - ex) ** 2 + (ty - ey) ** 2 < 30.0 ** 2
                 for ex, ey in self.points_of_interest):
              continue
          self.points_of_interest.append((tx, ty))
          self.points_classes.append(prev_classes[i] if i < len(prev_classes) else options[0])
          self.points_severities.append(prev_severities[i] if i < len(prev_severities) else severities[0])
          self.points_sources.append("auto")
          carried += 1

      self.tracking = [{"fromFrame": os.path.basename(prev_path),
                        "shift": [dx, dy],
                        "affine": M.tolist() if hasattr(M, "tolist") else M,
                        "response": response,
                        "inliers": inliers,
                        "method": "phaseCorrelateAffine" if inliers else "phaseCorrelate",
                        "fallback": fallback}] + records

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
      images = list(getattr(self.folderStreamer, "directoryList", None) or [])
      if not images and self.filepath and os.path.isfile(self.filepath):
          images = list_image_files(os.path.dirname(self.filepath))
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
      Returns (filled, skipped, failed, solved, aborted), or None with <2 frames."""
      images = list(getattr(self.folderStreamer, "directoryList", None) or [])
      if not images and self.filepath and os.path.isfile(self.filepath):
          images = list_image_files(os.path.dirname(self.filepath))
      if len(images) < 2:
          return None

      # Flush the currently-open frame so its JSON is up to date before the pass.
      self.onSave(None)

      def json_path_for(img):
          jp = resolve_annotation_json_path(img, prefer_existing=True)
          if not jp or not checkIfFileExists(jp):
              jp = os.path.splitext(img)[0] + ".json"
          return jp

      def read_json(jp):
          if checkIfFileExists(jp):
              try:
                  with open(jp) as f:
                      return json.load(f)
              except Exception:
                  pass
          return {}

      def tracking_list(data):
          tr = data.get("tracking", None)
          return [tr] if isinstance(tr, dict) else (tr or [])

      prog = wx.ProgressDialog(
          "Fill Tracking", "Measuring inter-frame shifts…", maximum=len(images),
          parent=self.frame,
          style=wx.PD_APP_MODAL | wx.PD_AUTO_HIDE | wx.PD_CAN_ABORT
                | wx.PD_ELAPSED_TIME | wx.PD_REMAINING_TIME)

      # PASS 1 — measure missing records.
      filled = skipped = failed = 0
      aborted = False
      for i in range(1, len(images)):
          cont, _ = prog.Update(
              i, f"{i+1}/{len(images)} — {filled} filled, {skipped} had tracking")
          if not cont:
              aborted = True
              break
          wx.GetApp().Yield(True)

          jp = json_path_for(images[i])
          data = read_json(jp)
          if tracking_list(data):
              skipped += 1
              continue

          try:
              M, (dx, dy), response, inliers = estimateFrameAffine(images[i - 1], images[i])
          except Exception as e:
              print("Fill Tracking: shift failed for", images[i], ":", e)
              failed += 1
              continue
          records = [{"fromFrame": os.path.basename(images[i - 1]),
                      "shift": [dx, dy],
                      "affine": M.tolist(),
                      "response": response,
                      "inliers": inliers,
                      "method": "phaseCorrelateAffine" if inliers else "phaseCorrelate",
                      "fallback": False}]

          # Best same-lighting link among the preceding frames (i-1 excluded).
          fp_i = self._lightFingerprintCached(images[i])
          best_j, best_sim = None, 0.0
          if fp_i is not None:
              for j in range(i - 2, max(-1, i - 2 - SAME_LIGHT_SEARCH_MAX), -1):
                  fp_j = self._lightFingerprintCached(images[j])
                  if fp_j is None:
                      continue
                  sim = float(fp_i @ fp_j)
                  if sim > best_sim:
                      best_j, best_sim = j, sim
          if best_j is not None and best_sim >= SAME_LIGHT_MIN_SIMILARITY:
              try:
                  sM, (sdx, sdy), sresp, sinl = estimateFrameAffine(images[best_j], images[i])
                  records.append({"fromFrame": os.path.basename(images[best_j]),
                                  "shift": [sdx, sdy],
                                  "affine": sM.tolist(),
                                  "response": sresp,
                                  "inliers": sinl,
                                  "method": "phaseCorrelateAffine" if sinl else "phaseCorrelate",
                                  "fallback": False,
                                  "lightSimilarity": round(best_sim, 3)})
              except Exception as e:
                  print("Fill Tracking: same-light shift failed for", images[i], ":", e)

          if not data:
              data = {"width": self.width, "height": self.height, "md5hash": "",
                      "regionClicks": [], "pointClicks": [], "pointClasses": [],
                      "pointSeverities": [], "pointSources": []}
          data["tracking"] = records
          try:
              with open(jp, "w") as f:
                  json.dump(data, f, sort_keys=False)
              filled += 1
          except Exception as e:
              print("Fill Tracking: write failed", jp, e)
              failed += 1

      prog.Destroy()

      # PASS 2 — weighted least squares over the pose graph: solve for per-frame
      # global positions p_i (p_0 = 0) from all pairwise measurements p_b - p_a = s.
      solved = 0
      if not aborted:
          name2idx = {os.path.basename(p): i for i, p in enumerate(images)}
          frame_json = []   # (json_path, data) per frame, aligned with images
          measurements = []  # (a, b, dx, dy, weight)
          for i, img in enumerate(images):
              jp = json_path_for(img)
              data = read_json(jp)
              frame_json.append((jp, data))
              for r in tracking_list(data):
                  if r.get("method") == "leastSquaresGlobal":
                      continue
                  a = name2idx.get(r.get("fromFrame", ""))
                  s = r.get("shift")
                  if a is None or a == i or not s:
                      continue
                  w = max(float(r.get("response") or 0.0), 0.01)
                  if r.get("fallback"):
                      w *= 0.3
                  if r.get("method") == "manual":
                      w = 2.0   # hand-corrected shifts outweigh any estimate
                  measurements.append((a, i, float(s[0]), float(s[1]), w))

          if measurements:
              N = len(images)
              A  = np.zeros((len(measurements), N - 1), np.float64)
              bx = np.zeros(len(measurements), np.float64)
              by = np.zeros(len(measurements), np.float64)
              constrained = set()
              for row, (a, b, dx, dy, w) in enumerate(measurements):
                  if b > 0:
                      A[row, b - 1] += w
                  if a > 0:
                      A[row, a - 1] -= w
                  bx[row] = w * dx
                  by[row] = w * dy
                  constrained.update((a, b))
              px = np.linalg.lstsq(A, bx, rcond=None)[0]
              py = np.linalg.lstsq(A, by, rcond=None)[0]

              first = os.path.basename(images[0])
              for i in range(1, N):
                  if i not in constrained:
                      continue  # unmeasured frame: a stored zero would be a lie
                  jp, data = frame_json[i]
                  if not data:
                      continue
                  recs = [r for r in tracking_list(data)
                          if r.get("method") != "leastSquaresGlobal"]
                  recs.append({"fromFrame": first,
                               "shift": [float(px[i - 1]), float(py[i - 1])],
                               "method": "leastSquaresGlobal"})
                  data["tracking"] = recs
                  try:
                      with open(jp, "w") as f:
                          json.dump(data, f, sort_keys=False)
                      solved += 1
                  except Exception as e:
                      print("Fill Tracking: write failed", jp, e)

      # The pass may have rewritten the open frame's JSON — reload it.
      self.onProcessNewImageSample(self.filepath)
      self.onView()
      return filled, skipped, failed, solved, aborted

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
          for i, src in enumerate(self.points_sources):
              if src == "auto" and i < len(self.points_of_interest):
                  x, y = self.points_of_interest[i]
                  self.points_of_interest[i] = (x + ddx, y + ddy)
          if self.tracking:
              rec = self.tracking[0]
              sx, sy = rec.get("shift", [0, 0])
              rec["shift"]  = [sx + ddx, sy + ddy]
              aff = rec.get("affine")
              if aff:
                  aff[0][2] += ddx
                  aff[1][2] += ddy
              rec["method"] = "manual"
          self.updatePointList()
          self.onView()

      up    = wx.Button(panel, label="▲");  up.Bind(wx.EVT_BUTTON,    lambda e: nudge(0, -step.GetValue()))
      down  = wx.Button(panel, label="▼");  down.Bind(wx.EVT_BUTTON,  lambda e: nudge(0,  step.GetValue()))
      left  = wx.Button(panel, label="◀");  left.Bind(wx.EVT_BUTTON,  lambda e: nudge(-step.GetValue(), 0))
      right = wx.Button(panel, label="▶");  right.Bind(wx.EVT_BUTTON, lambda e: nudge( step.GetValue(), 0))
      done  = wx.Button(panel, label="Done"); done.Bind(wx.EVT_BUTTON, lambda e: dlg.EndModal(wx.ID_OK))

      grid = wx.GridSizer(3, 3, 2, 2)
      for item in (wx.StaticText(panel), up, wx.StaticText(panel),
                   left, done, right,
                   wx.StaticText(panel), down, wx.StaticText(panel)):
          grid.Add(item, 0, wx.EXPAND)
      col = wx.BoxSizer(wx.VERTICAL)
      row = wx.BoxSizer(wx.HORIZONTAL)
      row.Add(wx.StaticText(panel, label="Step (mosaic px):"), 0,
              wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 4)
      row.Add(step, 0)
      col.Add(grid, 0, wx.ALL | wx.ALIGN_CENTER, 8)
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
      images = list(getattr(self.folderStreamer, "directoryList", None) or [])
      if not images and self.filepath and os.path.isfile(self.filepath):
          images = list_image_files(os.path.dirname(self.filepath))
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

          json_path = resolve_annotation_json_path(img, prefer_existing=True)
          if not json_path or not checkIfFileExists(json_path):
              json_path = os.path.splitext(img)[0] + ".json"

          # Skip frames that already carry annotations.
          if checkIfFileExists(json_path):
              try:
                  if json.load(open(json_path)).get("pointClicks"):
                      skipped += 1
                      continue
              except Exception:
                  pass

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
              if any((cx - px) ** 2 + (cy - py) ** 2 < 96 ** 2 for px, py in points):
                  continue
              points.append([cx, cy]); classes.append(cname)
              sevs.append("AI"); sources.append("classifier")
              classifier_marks += 1
          data = {
              "width":  self.width,
              "height": self.height,
              "md5hash": "",
              "regionClicks": [],
              "pointClicks":     points,
              "pointClasses":    classes,
              "pointSeverities": sevs,
              "pointSources":    sources,
          }
          try:
              with open(json_path, "w") as f:
                  json.dump(data, f, sort_keys=False)
              annotated += 1
              marks     += len(dets)
          except Exception as e:
              print(f"[Full Auto] Failed writing {json_path}: {e}")
              failed += 1

      # Frames where ONLY the classifier found something (no pen mark)
      for json_path, (c_pts, c_cls) in classifier_dets.items():
          data = {
              "width":  self.width,
              "height": self.height,
              "md5hash": "",
              "regionClicks": [],
              "pointClicks":     [[x, y] for x, y in c_pts],
              "pointClasses":    list(c_cls),
              "pointSeverities": ["AI"] * len(c_pts),
              "pointSources":    ["classifier"] * len(c_pts),
          }
          try:
              with open(json_path, "w") as f:
                  json.dump(data, f, sort_keys=False)
              annotated += 1
              classifier_marks += len(c_pts)
          except Exception as e:
              print(f"[Full Auto] Failed writing {json_path}: {e}")
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

   def onAutoOLD(self, event):
      print("Automatically retrieved annotations")
      print(self.AIAnnotations)
      print("User Submitted annotations")
      print(self.points_of_interest)
      print(self.points_classes)
      print(self.points_severities)
      tileSize = self.ClassifierPnm.tile_size

      # === Prepare cleaned lists ===
      new_points     = []
      new_classes    = []
      new_severities = []

      ai_points  = self.AIAnnotations.get("points", [])
      ai_classes = self.AIAnnotations.get("classes", [])

      # Helper: check if two tiles overlap
      def tiles_overlap(pt1, pt2):
        half = tileSize / 2
        x1_min, y1_min = pt1[0] - half, pt1[1] - half
        x1_max, y1_max = pt1[0] + half, pt1[1] + half

        x2_min, y2_min = pt2[0] - half, pt2[1] - half
        x2_max, y2_max = pt2[0] + half, pt2[1] + half

        # Overlap if bounding boxes intersect
        return not (x1_max < x2_min or x1_min > x2_max or
                    y1_max < y2_min or y1_min > y2_max)

      # === Iterate over AI annotations ===
      for pt, cls in zip(ai_points, ai_classes):
        # Skip if this AI tile overlaps any user-provided tile
        if any(tiles_overlap(pt, user_pt) for user_pt in self.points_of_interest):
            continue

        # Replace AI class with 'class Clean' since we assume AI is unreliable
        new_points.append(pt)
        new_classes.append("Clean")
        new_severities.append("AI")

      # === Store the cleaned results ===
      self.cleaned_points    = new_points
      self.cleaned_classes   = new_classes
      self.cleaned_severities = new_severities

      # === Print summary ===
      print("\nGenerated cleaned annotations:")
      for p, c in zip(new_points, new_classes):
        print(f"{p} -> {c}")

      self.points_of_interest.extend(new_points)
      self.points_classes.extend(new_classes)
      self.points_severities.extend(new_severities)
      self.points_sources.extend(["auto"] * len(new_points))


      print(f"Total new clean tiles: {len(new_points)}")

      if (self.incrementFrameAfterAnAdditionCheckbox.GetValue()):
          print(f"Auto incrementing due to checkbox")
          self.onNext(event)


      return new_points, new_classes
        

   def onSave(self, event):
        print("Save")
        if not self.filepath or not os.path.isfile(self.filepath):
            print("onSave: skipping — filepath is not a valid file:", self.filepath)
            return

        allData = dict()
        allData["width"]   = self.width #self.leftViewImage.shape[1]
        allData["height"]  = self.height #self.leftViewImage.shape[0] 
        allData["md5hash"] = self.filehash

        if self.tenengrad_focus_measure != 0.0:
            allData["tenengradFocusMeasure"] = self.tenengrad_focus_measure


        allData["regionClicks"] = list()
        for x, y in self.regions_of_interest:
              allData["regionClicks"].append((x,y))

        allData["pointClicks"] = list()
        for x, y in self.points_of_interest:
              allData["pointClicks"].append((x,y))

        allData["pointClasses"] = list()
        for aClass in self.points_classes:
              allData["pointClasses"].append(aClass)

        allData["pointSeverities"] = list()
        for aSeverity in self.points_severities:
              allData["pointSeverities"].append(aSeverity)

        # Per-point origin ("auto" | "manual"), aligned to pointClicks. Any point lacking
        # a recorded source (e.g. carried over from older code paths) is treated as manual.
        allData["pointSources"] = list()
        for i in range(len(self.points_of_interest)):
              src = self.points_sources[i] if i < len(self.points_sources) else "manual"
              allData["pointSources"].append(src)

        if (self.lightComboBox.GetValue()!="Unknown"):
              allData["lightDirection"] = self.lightComboBox.GetValue()

        if self.tracking:
              allData["tracking"] = self.tracking

        #primary_json = resolve_annotation_json_path(self.filepath, prefer_existing=True)
        #fallback_json = f"{self.filepath}.json"

        root, _ = os.path.splitext(self.filepath)
        newstyle_json = root + ".json"  # <-- colorFrame_0_00047.json
        primary_json = resolve_annotation_json_path(self.filepath, prefer_existing=True)
        if (not checkIfFileExists(primary_json)):
                  primary_json = newstyle_json

        try:
          with open(primary_json, "w") as outfile:
            json.dump(allData, outfile, sort_keys=False)
   
          self.folderStreamer.saveJSON()
        except Exception as e:
          print("Warning: Could not write annotations to disk", primary_json, ":", e)



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
               loadDataset = loadDatasetCase.lower()
               #Small check (this will need to  be updated if defects change)..
               if ("weld" in loadDataset):
                   app.defectComboBox.SetValue("Welding")
                   app.severityComboBox.SetValue("Class A")
               if ("positive" in loadDataset):
                   app.defectComboBox.SetValue("Positive Dent")
               if ("negative" in loadDataset):
                   app.defectComboBox.SetValue("Negative Dent")
               if ("class-a" in loadDataset):
                   app.severityComboBox.SetValue("Class A")
               if ("class-b" in loadDataset):
                   app.severityComboBox.SetValue("Class B")
               if ("class-c" in loadDataset):
                   app.severityComboBox.SetValue("Class C")
               if ("pda" in loadDataset) or ("posa" in loadDataset):
                   app.defectComboBox.SetValue("Positive Dent")
                   app.severityComboBox.SetValue("Class A")
               if ("pdb" in loadDataset) or ("posb" in loadDataset):
                   app.defectComboBox.SetValue("Positive Dent")
                   app.severityComboBox.SetValue("Class B")
               if ("pdc" in loadDataset) or ("posc" in loadDataset):
                   app.defectComboBox.SetValue("Positive Dent")
                   app.severityComboBox.SetValue("Class C")
               if ("nda" in loadDataset) or ("nega" in loadDataset):
                   app.defectComboBox.SetValue("Negative Dent")
                   app.severityComboBox.SetValue("Class A")
               if ("ndb" in loadDataset) or ("negb" in loadDataset):
                   app.defectComboBox.SetValue("Negative Dent")
                   app.severityComboBox.SetValue("Class B")
               if ("ndc" in loadDataset) or ("negc" in loadDataset):
                   app.defectComboBox.SetValue("Negative Dent")
                   app.severityComboBox.SetValue("Class C")
               if ("positive-dent-a" in loadDataset):
                   app.defectComboBox.SetValue("Positive Dent")
                   app.severityComboBox.SetValue("Class A")
               if ("positive-dent-b" in loadDataset):
                   app.defectComboBox.SetValue("Positive Dent")
                   app.severityComboBox.SetValue("Class B")
               if ("positive-dent-c" in loadDataset):
                   app.defectComboBox.SetValue("Positive Dent")
                   app.severityComboBox.SetValue("Class C")
               if ("negative-dent-a" in loadDataset):
                   app.defectComboBox.SetValue("Negative Dent")
                   app.severityComboBox.SetValue("Class A")
               if ("negative-dent-b" in loadDataset):
                   app.defectComboBox.SetValue("Negative Dent")
                   app.severityComboBox.SetValue("Class B")
               if ("negative-dent-c" in loadDataset):
                   app.defectComboBox.SetValue("Negative Dent")
                   app.severityComboBox.SetValue("Class C")

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


    self.filepath = self.folderStreamer.getImage()
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
       imgPNM = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
       if imgPNM is None:
           return None
       if (imgPNM.ndim == 3) and (imgPNM.shape[2] == 4):
           imgPNM = repackPolarToMosaic(imgPNM[:, :, 0], imgPNM[:, :, 1], imgPNM[:, :, 2], imgPNM[:, :, 3])
           if self.canonicalLightCheckbox.GetValue():
               imgPNM = self._canonicalizeLighting(imgPNM, filepath)
           imgCV  = cv2.merge([imgPNM, imgPNM, imgPNM])
       else:
           if (imgPNM.ndim == 2) and self.canonicalLightCheckbox.GetValue():
               imgPNM = self._canonicalizeLighting(imgPNM, filepath)
           imgCV  = imgPNM if (imgPNM.ndim == 3 and imgPNM.shape[2] == 3) else cv2.cvtColor(imgPNM, cv2.COLOR_GRAY2BGR)
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

   def _canonicalizeLighting(self, mosaic, filepath, numLights=6):
      """Remap the strobed light of this frame so it renders as light #0.

      Light identity is resolved with the ActiveLighting signature (per-channel
      global mean proportions — position independent). Exemplars come from the
      dataset's first numLights frames, assumed to be one clean strobe cycle
      (same bootstrap as ActiveLighting). The remap is a per-channel gain
      exemplar0/exemplarK applied on the 2x2 mosaic quadrants, so it works for
      both .pnm mosaics and re-bayered .png frames.
      """
      if mosaic is None or mosaic.ndim != 2:
          return mosaic
      dirpath = os.path.dirname(filepath)
      if getattr(self, '_canonLightDir', None) != dirpath:
          self._canonLightDir = dirpath
          self._canonExemplars = None
          # first numLights frames of the dataset = one clean strobe cycle
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
                  self._canonExemplars = exemplars
                  print(f"[CanonicalLight] bootstrapped {numLights} light exemplars from {dirpath}")
              else:
                  print(f"[CanonicalLight] bootstrap failed ({len(exemplars)}/{numLights} usable frames)")
          except Exception as e:
              print(f"[CanonicalLight] bootstrap error: {e}")
      if not self._canonExemplars:
          return mosaic

      quads = (mosaic[0::2, 0::2], mosaic[0::2, 1::2], mosaic[1::2, 0::2], mosaic[1::2, 1::2])
      means = np.array([float(q.mean()) for q in quads], dtype=np.float32)
      sig = means / max(float(means.sum()), 1e-6)
      dists = [float(np.linalg.norm(sig - e['sig'])) for e in self._canonExemplars]
      k = int(np.argmin(dists))
      srt = sorted(dists)
      print(f"[CanonicalLight] frame light={k} (distance {srt[0]:.4f}, margin {srt[1]-srt[0]:.4f})"
            + ("" if k else " — already canonical"))
      if k == 0:
          return mosaic
      gains = self._canonExemplars[0]['means'] / self._canonExemplars[k]['means']
      maxval = 65535.0 if mosaic.dtype == np.uint16 else 255.0
      out = mosaic.astype(np.float32)
      out[0::2, 0::2] *= float(gains[0])
      out[0::2, 1::2] *= float(gains[1])
      out[1::2, 0::2] *= float(gains[2])
      out[1::2, 1::2] *= float(gains[3])
      return np.clip(out, 0, maxval).astype(mosaic.dtype)

   def onProcessNewImageSample(self,filepath):
           # Always start from a clean frame; we may restore JSON or apply carried points below
           self.cleanThisFrameMetaData()
           self._pf_next = None  # set below if this frame is prefetch-eligible

           #if (checkIfFileExists("%s.json"%filepath)):
           #    print("There are saved data that need to be restored here")
           #    self.restoreFromJSON("%s.json" % filepath)
           #jsonPath = self.folderStreamer.getJSON()
           # Make .png/.jpg compatible with legacy annotations saved as *.pnm.json
           #jsonPath = resolve_annotation_json_path(filepath, prefer_existing=True) or jsonPath

           print("onProcessNewImageSample (", filepath, ") ")
           json_exists = False

           jsonPath = self.folderStreamer.getJSON()
           print(" self.folderStreamer.getJSON() = ", jsonPath, " ")

           # 1) Trust the streamer's answer first (HTTP streamer downloads stem.json)
           if jsonPath is not None and checkIfFileExists(jsonPath):
               print("There are saved data that need to be restored here (", jsonPath, ")")
               self.restoreFromJSON(jsonPath)
               json_exists = True
           else:
               # 2) Fallback to resolver (local legacy compatibility)
               resolved = resolve_annotation_json_path(filepath, prefer_existing=True)
               if resolved is not None and checkIfFileExists(resolved):
                   jsonPath = resolved
                   print("There are saved data that need to be restored here (", jsonPath, ")")
                   self.restoreFromJSON(jsonPath)
                   json_exists = True
               else:
                   print("No annotations found for ", filepath, " / ", resolved)

           _ = json_exists

           
           """
           if hasattr(self, 'controlsData'):
                   frame_idx = self.scrollBar.GetValue()
                   if 0 <= frame_idx < len(self.controlsData):
                       self.updateControlsTab(self.controlsData[frame_idx],sample_number = frame_idx)
           """
           ui_idx = self.scrollBar.GetValue()
           stream_idx = self._stream_from_ui(ui_idx)

           if hasattr(self, 'controlsData'):
               if 0 <= stream_idx < len(self.controlsData):
                   self.updateControlsTab(self.controlsData[stream_idx], sample_number=stream_idx)


           #self.filehash = get_md5(filepath) 
           #img = wx.Image(self.filepath, wx.BITMAP_TYPE_ANY)
           #img = self.rescaleBitmap(img)
           #self.imageCtrl.SetBitmap(wx.Bitmap(img))

           # Render the polarization image for both view panels
           global combineChannels
           #imgCV  = cv2.imread(filepath) #,cv2.IMREAD_UNCHANGED  # wasteful: this decode is overwritten below for 4-ch polar PNGs; imgCV is derived from imgPNM instead
           imgPNM = cv2.imread(filepath, cv2.IMREAD_UNCHANGED) #This is to be used without changes by the classifier

           if (imgPNM is None):
                  print("Could not load ",filepath)
                  return


           # if we got a 4-channel PNG (p0,p45,p90,p135), repack to the original 2x2 mosaic
           if (imgPNM.ndim == 3) and (imgPNM.shape[2] == 4):
               p0   = imgPNM[:, :, 0]
               p45  = imgPNM[:, :, 1]
               p90  = imgPNM[:, :, 2]
               p135 = imgPNM[:, :, 3]
               print("Re-bayering .PNG file to transparently treat it as .PNM")
               imgPNM = repackPolarToMosaic(p0, p45, p90, p135)   # now 2D, as classifier expects
               if self.canonicalLightCheckbox.GetValue():
                   imgPNM = self._canonicalizeLighting(imgPNM, filepath)
               imgCV  = cv2.merge([imgPNM, imgPNM, imgPNM])       # keep existing visualization logic happy
           else:
               if (imgPNM.ndim == 2) and self.canonicalLightCheckbox.GetValue():
                   imgPNM = self._canonicalizeLighting(imgPNM, filepath)  # .pnm mosaic path
               # non-polar PNG (e.g. boot logo / 3-ch / gray): reproduce the old cv2.imread() 3-channel BGR
               imgCV  = imgPNM if (imgPNM.ndim == 3 and imgPNM.shape[2] == 3) else cv2.cvtColor(imgPNM, cv2.COLOR_GRAY2BGR)


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
           if (processingString=="PolarizationRGB1"):
               self.processingWay=0
           elif (processingString=="PolarizationRGB2"):
               self.processingWay=1
           elif (processingString=="PolarizationRGB3"):
               self.processingWay=2
           elif (processingString=="Polarization_0_degree"):
               self.processingWay=5
           elif (processingString=="Polarization_45_degree"):
               self.processingWay=6
           elif (processingString=="Polarization_90_degree"):
               self.processingWay=7
           elif (processingString=="Polarization_135_degree"):
               self.processingWay=8
           elif (processingString=="AoLP"):
               self.processingWay=9
           elif (processingString=="DoLP"):
               self.processingWay=10
           elif (processingString=="Intensity"):
               self.processingWay=11
           elif (processingString=="s0"):
               self.processingWay=12
           elif (processingString=="s1"):
               self.processingWay=13
           elif (processingString=="s2"):
               self.processingWay=14
           elif (processingString=="s3"):
               self.processingWay=15
           elif (processingString=="AoLP (light)"):
               self.processingWay=16
           elif (processingString=="AoLP (dark)"):
               self.processingWay=17
           elif (processingString=="DoP"):
               self.processingWay=18
           elif (processingString=="DoCP"):
               self.processingWay=19
           elif (processingString=="ToP"):
               self.processingWay=20
           elif (processingString=="CoP"):
               self.processingWay=21
           elif (processingString=="RetardationMag"):
               self.processingWay=22
           elif (processingString=="MaxMinAvgRGB"):
               self.processingWay=23
           elif (processingString=="Normals"):
               self.processingWay=24
           elif (processingString=="Sobel"):
               self.processingWay=3
           elif (processingString=="Visible"):
               self.processingWay=4
           

           if combineChannels:
              print("Image CV Combining all channels to one")
              # Prefetch fast path: when the classifier is off, the heavy renders below are a
              # pure function of (filepath, way, brightness, contrast) and may already be
              # computed by the background worker for this frame.
              fast_eligible = (not (useClassifier and not self.classifierDisabledCheckbox.GetValue())
                               and app.photoTxt.GetValue() != "default")
              cached = None
              if fast_eligible:
                  key_way = self.processingWay  # capture before the Sobel branch mutates it
                  cached = self._takePrefetch(filepath, key_way, self.brightness_offset, self.contrast_offset)
                  self._pf_next = (key_way, self.brightness_offset, self.contrast_offset)

              if cached is not None:
                  imgCV = cached['foreground']
              else:
                  if (self.processingWay==3):
                      imgCV = detect_sobel_edges(imgCV)
                      self.processingWay=0
                  imgCV = self.rescaleCVMAT(convertPolarCVMATToRGB(imgCV,way=self.processingWay,brightness=self.brightness_offset, contrast=self.contrast_offset))

              if app.photoTxt.GetValue() != "default": #<- Don't trigger classification in logo "default dataset" when application boots

                if useClassifier and not self.classifierDisabledCheckbox.GetValue(): #<- Only use classifier when classifier is on
                  self.AIAnnotations=None
                  if self.classifierTwoStage.GetValue() and getattr(self, 'EnsembleClassifierPnm', None) is None:
                     print("2-stage ensemble requested but no allclass_* models are loaded — using single classifier")
                     self.classifierTwoStage.SetValue(False)
                  if self.classifierTwoStage.GetValue():
                     print("Image classification done through 2-stage ensemble classifier")
                     self.EnsembleClassifierPnm.step = self.classifierTileSize.GetValue()
                     self.EnsembleClassifierPnm.maxProbabilityThreshold = float(self.classifierThreshold.GetValue() / 100.0) #parallel=True	Re-tiles the full image per model (selected-tile optimization is lost); Python GIL + shared CUDA queue limits real overlap.
                     imgRGBFromClassifier, occupancy, self.AIAnnotations = self.EnsembleClassifierPnm.forward(imgPNM, majorityVote=self.classifierMajorityVoting.GetValue(), parallel=False, multimodel=self.parallellTwoStage.GetValue())
                     imgRGBFromClassifier = self.rescaleCVMAT(convertRGBCVMATToRGB(imgRGBFromClassifier,brightness=self.brightness_offset, contrast=self.contrast_offset))
                     processed_img = imgRGBFromClassifier
                     self.leftViewImage = imgRGBFromClassifier
                     self.classifierInfo.SetLabel("2-stage: %0.2f Hz" % self.EnsembleClassifierPnm.hz)
                  else:
                     print("Image classification done through regular 1-stage classifier")
                     self.ClassifierPnm.step = self.classifierTileSize.GetValue()
                     self.ClassifierPnm.maxProbabilityThreshold = float(self.classifierThreshold.GetValue() / 100.0)
                     imgRGBFromClassifier,occupancy, self.AIAnnotations = self.ClassifierPnm.forward(imgPNM, majorityVote=self.classifierMajorityVoting.GetValue(), erosion_kernel=self.erodeKernelSize.GetValue(),erosion_threshold=self.erodeThreshold.GetValue())
                     imgRGBFromClassifier = self.rescaleCVMAT(convertRGBCVMATToRGB(imgRGBFromClassifier,brightness=self.brightness_offset, contrast=self.contrast_offset))
                     processed_img = imgRGBFromClassifier
                     self.leftViewImage = imgRGBFromClassifier
                     self.classifierInfo.SetLabel("1-stage: %0.2f Hz" % self.ClassifierPnm.hz)
                  #print(" self.AIAnnotations: ",self.AIAnnotations)
                  #self.AIAnnotations:  {'points': [(1424, 368), (1360, 400), (1392, 400), (1424, 400), (1360, 432), (1392, 432)], 'classes': ['class_NegativeDentClassA', 'class_NegativeDentClassA', 'class_NegativeDentClassA', 'class_NegativeDentClassA', 'class_NegativeDentClassA', 'class_NegativeDentClassA']}
              else:
                #If we didn't trigger then show the raw image as processed image
                processed_img                  = imgCV
                self.leftViewImage       = imgCV

              if useClassifier and not self.classifierDisabledCheckbox.GetValue(): #<- Only use classifier when classifier is on
                current_hz = (self.EnsembleClassifierPnm.hz
                              if self.classifierTwoStage.GetValue()
                              else self.ClassifierPnm.hz)
                # The classifier runs on the DEMOSAICED (half-res) image, so its
                # activation coords are half the user-click (full mosaic) coords.
                # Scale AI points x2 so the spatial matcher compares like with like.
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
                
              if (self.lightComboBox.GetValue()=="Unknown"): #If we don't have a light orientation set
               print("We don't know Light Direction")
               if (self.calcFocusLightCheckbox.GetValue()):   #If we are ok with guessing 
                 print("We will try to guess light direction")
                 self.lightComboBox.SetValue(determine_intensity_region(imgCV, threshold=0.1))

              if useClassifier and not self.classifierDisabledCheckbox.GetValue():
                  #processed_img = imgRGBFromClassifier
                  #self.leftViewImage = imgRGBFromClassifier
                  pass
              else:
                  if cached is not None:
                      classifierBaseImg = cached['processed']
                  else:
                      rgba = readPolarPNMToRGBALive(imgPNM)
                      classifierBaseImg = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
                      classifierBaseImg = self.rescaleCVMAT(convertRGBCVMATToRGB(classifierBaseImg, brightness=self.brightness_offset, contrast=self.contrast_offset))
                  processed_img                  = classifierBaseImg
                  self.leftViewImage       = classifierBaseImg
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


   def onCameraSettings(self, event):
        #Deactivated
        """
        dlg = CameraSettingsDialog(self.frame, title='Camera Settings')
        dlg.ShowModal()
        if (self.filepath!=""):
           self.onSave(event) #Save current pre-existing image..
        self.filepath = dlg.filename
        dlg.Destroy()
        self.onProcessNewImageSample(self.filepath)
        """
        
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

   def _batchComputeFocusLight(self, local_dir):
        """Compute Tenengrad focus + light direction for every frame and store them in each
        frame's JSON. Used by Finalize when live focus/light calculation was left disabled."""
        images = list(getattr(self.folderStreamer, "directoryList", None) or [])
        if not images and self.filepath and os.path.isfile(self.filepath):
            images = list_image_files(os.path.dirname(self.filepath))
        if not images:
            return 0
        prog = wx.ProgressDialog(
            "Finalize — focus & light", "Computing focus and light direction…",
            maximum=len(images), parent=self.frame,
            style=wx.PD_APP_MODAL | wx.PD_AUTO_HIDE | wx.PD_CAN_ABORT
                  | wx.PD_ELAPSED_TIME | wx.PD_REMAINING_TIME)
        updated = skipped = 0
        for i, img in enumerate(images):
            cont, _ = prog.Update(i, f"{i+1}/{len(images)} frames ({skipped} already done)")
            if not cont:
                break
            wx.GetApp().Yield(True)

            # Read the frame JSON first so we can skip frames that already have focus + light.
            jp = resolve_annotation_json_path(img, prefer_existing=True)
            if not jp or not checkIfFileExists(jp):
                jp = os.path.splitext(img)[0] + ".json"
            data = {}
            if checkIfFileExists(jp):
                try:
                    with open(jp) as f:
                        data = json.load(f)
                except Exception:
                    data = {}
            if data.get("tenengradFocusMeasure") and \
               data.get("lightDirection", "Unknown") not in ("", "Unknown"):
                skipped += 1
                continue  # already calculated — don't decode/recompute

            raw = cv2.imread(img, cv2.IMREAD_UNCHANGED)
            if raw is None:
                continue
            if raw.ndim == 3 and raw.shape[2] == 4:
                mosaic = repackPolarToMosaic(raw[:, :, 0], raw[:, :, 1], raw[:, :, 2], raw[:, :, 3])
                imgCV  = cv2.merge([mosaic, mosaic, mosaic])
            else:
                imgCV  = raw if (raw.ndim == 3 and raw.shape[2] == 3) else cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
            focus = float(tenengrad_focus_measure(imgCV))
            light = determine_intensity_region(imgCV, threshold=0.1)

            if not data:
                data = {"width": self.width, "height": self.height, "md5hash": "",
                        "regionClicks": [], "pointClicks": [], "pointClasses": [],
                        "pointSeverities": [], "pointSources": []}
            data["tenengradFocusMeasure"] = focus
            data["lightDirection"]        = light
            try:
                with open(jp, "w") as f:
                    json.dump(data, f, sort_keys=False)
                updated += 1
            except Exception as e:
                print("Finalize: focus/light write failed", jp, e)
        prog.Destroy()
        print(f"Finalize focus/light: {updated} computed, {skipped} already had values")
        return updated

   def _detectLeadingDarkFrames(self, local_dir):
        """Count the consecutive dark ('No Light') frames at the very start of the dataset —
        caused by the latency between disabling the light safety and the scene light actually
        operating — so Finalize can set startFrame past them. Scans in sorted order and stops
        at the first correctly-lit frame; a dark frame later in the dataset (a genuine light
        failure) is left alone. An unreadable/placeholder leading frame counts as dark too."""
        images = list(getattr(self.folderStreamer, "directoryList", None) or [])
        if not images and self.filepath and os.path.isfile(self.filepath):
            images = list_image_files(os.path.dirname(self.filepath))
        leading = 0
        for img in images:
            raw = cv2.imread(img, cv2.IMREAD_UNCHANGED)
            if raw is not None:
                if raw.ndim == 3 and raw.shape[2] == 4:
                    mosaic = repackPolarToMosaic(raw[:, :, 0], raw[:, :, 1], raw[:, :, 2], raw[:, :, 3])
                    imgCV  = cv2.merge([mosaic, mosaic, mosaic])
                else:
                    imgCV  = raw if (raw.ndim == 3 and raw.shape[2] == 3) else cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
                if determine_intensity_region(imgCV, threshold=0.1) != "No Light":
                    break  # first correctly-lit frame — stop
            leading += 1
        return leading

   def onFinalize(self, event):
        """Finalize the dataset: write/augment info.json with certification info, the
        accumulated annotation-effort statistics, and the dataset-wide defect/severity totals."""
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

        info_path = os.path.join(local_dir, "info.json")

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
            leading_dark = self._detectLeadingDarkFrames(local_dir)
            if leading_dark > 0:
                info["startFrame"] = leading_dark
                print(f"Finalize: detected {leading_dark} leading dark frame(s); "
                      f"setting startFrame={leading_dark}")

        # Commit this session's active time before reading the counters.
        self._recordInteraction()

        defect_counts, severity_counts, total_defects = self._datasetDefectTotals(local_dir)

        info["certified_by"]    = getpass.getuser()
        info["annotated_at"]    = datetime.now().strftime("%Y/%m/%d %H:%M")
        info["annotation_time"] = int(info.get("annotation_time", 0)) + int(round(self._stat_active_seconds))
        info["clicks"]          = int(info.get("clicks", 0)) + self._stat_clicks
        info["keystrokes"]      = int(info.get("keystrokes", 0)) + self._stat_keystrokes
        info["points_added"]    = int(info.get("points_added", 0)) + self._stat_points_added
        info["points_deleted"]  = int(info.get("points_deleted", 0)) + self._stat_points_deleted
        info["defect_counts"]   = defect_counts
        info["severity_counts"] = severity_counts
        info["total_defects"]   = total_defects

        try:
            with open(info_path, "w") as f:
                json.dump(info, f, indent=1)
        except Exception as e:
            wx.MessageBox(f"Failed to write {info_path}:\n{e}", "Finalize", wx.OK | wx.ICON_ERROR)
            return

        # Reset so a second Finalize this session doesn't double-count the same effort.
        self._resetSessionStats()
        self.populateMetaData(info_path)
        wx.MessageBox(
            f"Finalized {os.path.basename(local_dir)}.\n\n"
            f"Total defects: {total_defects}\n"
            f"Defect types: {defect_counts}\n"
            f"Severities: {severity_counts}\n"
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
        from datasetSelector import DatasetSelector
        dlg = DatasetSelector(local_base_path=self.local_base_path)
        if dlg.ShowModal() == wx.ID_OK:
            selectedDirectory = self.local_base_path + "/" + dlg.selectedDirectory 
            print("Selected Dataset:",  dlg.selectedDataset)
            print("Caching Directory:", dlg.selectedDirectory)
            # You can pass this to your HTTPFolderStreamer
            #self.onNewInputPath(dlg.selectedDataset)
            from HTTPStream import HTTPFolderStreamer 
            self.folderStreamer = HTTPFolderStreamer(provider=dlg.selectedProvider, dataset=dlg.selectedDataset, local_dir=selectedDirectory, retrieve_zip=dlg.replaceAnnotations)

            # Must be set BEFORE openDataset: the classifier trigger checks
            # photoTxt != "default" while processing the FIRST frame
            app.photoTxt.SetValue(dlg.selectedDirectory)

            self.openDataset(
                             base_dir=selectedDirectory,   # cache dir where info.json/controller.csv live
                             streamer=self.folderStreamer,
                             is_directory=True
                            )
            #self.onNewInputPath(selectedDirectory)
  
            #self.populateMetaData("%s/info.json" % selectedDirectory)
            """
            self.loadControlsCSV("%s/controller.csv" % selectedDirectory)
            self._loadSensorPlotsNewDataset(directory = "%s/tactile/" %  self.folderStreamer.local_dir)
            self.onNext(event)
            self.onPrevious(event)
            """
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

   def rescaleBitmap(self,img):
        NewW,NewH = self.rescaleAnything(img.GetWidth(),img.GetHeight())
        img = img.Scale(int(NewW),int(NewH))
        return img

   def rescaleCVMAT(self,img):
        NewW,NewH = self.rescaleAnything(img.shape[1],img.shape[0])
        return cv2.resize(img, dsize=(int(NewW),int(NewH)), interpolation=cv2.INTER_CUBIC)
 

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
            self.imageCtrl.SetBitmap(left_bmp)
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
        self.imageCtrl.SetBitmap(left_overlay)

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
        selected_index = self.pointList.GetSelection()
        #if selected_index != -1:
        #    wx.MessageBox(f"Selected Point: {self.points_of_interest[selected_index]}")
 
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
            del self.points_of_interest[selected_index]
            del self.points_classes[selected_index]
            del self.points_severities[selected_index]
            if selected_index < len(self.points_sources):
                del self.points_sources[selected_index]
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

        pts = list(data.get('pointClicks', []))
        cls = list(data.get('pointClasses', []))
        sev = list(data.get('pointSeverities', []))
        src = list(data.get('pointSources', []))

        # Normalize lengths
        if len(cls) < len(pts):
            cls.extend([options[0]] * (len(pts) - len(cls)))
        if len(sev) < len(pts):
            sev.extend([severities[0]] * (len(pts) - len(sev)))
        if len(src) < len(pts):
            src.extend(["manual"] * (len(pts) - len(src)))
        cls = cls[:len(pts)]
        sev = sev[:len(pts)]
        src = src[:len(pts)]

        self.points_of_interest = pts
        self.points_classes = cls
        self.points_severities = sev
        self.points_sources = src
        self._stat_points_added += len(pts)
        self.updatePointList()
        self.onNext(event)

   def onRemoveRegion(self, event):
        selected_index = self.regionList.GetSelection()
        if selected_index != -1:
            del self.regions_of_interest[selected_index]
            self.updateRegionList()

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
        self.points_of_interest.append((self.x * self.clickRatioX, self.y * self.clickRatioY))
        selected_option = self.defectComboBox.GetValue()
        self.points_classes.append(selected_option)
        selected_option = self.severityComboBox.GetValue()
        self.points_severities.append(selected_option)
        self.points_sources.append("manual")

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
       """Distance between the two measure points, reported in raw debayered pixels and mm."""
       (x0, y0), (x1, y1) = self.measurePoints
       # measurePoints are in full mosaic coords; debayered channels are half that resolution
       debayer_dist = (((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5) / 2.0

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

       # pixels-per-mm scales inversely with sensor distance (pinhole model)
       if cur_h > 0 and ref_h > 0:
           px_per_mm = px_per_mm_ref * (ref_h / cur_h)
       else:
           px_per_mm = px_per_mm_ref

       if px_per_mm > 0:
           mm = debayer_dist / px_per_mm
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
      dists = [((px - fx) ** 2 + (py - fy) ** 2) for (px, py) in self.points_of_interest]
      idx = min(range(len(dists)), key=dists.__getitem__)
      # guard against deleting a far-away point on an empty-space click (~1.5 tiles)
      if dists[idx] > (144 * self.clickRatioX) ** 2:
          print("Right-click: no point near (%d,%d)" % (int(fx), int(fy)))
          return
      print("Right-click removing point %d at (%d,%d)" %
            (idx, int(self.points_of_interest[idx][0]), int(self.points_of_interest[idx][1])))
      del self.points_of_interest[idx]
      del self.points_classes[idx]
      del self.points_severities[idx]
      if idx < len(self.points_sources):
          del self.points_sources[idx]
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
            #self.handleZoomIn()  # Call a zoom-in method or similar action
        else:
            print("Mouse wheel moved down")
            self.onNext(event)
            #self.handleZoomOut()  # Call a zoom-out method or similar action

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

   def onUploadAnnotationsOLD(self, event):
      print("Local Dir: ",self.folderStreamer.local_dir)
      zip_path = "./upload.zip"  # replace with your real file path
      zipCommand = "zip %s -b %s %s/color*.json "% (zip_path, self.local_base_path, self.folderStreamer.local_dir) 
      print("Zip command : ",zipCommand)
      os.system(zipCommand)
      dlg = UploadDialog(self.frame, zip_path, self.folderStreamer.local_dir)
      dlg.ShowModal()
      dlg.Destroy()
      os.system("rm upload.zip")

   def onUploadAnnotations(self, event):
    print("Local Dir: ", self.folderStreamer.local_dir)


    base_dir = self.local_base_path                 # e.g. /media/ammar/games2/Datasets/Magician
    zip_path = os.path.join(base_dir, "upload.zip")
    rel_dir  = os.path.basename(self.folderStreamer.local_dir.rstrip("/"))
    # rel_dir should be "AltinayKapoDefect"

    # zip APPENDS to an existing archive — start fresh, otherwise previously-uploaded
    # datasets accumulate in the same upload.zip.
    try:
        if os.path.isfile(zip_path):
            os.remove(zip_path)
    except Exception as e:
        print("Could not remove stale zip:", zip_path, e)

    # Include the per-frame annotation JSONs AND the (finalized) info.json for this dataset only.
    zipCommand = (
        f'cd "{base_dir}" && '
        f'zip "{zip_path}" -b "{base_dir}" "{rel_dir}"/color*.json "{rel_dir}"/info.json'
    )

    print("Zip command : ", zipCommand)
    os.system(zipCommand)

    dlg = UploadDialog(self.frame, zip_path, self.folderStreamer.local_dir)
    dlg.ShowModal()
    dlg.Destroy()
    os.system(f'rm -f "{zip_path}"')

   def onRunBatch(self, event):
        dlg = BatchProcessDialog(self.frame, self.folderStreamer)
        dlg.ShowModal()
        dlg.Destroy()

   def onOpenMagnifierOLD(self, event):
     """Open a magnifier window."""
     if hasattr(self, 'magnifier') and self.magnifier:
        self.magnifier.Raise()
        return

     self.magnifier = MagnifierFrame(self.frame)
     self.magnifier.Show()

     # Pass the current image (wx.Image) to magnifier
     bmp = self.imageCtrl.GetBitmap()
     if bmp.IsOk():
        img = bmp.ConvertToImage()
        self.magnifier.setImage(img)

     # Bind mouse motion to update magnifier for both images
     self.imageCtrl.Bind(wx.EVT_MOTION, self.onMouseMoveMagnifier)
     self.secondaryImageCtrl.Bind(wx.EVT_MOTION, self.onMouseMoveMagnifier)

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
       os.system("python3 magician_grabber_frontend.py %s" % self.local_base_path) #<- Lazy

   def onCreateDataset(self,event):
       os.system("python3 datasetCreator.py %s" % self.local_base_path) #<- Lazy

   def onTileExplorer(self,event):
       os.system("python3 tileExplorer.py %s" % self.local_base_path) #<- Lazy

   def onStreamer(self,event):
       try:
          selectedDirectory = self.folderStreamer.local_dir
          print("Streamer set directory : ",selectedDirectory)
          os.system("python3 streamDataset.py %s" % selectedDirectory) #<- Lazy
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
           self.stats.run_info = (f"threshold={thr:.2f}  {step_info}  "
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
    """Render every frame (left + right side-by-side) to JPEGs then encode with ffmpeg."""
    import subprocess, glob, threading, tempfile

    if not self.filePathIsDirectory:
        wx.MessageBox("Please open a directory first.", "Make Video", wx.OK | wx.ICON_INFORMATION)
        return

    total = self.folderStreamer.max()
    if total == 0:
        wx.MessageBox("No frames found.", "Make Video", wx.OK | wx.ICON_WARNING)
        return

    # Ask for output path
    with wx.FileDialog(self.frame, "Save video as", wildcard="MP4 files (*.mp4)|*.mp4",
                       style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as fd:
        if fd.ShowModal() != wx.ID_OK:
            return
        out_path = fd.GetPath()
    if out_path.endswith(".mp4"):
        out_path = out_path[:-4]

    # Work in a temp directory so frame JPEGs don't clutter the dataset
    tmp_dir = tempfile.mkdtemp(prefix="wxAnnotator_video_")

    dlg = wx.ProgressDialog(
        "Making Video", "Rendering frames…",
        maximum=total, parent=self.frame,
        style=wx.PD_APP_MODAL | wx.PD_AUTO_HIDE | wx.PD_CAN_ABORT | wx.PD_ELAPSED_TIME)

    def _worker():
        import cv2
        import numpy as np

        saved_ui = self.scrollBar.GetValue()
        aborted  = False
        done_evt = threading.Event()
        cont_box = [True]

        def _bmp_to_cv(bmp):
            if not (bmp and bmp.IsOk()):
                return None
            img = bmp.ConvertToImage()
            arr = np.frombuffer(img.GetData(), dtype=np.uint8)
            arr = arr.reshape(img.GetHeight(), img.GetWidth(), 3)
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

        for i in range(total):
            done_evt.clear()

            def _gui_step(fi=i):
                # Update progress; check for user abort
                cont_box[0], _ = dlg.Update(fi, f"Frame {fi+1}/{total}")

                # Advance frame and force redraw
                self.gotoFrameUI(fi)
                self.panel.Update()
                wx.GetApp().Yield(True)

                left_cv  = _bmp_to_cv(self.imageCtrl.GetBitmap())
                right_cv = _bmp_to_cv(self.secondaryImageCtrl.GetBitmap())

                if left_cv is not None and right_cv is not None:
                    h = max(left_cv.shape[0], right_cv.shape[0])
                    def _pad(img, th):
                        ph = th - img.shape[0]
                        return np.pad(img, ((0, ph), (0, 0), (0, 0))) if ph > 0 else img
                    frame = np.concatenate([_pad(left_cv, h), _pad(right_cv, h)], axis=1)
                elif left_cv is not None:
                    frame = left_cv
                elif right_cv is not None:
                    frame = right_cv
                else:
                    done_evt.set()
                    return

                fname = os.path.join(tmp_dir, f"colorFrame_0_{fi:05d}.jpg")
                cv2.imwrite(fname, frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                done_evt.set()

            wx.CallAfter(_gui_step)
            done_evt.wait(timeout=60)   # wait for GUI thread to finish this frame

            if not cont_box[0]:
                aborted = True
                break

        def _finalize():
            dlg.Destroy()

            if aborted:
                wx.MessageBox("Rendering aborted.", "Make Video", wx.OK | wx.ICON_INFORMATION)
                import shutil
                shutil.rmtree(tmp_dir, ignore_errors=True)
                return

            # Restore original frame
            self.gotoFrameUI(saved_ui)

            # Encode with ffmpeg
            enc_dlg = wx.ProgressDialog("Encoding", "Running ffmpeg…", maximum=1,
                                         parent=self.frame, style=wx.PD_APP_MODAL)
            enc_dlg.Pulse()

            ret = subprocess.run([
                "ffmpeg", "-nostdin", "-framerate", "25",
                "-i", os.path.join(tmp_dir, "colorFrame_0_%05d.jpg"),
                "-vf", "scale=-2:720", "-y", "-r", str(self.metadata.get("frameRate",23)),
                "-pix_fmt", "yuv420p", "-threads", "8",
                f"{out_path}_lastRun3DHiRes.mp4",
            ], check=False)

            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)
            enc_dlg.Destroy()

            if ret.returncode == 0:
                wx.MessageBox(f"Video saved:\n{out_path}_lastRun3DHiRes.mp4",
                              "Make Video", wx.OK | wx.ICON_INFORMATION)
            else:
                wx.MessageBox(
                    f"ffmpeg exited with code {ret.returncode}.\n"
                    "Is ffmpeg installed and on PATH?",
                    "Make Video", wx.OK | wx.ICON_ERROR)

        wx.CallAfter(_finalize)

    threading.Thread(target=_worker, daemon=True).start()

   def onMouseMoveMagnifierOLD(self, event):
     if hasattr(self, 'magnifier') and self.magnifier and self.magnifier.IsShown():
        x, y = event.GetX(), event.GetY()
        self.magnifier.updateMagnifier(x, y)
     event.Skip()

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
               if ("positive" in loadDataset):
                   app.defectComboBox.SetValue("Positive Dent")
               if ("negative" in loadDataset):
                   app.defectComboBox.SetValue("Negative Dent")
               if ("class-a" in loadDataset):
                   app.severityComboBox.SetValue("Class A")
               if ("class-b" in loadDataset):
                   app.severityComboBox.SetValue("Class B")
               if ("class-c" in loadDataset):
                   app.severityComboBox.SetValue("Class C")

               app.photoTxt.SetValue(loadDataset)
               app.onNewInputPath(loadDataset)
               inputIsSet = True
 

    if not inputIsSet:
               app.photoTxt.SetValue("default")
               app.onNewInputPath("default")


    app.MainLoop()

