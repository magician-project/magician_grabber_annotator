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

version         = "0.64"
useClassifier   = True #<- Master switch classifier off if you have hw/sw limitations
benchmark       = False #<- Set to True to run a forward-pass timing test on each model at startup
combineChannels = True
options         = ["Unknown", "Material Defect", "Positive Dent", "Negative Dent", "Deformation", "Seal", "Welding", "Suspicious", "Clean", "RLClean"]
severities      = ["Class A","Class B","Class C"]
directions      = ["Unknown","Bottom Left","Top Left","Top","Top Right", "Bottom Right", "Bottom"]
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

from folderStream import FolderStreamer
from classifierGrading import AnnotationCorrelationStats
from downloadAllFrames import BatchProcessDialog
from magnifier import MagnifierFrame
from modelUpdater import ModelUpdaterDialog
from rlAnnotator import RLAnnotatorDialog

# AutoAnnotator needs gradio_client (optional). Import lazily-safe so the GUI still
# launches if the dependency / servers are unavailable; onAuto reports the error.
try:
    from AutoAnnotator import AutoAnnotator
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


        screen = wx.Display(0)  # Get the primary display
        screen_width, screen_height = screen.GetGeometry().GetSize()
        #screen_width = 1900
        print("Screen Resolution: {}x{}".format(screen_width, screen_height))

        if (screen_width>=1920):
          self.PhotoMaxSizeWidth   = 800
          self.PhotoMaxSizeHeight  = 620
        else:
          self.PhotoMaxSizeWidth   = 475 #<- Small monitor
          self.PhotoMaxSizeHeight  = 400
         
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
           self.EnsembleClassifierPnm = EnsembleClassifierPnm(
                                                            #("../magician_vision_classifier/binary_small_cnn.pth","../magician_vision_classifier/binary_small_cnn.json")
                                                            initial_model_cfg = ("../magician_vision_classifier/allclass_verysmall_cnn.pth","../magician_vision_classifier/allclass_verysmall_cnn.json"),
                                                            model_cfg_list=self._scan_allclass_models(classifier_relative_directory),
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

    # Brightness controls
    self.minusButton = wx.Button(self.panel, label="-", size=(40, 30))
    self.minusButton.Bind(wx.EVT_BUTTON, self.decrease_brightness)
    self.brightnessLabel = wx.StaticText(self.panel, label="Br.")
    self.brightnessText = wx.TextCtrl(self.panel, value="0", size=(50, 30), style=wx.TE_CENTER)
    self.brightnessText.Bind(wx.EVT_TEXT, self.on_brightness_change)
    self.plusButton = wx.Button(self.panel, label="+", size=(40, 30))
    self.plusButton.Bind(wx.EVT_BUTTON, self.increase_brightness)

    # Contrast controls
    self.minusContrastButton = wx.Button(self.panel, label="-", size=(40, 30))
    self.minusContrastButton.Bind(wx.EVT_BUTTON, self.decrease_contrast)
    self.contrastLabel = wx.StaticText(self.panel, label="Co.")
    self.contrastText = wx.TextCtrl(self.panel, value="0", size=(50, 30), style=wx.TE_CENTER)
    self.contrastText.Bind(wx.EVT_TEXT, self.on_contrast_change)
    self.plusContrastButton = wx.Button(self.panel, label="+", size=(40, 30))
    self.plusContrastButton.Bind(wx.EVT_BUTTON, self.increase_contrast)

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

    self.underImage.Add(self.minusButton, 0, wx.ALL, 5)
    self.underImage.Add(self.brightnessLabel, 0, wx.ALL, 5)
    self.underImage.Add(self.brightnessText, 0, wx.ALL, 5)
    self.underImage.Add(self.plusButton, 0, wx.ALL, 5)

    self.underImage.Add(self.minusContrastButton, 0, wx.ALL, 5)
    self.underImage.Add(self.contrastLabel, 0, wx.ALL, 5)
    self.underImage.Add(self.contrastText, 0, wx.ALL, 5)
    self.underImage.Add(self.plusContrastButton, 0, wx.ALL, 5)

    self.mainSizer.Add(self.underImage, 0, wx.ALL | wx.EXPAND, 5)

    # Finalize
    self.panel.SetSizer(self.mainSizer)
    self.mainSizer.Fit(self.frame)

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

    # Image Regions
    self.regionLabel = wx.StaticText(parent, label="Image Regions")
    regionListSize = wx.Size(-1, 40)
    self.regionList = wx.ListBox(parent, size=regionListSize, choices=[], style=wx.LB_SINGLE)
    self.regionList.Bind(wx.EVT_LISTBOX, self.onSelectPoint)
    self.removeRegionBtn = wx.Button(parent, label='Remove Selected Point')
    self.removeRegionBtn.Bind(wx.EVT_BUTTON, self.onRemovePoint)

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
    self.pointList = wx.ListBox(parent, choices=[], style=wx.LB_SINGLE)
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
    self.saveBtn = wx.Button(parent, label='Save')
    self.saveBtn.Bind(wx.EVT_BUTTON, self.onSave)
    self.deleteMetadataBtn = wx.Button(parent, label='Delete')
    self.deleteMetadataBtn.Bind(wx.EVT_BUTTON, self.ondeleteMetadata)

    comboButtons = wx.BoxSizer(wx.HORIZONTAL)
    comboButtons.Add(self.autoBtn, 0, wx.ALL, 5)
    comboButtons.Add(self.fullAutoBtn, 0, wx.ALL, 5)
    comboButtons.Add(self.saveBtn, 0, wx.ALL, 5)
    comboButtons.Add(self.deleteMetadataBtn, 0, wx.ALL, 5)

    # Checkboxes (up to Guess lighting direction)
    self.incrementFrameAfterAnAdditionCheckbox = wx.CheckBox(parent, label="Increment frame after defect annotation")        
    self.incrementFrameAfterAnAddition=True
    self.incrementFrameAfterAnAdditionCheckbox.SetValue(self.incrementFrameAfterAnAddition)
    self.calcFocusLightCheckbox = wx.CheckBox(parent, label="Calculate Focus & Light Direction")
    self.calcFocusLightCheckbox.SetValue(False)

    # Layout stack for Annotator tab
    s.Add(self.datasetLabel, 0, wx.ALL | wx.EXPAND, 5)
    s.Add(self.datasetList, 0, wx.ALL | wx.EXPAND, 5)

    s.Add(self.regionLabel, 0, wx.ALL | wx.EXPAND, 5)
    s.Add(self.regionList, 0, wx.ALL | wx.EXPAND, 5)
    s.Add(self.removeRegionBtn, 0, wx.ALL, 5)

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

    s.Add(comboButtons, 0, wx.ALL, 5)
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
    self.classifierThreshold = wx.Slider(parent, value=0, minValue=0, maxValue=100, style=wx.SL_HORIZONTAL)
    self.classifierThresholdValue = wx.StaticText(parent, label="0.00")
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
    erodeKernelLbl        = wx.StaticText(parent, label="Erode Kernel Size")
    self.erodeKernelSize  = wx.Slider(parent, value=0, minValue=0, maxValue=8, style=wx.SL_HORIZONTAL)
    self.erodeKernelValue = wx.StaticText(parent, label="0")
    def _on_erodkrnthr(evt):
        self.erodeKernelValue.SetLabel(f"{self.erodeKernelSize.GetValue()}")
        evt.Skip()
    self.erodeKernelSize.Bind(wx.EVT_SLIDER, _on_erodkrnthr)
    erodeKernelRow.Add(erodeKernelLbl, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 8)
    erodeKernelRow.Add(self.erodeKernelSize, 1, wx.RIGHT, 8)
    erodeKernelRow.Add(self.erodeKernelValue, 0, wx.ALIGN_CENTER_VERTICAL)


    # --- Erode Threshold Value ---
    erodeThresholdRow        = wx.BoxSizer(wx.HORIZONTAL)
    erodeThresholdLbl        = wx.StaticText(parent, label="Erode Min. Threshold to Keep")
    self.erodeThreshold      = wx.Slider(parent, value=0, minValue=0, maxValue=8, style=wx.SL_HORIZONTAL)
    self.erodeThresholdValue = wx.StaticText(parent, label="0")
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

   def onAuto(self, event):
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

      annotated = marks = skipped = failed = 0
      for i, img in enumerate(images):
          cont, _ = prog.Update(
              i, f"{i+1}/{len(images)} — {marks} mark(s) in {annotated} frame(s), {skipped} skipped")
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
              dets = self.autoAnnotator.detect(raw, polarity=polarity)
          except Exception as e:
              prog.Destroy()
              wx.MessageBox(f"Detection failed on:\n{img}\n\n{e}",
                            "Full Auto", wx.OK | wx.ICON_ERROR)
              return
          if not dets:
              continue

          data = {
              "width":  self.width,
              "height": self.height,
              "md5hash": "",
              "regionClicks": [],
              "pointClicks":     [[x, y] for x, y, _a in dets],
              "pointClasses":    [defect] * len(dets),
              "pointSeverities": [severity] * len(dets),
              "pointSources":    ["auto"] * len(dets),
          }
          try:
              with open(json_path, "w") as f:
                  json.dump(data, f, sort_keys=False)
              annotated += 1
              marks     += len(dets)
          except Exception as e:
              print(f"[Full Auto] Failed writing {json_path}: {e}")
              failed += 1

      prog.Destroy()
      wx.MessageBox(
          f"Full Auto complete.\n\n"
          f"Annotated {marks} mark(s) across {annotated} frame(s).\n"
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



   def sensibleDefaults(self,loadDatasetCase):
               loadDataset = loadDatasetCase.lower()
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

   def openDataset(self, base_dir, streamer, is_directory=True):
    """
    base_dir: local path where info.json/controller.csv/tactile live
              (for network: the cache dir, e.g. selectedDirectory)
    streamer: FolderStreamer or HTTPFolderStreamer
    is_directory: whether we should run directory-mode behaviors
    """
    self.folderStreamer = streamer
    self.filePathIsDirectory = is_directory
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

    # Jump to first visible frame in range
    self.gotoFrameUI(0)

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
    self.onProcessNewImageSample(self.filepath)
    self.updateMinMaxSlider()
    #self.onView() # redundant: onProcessNewImageSample already calls onView()



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
           imgCV  = cv2.merge([imgPNM, imgPNM, imgPNM])
       else:
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
               imgCV  = cv2.merge([imgPNM, imgPNM, imgPNM])       # keep existing visualization logic happy
           else:
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
                self.stats.update(
                                 frame_id=self.filepath,
                                 user_ann={
                                           "points":     self.points_of_interest,
                                           "classes":    self.points_classes,
                                           "severities": self.points_severities,
                                          },
                                 ai_ann=self.AIAnnotations,
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

        info_path = os.path.join(local_dir, "info.json")

        # Preserve existing (camera) fields; tolerate a missing or malformed info.json.
        info = {}
        if os.path.isfile(info_path):
            try:
                with open(info_path) as f:
                    info = json.load(f)
            except Exception as e:
                print(f"Finalize: existing info.json unreadable ({e}); starting fresh.")

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
            f"(info.json updated with effort statistics.)",
            "Finalize", wx.OK | wx.ICON_INFORMATION)

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
            app.photoTxt.SetValue(dlg.selectedDirectory)
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


   def increase_brightness(self, event):
        if self.brightness_offset < 5:
            self.brightness_offset += 1
            self.brightnessText.SetValue(str(self.brightness_offset))
        self.onRedrawData(event) 

   def decrease_brightness(self, event):
        if self.brightness_offset > 0:
            self.brightness_offset -= 1
            self.brightnessText.SetValue(str(self.brightness_offset))
        self.onRedrawData(event)

   def on_brightness_change(self, event):
        value = self.brightnessText.GetValue()
        if value.isdigit():  # Check if input is a number
            brightness_offset = int(value)
            if 0 <= brightness_offset <= 5:
                self.brightness_offset = brightness_offset
            else:
                self.brightnessText.SetValue(str(self.brightness_offset))  # Reset to last valid value
        else:
            self.brightnessText.SetValue(str(self.brightness_offset))  # Reset if input is invalid
        self.onRedrawData(event) 

   def increase_contrast(self, event):
        if self.contrast_offset < 5:
            self.contrast_offset += 1
            self.contrastText.SetValue(str(self.contrast_offset))
        self.onRedrawData(event) 

   def decrease_contrast(self, event):
        if self.contrast_offset > 0:
            self.contrast_offset -= 1
            self.contrastText.SetValue(str(self.contrast_offset))
        self.onRedrawData(event)

   def on_contrast_change(self, event):
        value = self.contrastText.GetValue()
        if value.isdigit():  # Check if input is a number
            contrast_offset = int(value)
            if 0 <= contrast_offset <= 5:
                self.contrast_offset = contrast_offset
            else:
                self.contrastText.SetValue(str(self.contrast_offset))  # Reset to last valid value
        else:
            self.contrastText.SetValue(str(self.contrast_offset))  # Reset if input is invalid
        self.onRedrawData(event) 

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
      if self.photoTxt.GetValue() != "default": #<- Don't trigger in logo on boot
        self._stat_clicks += 1
        self._recordInteraction()
        self.x, self.y = event.GetPosition()
        self.regions_of_interest.append((self.x * self.clickRatioX, self.y * self.clickRatioY))
        self.updateRegionList()
        print("Click ",self.x,",",self.y)
        print("Rescaled Click ",int(self.x*self.clickRatioX),",",int(self.y*self.clickRatioY))
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

