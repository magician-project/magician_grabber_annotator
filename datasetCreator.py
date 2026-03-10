#!/usr/bin/python3

""" 
Author : "Ammar Qammaz"
Copyright : "2022 Foundation of Research and Technology, Computer Science Department Greece, See license.txt"
License : "FORTH" 
"""

"""
wxPython GUI to select dataset directories and dump them into a Keras-style dataset layout.

Features:
 - Lists immediate subdirectories of a starting directory (from command line or default)
 - Shows whether the directory contains any .json files (marked in the list)
 - Allows the user to check any combination of datasets to process
 - Lets the user pick a target output directory
 - Runs the dump procedure (uses readData.loadMoreImages and helper checks) in a worker thread
 - Shows overall progress, live cleanThreshold, and per-class occurrence updates
 - At finish offers to zip the created output directory

Notes:
 - This script assumes the `readData.py` module (with functions loadMoreImages, checkIfFileExists,
   checkIfPathExists, check_threshold, check_variation) is available on PYTHONPATH.
 - It also re-implements `dump_dataset_to_keras_data_loader` from your template (slightly adapted)

Requirements: wxPython (pip install wxPython)
"""

import wx
import wx.lib.scrolledpanel as scrolled
import os
import gc
import sys
import csv
import threading
import time
import shutil
import cv2
import random
import re


import json


#from memory_profiler import profile
def find_annotation_for_image(image_path: str) -> str | None:
    """Find an existing annotation file for an image, using readData.resolve_annotation_json_path."""
    if resolve_annotation_json_path is None:
        return None

    ann = resolve_annotation_json_path(image_path, prefer_existing=True)
    return ann if (ann is not None and os.path.exists(ann)) else None


def ensure_annotation_sidecar(image_path: str) -> str | None:
    """
    Ensure loadImage(...) can find annotations at '<image_path>.json'.
    If annotations exist under a legacy name, create '<image_path>.json' as a symlink (or copy).
    """
    if resolve_annotation_json_path is None:
        return None

    wanted = f"{image_path}.json"  # what loadImage expects (new style)
    if os.path.exists(wanted):
        return wanted

    existing = resolve_annotation_json_path(image_path, prefer_existing=True)
    if existing is None or not os.path.exists(existing):
        return None

    # If existing already equals wanted we're done
    if os.path.abspath(existing) == os.path.abspath(wanted):
        return wanted

    try:
        # Prefer symlink if possible, fallback to copy.
        try:
            os.symlink(os.path.abspath(existing), wanted)
        except Exception:
            shutil.copy2(existing, wanted)
        return wanted
    except Exception as e:
        print(f"Warning: could not create compat sidecar json for {image_path}: {e}")
        # Fallback: return what we *do* have
        return existing


# --- Import external utilities assumed to exist in readData.py ---
try:
    from readData import loadImageAndJSON, checkIfFileExists, checkIfPathExists, check_threshold, check_variation, resolve_annotation_json_path
except Exception as e:
    print("Could not import readData utilities. Make sure readData.py exists and is importable.\n", e)
    # We'll still continue; the UI will warn the user if the import failed.
    loadImageAndJSON = None
    checkIfFileExists = lambda p: os.path.isfile(p)
    checkIfPathExists = lambda p: os.path.exists(p)
    def check_threshold(tile, t):
        return True
    def check_variation(tile, v):
        return True
# ------------------------------------------------------------------
# Reuse helper functions from template
# ------------------------------------------------------------------
def retrieve_list_of_potential_directories(start_dir):
    entries = list()
    for d in os.listdir(start_dir):
        full_path = os.path.join(start_dir, d)
        if os.path.isdir(full_path):
           if checkIfFileExists(os.path.join(full_path, "info.json")):
               entries.append(full_path)

        #entries = [os.path.join(self.start_dir, d) for d in os.listdir(self.start_dir) if os.path.isdir(os.path.join(self.start_dir, d))]
    return entries
# ------------------------------------------------------------------
def isTileWorthAnNN(tile):
    doDump = True
    if not check_threshold(tile, 20):
        doDump = False
    if not check_variation(tile, 30):
        doDump = False
    return doDump
# ------------------------------------------------------------------
def sum_non_clean_entries(dictionary):
    total = 0
    for key, value in dictionary.items():
        if key != 'clean':
            total += value
    return total
# ------------------------------------------------------------------
def max_non_clean_entry(dictionary):
    max_value = float('-inf')
    for key, value in dictionary.items():
        if key != 'clean' and value > max_value:
            max_value = value
    return max_value if max_value != float('-inf') else None
# ------------------------------------------------------------------
def count_files_with_extension(path, extension):
    """
    Counts how many files in the given folder have the specified extension.
    
    Args:
        path (str): The directory path to search.
        extension (str): The file extension to look for (e.g., '.txt' or 'txt').
    
    Returns:
        int: Number of files matching the given extension.
    """
    # Normalize extension (ensure it starts with a dot)
    if not extension.startswith('.'):
        extension = '.' + extension

    # Count matching files
    count = sum(
        1 for filename in os.listdir(path)
        if os.path.isfile(os.path.join(path, filename)) and filename.lower().endswith(extension.lower())
    )
    return count
# ------------------------------------------------------------------
def add_png_comment(png_path: str, comment):
    """
    Adds or updates PNG metadata.
    If comment is a dict, it is stored as JSON in the Comment field.
    Otherwise it is stored as plain text.
    """
    from PIL import Image, PngImagePlugin
    import json

    img = Image.open(png_path)

    meta = PngImagePlugin.PngInfo()
    for k, v in img.info.items():
        meta.add_text(k, str(v))

    if isinstance(comment, dict):
        meta.add_text("Comment", json.dumps(comment))
    else:
        meta.add_text("Comment", str(comment))

    img.save(png_path, "PNG", pnginfo=meta)

#@profile
def dump_dataset_to_keras_data_loader(tiles, tile_classes, occurances, ratio_clean, outputDirectory, tile_info=None, tiles_annotated_by_ai=0, progress_callback=None):
    """
    tiles, tile_classes: lists aligned
    occurances: dict to update
    cleanThreshold: float
    outputDirectory: str
    progress_callback: function(status_dict) called periodically with dict containing keys:
       'sample_id', 'total_samples', 'occurances', 'cleanThreshold'
    Returns updated (occurances, cleanThreshold)
    """

    #Maintain labels and set clean labels where there is no title
    for tileID in range(len(tiles)):
        thisClass = tile_classes[tileID]
        if (thisClass == ""):
            thisClass = "clean"
        if thisClass in occurances:
            pass
        else:
            occurances[thisClass] = 1

    total = len(tiles)
    for tileID in range(len(tiles)):
 
        # Compute current non-clean sum and clean count
        non_clean_sum = sum_non_clean_entries(occurances)
        clean_count   = occurances.get("clean", 0)

        #Do not count tiles annotated by AI as clean
        if (clean_count>tiles_annotated_by_ai):
            clean_count-=tiles_annotated_by_ai
        else:
            clean_count=0

        target_clean_to_non_clean_ratio = ratio_clean  # 1.0 = target 1:1  of non-clean

        # Auto-balance clean samples
        # Determine probability to skip this clean tile
        skip_clean_prob = 0.0
        if thisClass == "clean":
            if non_clean_sum == 0:
                skip_clean_prob = 1.0
            else:
                max_clean_allowed = target_clean_to_non_clean_ratio * non_clean_sum
                if max_clean_allowed > 0:
                    ratio = clean_count / max_clean_allowed
                    skip_clean_prob = min(1.0, max(0.0, ratio))
 
        print("Tile ",tileID,"/",len(tiles)," | Clean ",clean_count," / Non-Clean ",non_clean_sum," / AI-Ann ",tiles_annotated_by_ai," | ",len(tiles)," tiles |  Accept Clean Prob % 0.2f      " % (1.0-skip_clean_prob),end="\r")

        thisClass = tile_classes[tileID]
        if (thisClass == ""):
            thisClass = "clean"
        elif (thisClass == "Clean"):
            thisClass = "clean"

        doDump = True
        if (thisClass == "clean"):
            if random.random() < skip_clean_prob:
                doDump = False

        if doDump:
            thisClassNoSpace = thisClass.replace(" ", "")
            class_dir = os.path.join(outputDirectory, f"class_{thisClassNoSpace}")
            if not checkIfPathExists(class_dir):
                os.makedirs(class_dir, exist_ok=True)

            targetPath = os.path.join(class_dir, f"{thisClassNoSpace}_image_{occurances[thisClass]}.png")
            # write image
            cv2.imwrite(targetPath, tiles[tileID])
            if tile_info is not None:
                add_png_comment(targetPath,tile_info[tileID])
                #To see this comment from linux shell: 
                #      identify -verbose path/to/data/NegativeDent_image_XXXX.png | grep -A1 "Comment"
            occurances[thisClass] = occurances.get(thisClass, 1) + 1

        # progress callback
        if progress_callback is not None:
            progress_callback({
                'sample_id': tileID + 1,
                'total_samples': total,
                'occurances': dict(occurances),
                'cleanThreshold': ratio_clean,
                'current_class': thisClass,
            })

    #print("Clean Threshold", cleanThreshold)
    return occurances, ratio_clean

# ------------------------------------------------------------------
# Worker thread that actually processes selected directories
# ------------------------------------------------------------------
class ProcessorThread(threading.Thread):
    def __init__(self, 
                 directories, 
                 target_dir,
                 ui_callbacks, 
                 ratio_clean=1.0,
                 threshold=0,
                 border=0,
                 step=32, 
                 tile_size=64,
                 ignoreSamplesWithNoMetadata=True,
                 includeTilesNotAnnotated=False,
                 includeTilesAnnotatedByAI=True,
                 use_severity=False,
                 use_clean_class=True):
        super().__init__()
        self.directories = directories
        self.target_dir = target_dir
        self.ui_callbacks = ui_callbacks  # dict of functions for UI updates
        self.step = step
        self.tile_size = tile_size
        self.ratio_clean = ratio_clean
        self.border = border
        self.threshold = threshold
        self.use_severity = use_severity
        self.use_clean_class = use_clean_class
        self._stop = False
        self.ignoreSamplesWithNoMetadata = ignoreSamplesWithNoMetadata
        self.includeTilesNotAnnotated    = includeTilesNotAnnotated
        self.includeTilesAnnotatedByAI   = includeTilesAnnotatedByAI
        self.total_tiles_annotated_by_ai = 0
        self.controlsData = []

    def readControllerCSV(self,pathToCSV):
        self.controlsData = []
        try:
          with open(pathToCSV, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            self.controlsData = list(reader)
        except Exception:
            print("Failed opening CSV file ",pathToCSV)
            pass

    def getControllerCSVRowData(self,frameID):
      if frameID is None:
         return None
      try:
        data_row = self.controlsData[frameID]
        print("Controller data for frame #",frameID,"/",len(self.controlsData)," : ",data_row)


        #DistanceAverage calculation
        #==============================================================
        distances = []
        for key in ["Distance1", "Distance2", "Distance3"]:
            val = data_row.get(key)
            try:
                distances.append(float(val))
            except (TypeError, ValueError):
                # ignore H, F, None, missing, etc.
                pass
        #==============================================================
        if distances:
            data_row["DistanceAverage"] = sum(distances) / len(distances)
        #else:
        #    data_row["DistanceAverage"] = None
        #==============================================================

        return data_row
      except Exception as e:
            print(f"Failed {e} getControllerCSVRowData for frame #",frameID,"         ")
            pass
      return None

    def get_frame_number_from_path(self,path): # path is something like /media/ammar/games2/Datasets/Magician/AltinayWelding/colorFrame_0_00047.png
        filename = os.path.basename(path)          # colorFrame_0_00047.png
        if ("colorFrame" in filename):
          number = filename.split('_')[-1].split('.')[0]  # 00047
          return int(number)
        return None
  
    def process_a_file(self, file_index, file_path, metaData, occurances):
        print("File : ", file_path, end=" ")

        # Resolve the *existing* annotation path (handles .pnm.json / .png.json etc)
        json_path = resolve_annotation_json_path(file_path, prefer_existing=True)

        if (json_path is None) or (not os.path.exists(json_path)):
            # No annotations found for this image
            if not self.includeTilesNotAnnotated:
                print(" -> no annotation json, skipping")
                return occurances
            # If you DO allow unannotated tiles, still pass the default new-style path
            # (tileImages should then treat it as missing / empty)
            json_path = f"{file_path}.json"

        # Use the explicit loader that accepts a json path
        tiles, tile_classes, tile_info, tiles_annotated_by_ai = loadImageAndJSON(
                                                                                 file_path,
                                                                                 json_path,
                                                                                 file_index,
                                                                                 border=self.border,
                                                                                 tile_size=self.tile_size,
                                                                                 step=self.step,
                                                                                 low_value_tile_threshold=self.threshold,
                                                                                 use_severity=self.use_severity,
                                                                                 use_clean_class=self.use_clean_class,
                                                                                 includeTilesAnnotatedByAI=self.includeTilesAnnotatedByAI,
                                                                                 debug=True
                                                                                )

        self.total_tiles_annotated_by_ai += tiles_annotated_by_ai
        print("Tiles : ", len(tiles), "             ")

        outdir = os.path.abspath(self.target_dir)
        os.makedirs(outdir, exist_ok=True)

        #tile_info has a string like: /media/ammar/games2/Datasets/Magician/AltinayWelding/colorFrame_0_00045.json(960,1008)
        #metaData is a dictionary with entries like : {'timestamp': '6661870', 'dev_timestamp': '5199', 'Button1': '0', 'Button2': '0', 'Distance1': 'H', 'Distance2': '156', 'Distance3': '202', 'Light1': '0', 'Light2': '0', 'Light3': '0', 'Light4': '1', 'Light5': '0', 'Light6': '0'}
        if metaData is None:
               metaData = {} 

        combinedTileInfo = [
                            {"source": one_tile_info, **metaData}
                            for one_tile_info in tile_info
                           ]
        #print("tile_info ",tile_info)
        #print("combinedTileInfo ",combinedTileInfo)


        occurances, cleanThreshold = dump_dataset_to_keras_data_loader(
                                                                        tiles, tile_classes, occurances, self.ratio_clean, outdir,
                                                                        tile_info=combinedTileInfo,
                                                                        tiles_annotated_by_ai=self.total_tiles_annotated_by_ai,
                                                                        progress_callback=None
                                                                       )

        del tiles
        del tile_classes
        gc.collect()
        return occurances


    def run(self):
        occurances = {'clean': 1}
        cleanThreshold = 60.0

        total_dirs = len(self.directories)
        dir_index = 0
        for dataset_dir in self.directories:
            if self._stop:
                break
            dir_index += 1
            # gather files in dataset_dir (non recursive) that are images and have .json alongside
            file_list = []
            for entry in sorted(os.listdir(dataset_dir)):
                entry_path = os.path.join(dataset_dir, entry)
                if not os.path.isfile(entry_path):
                    continue

                if (self.includeTilesNotAnnotated):
                    if entry.lower().endswith(('.jpg', '.png', '.jpeg', '.pnm')):
                       file_list.append(entry_path)
                else:
                    # Only include images that have annotations (supports legacy '*.pnm.json')
                    if entry.lower().endswith(('.jpg', '.png', '.jpeg', '.pnm')):
                      ann = find_annotation_for_image(entry_path)
                      if ann is not None:
                        # Keep the REAL image path (png/jpg/pnm). We'll ensure a compatible
                        # '<image>.json' sidecar exists right before processing.
                        file_list.append(entry_path)


            #print("Files to run on :",file_list)
            # Apply optional frame limits from info.json (startFrame/endFrame)
            start_frame = 0
            end_frame = len(file_list) - 1
            info_path = os.path.join(dataset_dir, "info.json")
            if os.path.isfile(info_path):
                try:
                    with open(info_path, "r") as f:
                        info = json.load(f)
                    if "startFrame" in info:
                        start_frame = int(info.get("startFrame", start_frame))
                    if "endFrame" in info:
                        end_frame = int(info.get("endFrame", end_frame))
                except Exception as e:
                    print("Warning: could not parse frame limits from", info_path, "->", e)

            # Clamp + slice list (indices correspond to sorted file order)
            if len(file_list) > 0:
                start_frame = max(0, min(start_frame, len(file_list) - 1))
                end_frame = max(0, min(end_frame, len(file_list) - 1))
                if end_frame < start_frame:
                    end_frame = start_frame
                if start_frame != 0 or end_frame != len(file_list) - 1:
                    print(f"Applying frame range {start_frame}..{end_frame} for {dataset_dir} (total {len(file_list)})")
                file_list = file_list[start_frame:end_frame + 1]
            else:
                start_frame = 0
                end_frame = -1

            # report start of dataset
            #wx.CallAfter(self.ui_callbacks['on_dataset_start'], dataset_dir, len(file_list), dir_index, total_dirs)

            self.readControllerCSV("%s/controller.csv" % dataset_dir) 


            files_processed = 0
            for file_index, file_path in enumerate(file_list, start=start_frame+1):
                if self._stop:
                    break
 
                frameNumber = self.get_frame_number_from_path(file_path)
                metaData    = self.getControllerCSVRowData(frameNumber)
                #print("File Index %u / Frame Number %u / File Path %s "%(file_index,frameNumber,file_path)," data = ",metaData)
                print("File Index %u / Frame Number %u / File Path %s "%(file_index,frameNumber,file_path)) #Less Verbose
          
                
                if (frameNumber is None):
                     print("Frame Number was not correctly resolved, stopping execution")
                     sys.exit(1)

                #if (metaData is None):
                #     print("Metadata was not correctly resolved, stopping execution")
                #     sys.exit(1) 

                okToProcessFile = (frameNumber is not None)
                if (self.ignoreSamplesWithNoMetadata):
                      okToProcessFile = (metaData is not None) and (frameNumber is not None)
 
                if okToProcessFile: #Only process metadata
                   self.process_a_file(file_index,file_path,metaData,occurances)

                   files_processed += 1
                #wx.CallAfter(self.ui_callbacks['on_file_done'], dataset_dir, file_path, files_processed, len(file_list))

            wx.CallAfter(self.ui_callbacks['on_dataset_done'], dataset_dir)

        print("Done running")
        wx.CallAfter(self.ui_callbacks['on_all_done'], self.target_dir)

    def stop(self):
        self._stop = True

# ------------------------------------------------------------------
# GUI
# ------------------------------------------------------------------
class MainFrame(wx.Frame):
    def __init__(self, start_dir):
        super().__init__(None, title="Magician Dataset Creator", size=(950, 700))
        self.start_dir = os.path.abspath(start_dir)
        self.processor = None

        panel = wx.Panel(self)
        main_sizer = wx.BoxSizer(wx.VERTICAL)

        # Top: directory list and controls
        top_sizer = wx.BoxSizer(wx.HORIZONTAL)

        # ---- Left side: dataset selection ----
        left_box = wx.StaticBox(panel, label="Datasets (select which to process)")
        left_sizer = wx.StaticBoxSizer(left_box, wx.VERTICAL)

        self.checklist = wx.CheckListBox(panel)
        left_sizer.Add(self.checklist, 1, wx.EXPAND | wx.ALL, 5)





        # --- Regex select + refresh row ---
        regex_row = wx.BoxSizer(wx.HORIZONTAL)

        regex_lbl = wx.StaticText(panel, label="Regex:")
        regex_row.Add(regex_lbl, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 6)

        self.regex_ctrl = wx.TextCtrl(panel, style=wx.TE_PROCESS_ENTER)
        self.regex_ctrl.SetHint("e.g. ^2026_|(Altinay|iit)")
        self.regex_ctrl.SetToolTip("Datasets matching this regex will be auto-selected (checked).")
        self.regex_ctrl.Bind(wx.EVT_TEXT, self.on_regex_changed)
        self.regex_ctrl.Bind(wx.EVT_TEXT_ENTER, self.on_regex_changed)
        regex_row.Add(self.regex_ctrl, 1, wx.EXPAND | wx.RIGHT, 8)

        refresh_btn = wx.Button(panel, label="Refresh list")
        refresh_btn.Bind(wx.EVT_BUTTON, self.on_refresh)
        regex_row.Add(refresh_btn, 0)

        left_sizer.Add(regex_row, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 5)





        top_sizer.Add(left_sizer, 2, wx.EXPAND | wx.ALL, 5)

        # ---- Right side: controls and progress ----
        right_box = wx.StaticBox(panel, label="Target, Config & Progress")
        right_sizer = wx.StaticBoxSizer(right_box, wx.VERTICAL)

        # Target folder chooser
        h = wx.BoxSizer(wx.HORIZONTAL)
        target_lbl = wx.StaticText(panel, label="Target output folder:")
        self.dirpicker = wx.DirPickerCtrl(panel, path=os.path.join(self.start_dir, 'keras_dataset'))
        self.dirpicker.Bind(wx.EVT_DIRPICKER_CHANGED, self.on_dir_changed)
        h.Add(target_lbl, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 8)
        h.Add(self.dirpicker, 1, wx.EXPAND)
        right_sizer.Add(h, 0, wx.EXPAND | wx.ALL, 5)

        # ---- Config Section ----
        config_box = wx.StaticBox(panel, label="Processing Options")
        config_sizer = wx.StaticBoxSizer(config_box, wx.VERTICAL)

        grid = wx.FlexGridSizer(0, 2, 8, 12)
        grid.AddGrowableCol(1, 1)

        # Tile size control
        grid.Add(wx.StaticText(panel, label="Tile size:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.tile_size_ctrl = wx.SpinCtrl(panel, min=8, max=512, initial=48)
        grid.Add(self.tile_size_ctrl, 1, wx.EXPAND)

        # Step size control
        grid.Add(wx.StaticText(panel, label="Step size:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.step_size_ctrl = wx.SpinCtrl(panel, min=1, max=512, initial=4)
        grid.Add(self.step_size_ctrl, 1, wx.EXPAND)

        # Border size control
        grid.Add(wx.StaticText(panel, label="Border safe-zone size:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.border_size_ctrl = wx.SpinCtrl(panel, min=0, max=16, initial=0)
        grid.Add(self.border_size_ctrl, 1, wx.EXPAND)

        # Ratio control
        grid.Add(wx.StaticText(panel, label="Clean/Non-clean ratio:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.ratio_ctrl = wx.SpinCtrl(panel, min=1, max=32, initial=10)
        grid.Add(self.ratio_ctrl, 1, wx.EXPAND)

        # Threshold control
        grid.Add(wx.StaticText(panel, label="Threshold per tile:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.threshold_ctrl = wx.SpinCtrl(panel, min=0, max=100, initial=20)
        grid.Add(self.threshold_ctrl, 1, wx.EXPAND)

        # Filtering checkbox
        grid.Add(wx.StaticText(panel, label="Options:"), 0, wx.ALIGN_CENTER_VERTICAL)

        self.filter_checkbox = wx.CheckBox(panel, label="Filtering")
        self.filter_checkbox.SetValue(True)
        self.severity_checkbox = wx.CheckBox(panel, label="Severities")
        self.severity_checkbox.SetValue(True)
        self.clean_class_checkbox = wx.CheckBox(panel, label="Clean Class")
        self.clean_class_checkbox.SetValue(True)
        comboButtons      = wx.BoxSizer(wx.HORIZONTAL)  
        comboButtons.Add(self.filter_checkbox, 0, wx.ALL, 5)
        comboButtons.Add(self.severity_checkbox, 0, wx.ALL, 5)
        comboButtons.Add(self.clean_class_checkbox, 0, wx.ALL, 5)
        grid.Add(comboButtons)

        # AI Annotation checkbox
        grid.Add(wx.StaticText(panel, label="Automatic Annotations:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.aiannotated_checkbox = wx.CheckBox(panel, label="Enable self-supervised annotations")
        self.aiannotated_checkbox.SetValue(True)
        grid.Add(self.aiannotated_checkbox, 1, wx.EXPAND)

        config_sizer.Add(grid, 0, wx.EXPAND | wx.ALL, 5)
        right_sizer.Add(config_sizer, 0, wx.EXPAND | wx.ALL, 5)

        # ---- Start/Stop buttons ----
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.start_btn = wx.Button(panel, label="Start")
        self.start_btn.Bind(wx.EVT_BUTTON, self.on_start)
        self.train_cmd_btn = wx.Button(panel, label="CLI")
        self.train_cmd_btn.SetToolTip("Show/copy the CLI commands to reproduce this dump + conversion (and training template) with the current GUI settings.")
        self.train_cmd_btn.Bind(wx.EVT_BUTTON, self.on_training_command)
        self.stop_btn = wx.Button(panel, label="Stop")
        self.stop_btn.Bind(wx.EVT_BUTTON, self.on_stop)
        self.stop_btn.Disable()

        # ---- Merge buttons ----
        self.merge_btn = wx.Button(panel, label="Merge")
        self.merge_btn.Bind(wx.EVT_BUTTON, self.on_merge)
        self.merge_btn.Disable()

        # ---- H5 Package button ----
        self.h5_btn = wx.Button(panel, label="H5 Package")
        self.h5_btn.Bind(wx.EVT_BUTTON, self.on_h5_package)
        self.h5_btn.Disable()

        btn_sizer.Add(self.start_btn)
        btn_sizer.Add(self.train_cmd_btn, 0, wx.LEFT, 8)
        btn_sizer.Add(self.stop_btn, 0, wx.LEFT, 8)
        btn_sizer.Add(self.merge_btn, 0,  wx.LEFT, 5)
        btn_sizer.Add(self.h5_btn, 0, wx.LEFT, 5)
        right_sizer.Add(btn_sizer, 0, wx.ALL, 5)

        # ---- Overall progress ----
        right_sizer.Add(wx.StaticText(panel, label="Overall progress:"), 0, wx.LEFT | wx.TOP, 6)
        self.overall_gauge = wx.Gauge(panel, range=100)
        right_sizer.Add(self.overall_gauge, 0, wx.EXPAND | wx.ALL, 5)

        # ---- Clean threshold display ----
        ct_sizer = wx.BoxSizer(wx.HORIZONTAL)
        ct_sizer.Add(wx.StaticText(panel, label="Clean Threshold:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 8)
        self.clean_threshold_gauge = wx.Gauge(panel, range=100)
        ct_sizer.Add(self.clean_threshold_gauge, 1, wx.EXPAND)
        self.clean_threshold_label = wx.StaticText(panel, label="60.0")
        ct_sizer.Add(self.clean_threshold_label, 0, wx.LEFT, 8)
        right_sizer.Add(ct_sizer, 0, wx.EXPAND | wx.ALL, 5)

        # ---- Classes encountered ----
        right_sizer.Add(wx.StaticText(panel, label="Classes encountered:"), 0, wx.LEFT | wx.TOP, 6)
        self.classes_panel = scrolled.ScrolledPanel(panel, size=(-1, 240))
        self.classes_panel.SetupScrolling()
        self.classes_sizer = wx.BoxSizer(wx.VERTICAL)
        self.classes_panel.SetSizer(self.classes_sizer)
        right_sizer.Add(self.classes_panel, 1, wx.EXPAND | wx.ALL, 5)

        # ---- Log area ----
        right_sizer.Add(wx.StaticText(panel, label="Log:"), 0, wx.LEFT | wx.TOP, 6)
        self.log_ctrl = wx.TextCtrl(panel, style=wx.TE_MULTILINE | wx.TE_READONLY, size=(-1, 120))
        right_sizer.Add(self.log_ctrl, 0, wx.EXPAND | wx.ALL, 5)

        top_sizer.Add(right_sizer, 3, wx.EXPAND | wx.ALL, 5)

        main_sizer.Add(top_sizer, 1, wx.EXPAND)

        # Bottom: status
        self.status = wx.StaticText(panel, label=f"Start directory: {self.start_dir}")
        main_sizer.Add(self.status, 0, wx.LEFT | wx.BOTTOM, 8)

        panel.SetSizer(main_sizer)

        # Init
        self.class_widgets = {}  # class_name -> (label, gauge)
        self.populate_dataset_list()


    # -----------------------------
    # UI helpers and handlers
    # -----------------------------
    def log(self, msg):
        timestamp = time.strftime('%H:%M:%S')
        self.log_ctrl.AppendText(f"[{timestamp}] {msg}\n")
        print(f"[{timestamp}] {msg}\n")

    def on_dir_changed(self, event):
        path = self.dirpicker.GetPath()
        #wx.MessageBox(f"New directory selected:\n{path}", "Directory Changed")
        self.merge_btn.Enable()
        self.h5_btn.Enable()

    def populate_dataset_list(self):
        self.checklist.Clear()
        if not os.path.isdir(self.start_dir):
            self.log(f"Start directory does not exist: {self.start_dir}")
            return

        #entries = [os.path.join(self.start_dir, d) for d in os.listdir(self.start_dir) if os.path.isdir(os.path.join(self.start_dir, d))]
        entries = retrieve_list_of_potential_directories(self.start_dir)

        entries.sort()
        items = []
        self.dataset_paths = []
        for p in entries:
            has_json = False
            # scan for .json files in directory
            try:
                for f in os.listdir(p):
                    if f.lower().endswith('.json'):
                        has_json = True
                        break
            except Exception:
                pass
            #Create entry label 
            #----------------------------------------------------------
            label = os.path.basename(p)
            if has_json:
               jsonFilesCount  = count_files_with_extension(p,".json")
               label           = label + " [%u json]" % jsonFilesCount
            imageFilesCount = count_files_with_extension(p,".pnm") + count_files_with_extension(p,".png")
            label           = label + " [%u .img]" % imageFilesCount
            #----------------------------------------------------------
            items.append(label)
            self.dataset_paths.append(p)
        if not items:
            items = ["(no subdirectories found)"]
        self.checklist.InsertItems(items, 0)
        # default check those that have json
        for idx, p in enumerate(self.dataset_paths):
            try:
                for f in os.listdir(p):
                    if f.lower().endswith('.json'):
                        self.checklist.Check(idx, True)
                        break
            except Exception:
                pass

        # If user has a regex set, re-apply it after rebuilding the list
        if hasattr(self, "regex_ctrl"):
            self.apply_regex_selection()



    def on_merge(self, event):
        self.log("Executing merging application")
        os.system("python3 mergeDatasets.py %s" % self.dirpicker.GetPath())

    def on_refresh(self, event):
        self.populate_dataset_list()
        self.log("Refreshed dataset list")


    def apply_regex_selection(self):
        # If list is empty / placeholder, do nothing
        if not hasattr(self, "dataset_paths") or self.checklist.GetCount() == 0:
            return

        pattern = ""
        if hasattr(self, "regex_ctrl") and self.regex_ctrl is not None:
            pattern = self.regex_ctrl.GetValue().strip()

        # Empty pattern: do nothing (keeps user’s manual selection)
        if pattern == "":
            return

        try:
            rx = re.compile(pattern)
            self.regex_ctrl.SetBackgroundColour(wx.NullColour)
            self.regex_ctrl.SetToolTip("Datasets matching this regex will be auto-selected (checked).")
        except re.error as e:
            # Invalid regex: highlight field, don’t change selection
            self.regex_ctrl.SetBackgroundColour(wx.Colour(255, 220, 220))
            self.regex_ctrl.SetToolTip(f"Invalid regex: {e}")
            self.regex_ctrl.Refresh()
            return

        # Check/uncheck items based on match against label OR folder name
        for i in range(len(self.dataset_paths)):
            label = self.checklist.GetString(i)
            folder = os.path.basename(self.dataset_paths[i])

            match = (rx.search(label) is not None) or (rx.search(folder) is not None)
            self.checklist.Check(i, match)

    def on_regex_changed(self, event):
        self.apply_regex_selection()
        event.Skip()

    def maybe_clear_target_dir(self, target_dir):
     # Check if target_dir exists and is not empty
     if os.path.isdir(target_dir) and os.listdir(target_dir):
        dlg = wx.MessageDialog(
            self,
            f"The target folder already contains data:\n\n{target_dir}\n\n"
            "Do you want to empty it?",
            "Delete previous data?",
            wx.YES_NO | wx.ICON_QUESTION
        )
        res = dlg.ShowModal()
        dlg.Destroy()

        if res == wx.ID_YES:
            try:
                # Clean target_dir contents but keep the folder itself
                for filename in os.listdir(target_dir):
                    file_path = os.path.join(target_dir, filename)
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)  # delete file or symlink
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)  # delete directory

                """
                wx.MessageBox(
                    f"Emptied previous data from archive:\n{target_dir}",
                    "Target output directory is clean",
                    wx.OK | wx.ICON_INFORMATION
                )
                """

                self.log(f"Removed previously existing data from: {target_dir}")
            except Exception as e:
                wx.MessageBox(
                    f"Failed to empty target dataset directory: {e}",
                    "Error",
                    wx.OK | wx.ICON_ERROR
                )
                self.log(f"Failed to empty target dataset directory: {e}")
     else:
        # Directory doesn't exist or is empty, nothing to do
        self.log(f"Target directory is already empty: {target_dir}")

    def on_start(self, event):
        # collect selected dataset paths
        selections = [i for i in range(self.checklist.GetCount()) if self.checklist.IsChecked(i)]
        if not selections:
            wx.MessageBox("Please select at least one dataset to process.", "No selection", wx.ICON_WARNING)
            return
        selected_dirs = [self.dataset_paths[i] for i in selections]
        target_dir = self.dirpicker.GetPath()
        if not target_dir:
            wx.MessageBox("Please choose a target output directory.", "No target", wx.ICON_WARNING)
            return

        # ask to delete previous data
        self.maybe_clear_target_dir(target_dir)

        # disable controls
        self.start_btn.Disable()
        self.stop_btn.Enable()
        self.log_ctrl.Clear()
        self.overall_gauge.SetValue(0)
        self.clean_threshold_gauge.SetValue(60)
        self.clean_threshold_label.SetLabel("60.0")
        # reset class widgets
        for key, (lbl, g) in list(self.class_widgets.items()):
            lbl.Destroy()
            g.Destroy()
        self.class_widgets.clear()
        self.classes_sizer.Clear(True)
        self.classes_panel.SetupScrolling()

        ui_callbacks = {
            'on_dataset_start': self.on_dataset_start,
            'on_progress_update': self.on_progress_update,
            'on_file_done': self.on_file_done,
            'on_dataset_done': self.on_dataset_done,
            'on_all_done': self.on_all_done,
        }
        self.processor = ProcessorThread(selected_dirs, target_dir, ui_callbacks,
                                         ratio_clean     = self.ratio_ctrl.GetValue(), 
                                         threshold       = self.threshold_ctrl.GetValue(),
                                         border          = self.border_size_ctrl.GetValue(), 
                                         step            = self.step_size_ctrl.GetValue(), 
                                         tile_size       = self.tile_size_ctrl.GetValue(),
                                         use_severity    = self.severity_checkbox.GetValue(),
                                         use_clean_class = self.clean_class_checkbox.GetValue(),
                                         includeTilesAnnotatedByAI = self.aiannotated_checkbox.GetValue()
                                         )
        self.processor.start()
        self.log(f"Started processing {len(selected_dirs)} datasets -> {target_dir}")


    def on_training_command(self, event):
        """Build a CLI command that reproduces the current GUI dump configuration."""
        selections = [i for i in range(self.checklist.GetCount()) if self.checklist.IsChecked(i)]
        if not selections:
            wx.MessageBox("Please select at least one dataset first.", "No selection", wx.ICON_WARNING)
            return

        selected_dirs = [self.dataset_paths[i] for i in selections]
        target_dir = self.dirpicker.GetPath()
        if not target_dir:
            wx.MessageBox("Please choose a target output directory first.", "No target", wx.ICON_WARNING)
            return

        # Mirror GUI options -> CLI flags
        tile_size = self.tile_size_ctrl.GetValue()
        step      = self.step_size_ctrl.GetValue()
        border    = self.border_size_ctrl.GetValue()
        thresh    = self.threshold_ctrl.GetValue()
        ratio     = self.ratio_ctrl.GetValue()

        use_sev   = self.severity_checkbox.GetValue()
        use_clean = self.clean_class_checkbox.GetValue()
        ai_ann    = self.aiannotated_checkbox.GetValue()

        # We call the updated CLI dumper (drop-in replacement of dumpKerasDataset.py)
        dumper = "python3 dumpDataset.py"

        common_flags = [
            f'--tile-size {tile_size}',
            f'--step {step}',
            f'--border {border}',
            f'--threshold {thresh}',
            f'--ratio-clean {ratio}',
            ('--use-severity' if use_sev else '--no-use-severity'),
            ('--use-clean-class' if use_clean else '--no-use-clean-class'),
            ('--include-ai-annotated' if ai_ann else '--no-include-ai-annotated'),
        ]
        common = " ".join(common_flags)

        lines = []
        for k, d in enumerate(selected_dirs):
            clear = " --clear-output" if k == 0 else ""
            lines.append(f'{dumper} --directory "{d}" -o "{target_dir}"{clear} {common}')

        # Optional next steps (same as GUI buttons / typical flow)
        lines.append(f'python3 mergeDatasets.py "{target_dir}"  # optional (same as "Merge Outputs")')
        lines.append(f'python3 ../magician_vision_classifier/DatasetConverter.py "{target_dir}"  # same as "H5 Package"')
        lines.append('# Training step (adjust to your training script / entrypoint):')
        lines.append(f'# python3 ../magician_vision_classifier/train.py --dataset "{target_dir}"')

        cmd_text = "\n".join(lines)

        # Copy to clipboard for convenience
        try:
            if wx.TheClipboard.Open():
                wx.TheClipboard.SetData(wx.TextDataObject(cmd_text))
                wx.TheClipboard.Close()
                copied = True
            else:
                copied = False
        except Exception:
            copied = False

        msg = "Commands copied to clipboard." if copied else "Commands (could not copy to clipboard automatically):"
        dlg = wx.MessageDialog(self, f"{msg}\n\n{cmd_text}", "Training command", wx.OK | wx.ICON_INFORMATION)
        dlg.ShowModal()
        dlg.Destroy()

    def on_stop(self, event):
        if self.processor:
            self.processor.stop()
            self.log("Stop requested. Waiting for worker to stop...")
            self.stop_btn.Disable()

    # -----------------------------
    # Callbacks invoked from worker via wx.CallAfter
    # -----------------------------
    def on_dataset_start(self, dataset_dir, file_count, index, total_dirs):
        self.log(f"Dataset {index}/{total_dirs}: {dataset_dir} - {file_count} files to process")

    def on_progress_update(self, info):
        # info contains sample_id, total_samples, occurances, cleanThreshold, current_class
        sample_id      = info.get('sample_id', 0)
        total_samples  = info.get('total_samples', 1)
        occurances     = info.get('occurances', {})
        cleanThreshold = info.get('cleanThreshold', 60.0)
        current_class  = info.get('current_class', '')

        # update overall gauge (we don't have a global total across all datasets here, so we'll use sample %)
        try:
            pct = int((sample_id / max(1, total_samples)) * 100)
        except Exception:
            pct = 0
        self.overall_gauge.SetValue(min(100, pct))

        # update clean threshold gauge
        self.clean_threshold_gauge.SetValue(int(min(100, max(0, cleanThreshold))))
        self.clean_threshold_label.SetLabel(f"{cleanThreshold:.3f}")

        # update classes list
        # ensure widgets exist for each class
        for cls, cnt in sorted(occurances.items(), key=lambda kv: kv[0]):
            if cls not in self.class_widgets:
                h      = wx.BoxSizer(wx.HORIZONTAL)
                lbl    = wx.StaticText(self.classes_panel, label=f"{cls}")
                gauge  = wx.Gauge(self.classes_panel, range=100)
                valtxt = wx.StaticText(self.classes_panel, label=str(cnt))
                h.Add(lbl, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 8)
                h.Add(gauge, 1, wx.EXPAND | wx.RIGHT, 8)
                h.Add(valtxt, 0, wx.ALIGN_CENTER_VERTICAL)
                self.classes_sizer.Add(h, 0, wx.EXPAND | wx.ALL, 4)
                self.class_widgets[cls] = (lbl, gauge, valtxt)
                self.classes_panel.Layout()
            else:
                lbl, gauge, valtxt = self.class_widgets[cls]
            # update gauge proportionally to max count among classes
        # compute max
        maxcnt = max([v for k, v in occurances.items()]) if occurances else 1
        for cls, cnt in occurances.items():
            lbl, gauge, valtxt = self.class_widgets[cls]
            try:
                gval = int((cnt / maxcnt) * 100)
            except Exception:
                gval = 0
            gauge.SetValue(gval)
            valtxt.SetLabel(str(cnt))
        self.classes_panel.SetupScrolling(scroll_x=False)

    def on_file_done(self, dataset_dir, file_path, files_done, files_total):
        self.log(f"Processed file {files_done}/{files_total}: {os.path.basename(file_path)}")

    def on_dataset_done(self, dataset_dir):
        self.log(f"Finished dataset: {dataset_dir}")

    def on_h5_package(self, event):
        output_path = self.dirpicker.GetPath()
    
        if not output_path or not os.path.isdir(output_path):
            wx.MessageBox("Please select a valid output directory first.",
                          "Invalid Output Path",
                          wx.OK | wx.ICON_WARNING)
            return
    
        cmd = f"python3 ../magician_vision_classifier/DatasetConverter.py {output_path}"
        self.log(f"Executing H5 packaging:\n{cmd}")
    
        ret = os.system(cmd)
    
        if ret == 0:
            wx.MessageBox("H5 Packaging completed successfully.",
                          "Success",
                          wx.OK | wx.ICON_INFORMATION)
            self.log("H5 Packaging completed successfully.")
        else:
            wx.MessageBox("H5 Packaging failed. Check console output.",
                          "Error",
                          wx.OK | wx.ICON_ERROR)
            self.log("H5 Packaging failed.")
    

    def on_all_done(self, target_dir):
        self.log(f"All done. Output written to: {target_dir}")
        self.start_btn.Enable()
        self.stop_btn.Disable()

        #================================================================================================
        # Ask to zip
        #================================================================================================
        dlg = wx.MessageDialog(self, f"Processing finished. Do you want to zip the output folder {target_dir}?\n It will take a lot of time!", "Zip output?", wx.YES_NO | wx.ICON_QUESTION)
        res = dlg.ShowModal()
        dlg.Destroy()
        if res == wx.ID_YES:
            try:
                base = os.path.abspath(target_dir)
                print("Creating Zip Archive .. (this will take a lot of time..)")
                self.log(f"Creating Zip Archive .. (this will take a lot of time..)")
                #archive_name = shutil.make_archive(base, 'zip', base_dir=base)
                archive_name = os.path.abspath(target_dir)
                os.system(f'zip -rv "{archive_name}.zip" "{target_dir}/"')
                wx.MessageBox(f"Created archive: {archive_name}", "Zip created", wx.ICON_INFORMATION)
                self.log(f"Created archive: {archive_name}")
            except Exception as e:
                wx.MessageBox(f"Failed to create archive: {e}", "Error", wx.ICON_ERROR)
                self.log(f"Failed to create archive: {e}")



        # ---------------------------------------------------
        # Ask if user wants to convert dataset to HDF5
        # ---------------------------------------------------
        dlg = wx.MessageDialog(
            self,
            f"Do you want to package the dataset into HDF5 format (dataset.h5)?\n\n"
            f"Now is the time to remove classes you dont want from the directory.\nThis will run DatasetConverter.py on:\n{target_dir}",
            "Create HDF5 dataset?",
            wx.YES_NO | wx.ICON_QUESTION
        )

        res = dlg.ShowModal()
        dlg.Destroy()

        if res == wx.ID_YES:
            try:
                cmd = f"python3 ../magician_vision_classifier/DatasetConverter.py {target_dir}"
                self.log(f"Executing H5 packaging:\n{cmd}")
                ret = os.system(cmd)

                if ret == 0:
                    wx.MessageBox(
                        "HDF5 dataset successfully created.",
                        "H5 Packaging",
                        wx.OK | wx.ICON_INFORMATION
                    )
                    self.log("HDF5 packaging completed.")
                else:
                    wx.MessageBox(
                        "HDF5 packaging failed. Check console output.",
                        "H5 Packaging Error",
                        wx.OK | wx.ICON_ERROR
                    )
                    self.log("HDF5 packaging failed.")

            except Exception as e:
                wx.MessageBox(
                    f"Failed to run DatasetConverter.py:\n{e}",
                    "Error",
                    wx.OK | wx.ICON_ERROR
                )
                self.log(f"HDF5 packaging error: {e}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    start_dir = os.path.join(os.getcwd(), './')
    if len(sys.argv) > 1:
        start_dir = sys.argv[1]

    app = wx.App(False)
    frame = MainFrame(start_dir=start_dir)
    frame.Show()
    app.MainLoop()

if __name__ == '__main__':
    main()

