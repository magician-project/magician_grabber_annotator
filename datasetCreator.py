#!/usr/bin/env python3
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
import threading
import time
import shutil
import cv2
import random


#from memory_profiler import profile


# --- Import external utilities assumed to exist in readData.py ---
try:
    from readData import loadImage, checkIfFileExists, checkIfPathExists, check_threshold, check_variation
except Exception as e:
    print("Could not import readData utilities. Make sure readData.py exists and is importable.\n", e)
    # We'll still continue; the UI will warn the user if the import failed.
    loadImage = None
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
#@profile
def dump_dataset_to_keras_data_loader(tiles, tile_classes, occurances, ratio_clean, outputDirectory, progress_callback=None):
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

        target_clean_to_non_clean_ratio = ratio_clean  # 1.0 = target 1:1  of non-clean

        # Auto-balance clean samples
        # Determine probability to skip this clean tile
        skip_clean_prob = 0.0
        if thisClass == "clean" and non_clean_sum > 0:
            max_clean_allowed = target_clean_to_non_clean_ratio * non_clean_sum
            if max_clean_allowed > 0:
              ratio = clean_count / max_clean_allowed 
            else: 
              ratio = 1.0
            # Smoothly increase skip probability as clean_count approaches max_clean_allowed
            skip_clean_prob = min(1.0, max(0.0, ratio))

        print("Tile ",tileID,"/",len(tiles)," | Clean ",clean_count," / Non-Clean ",non_clean_sum," | ",len(tiles)," tiles |  Skip Clean Prob " , skip_clean_prob,end="\r")

        thisClass = tile_classes[tileID]
        if (thisClass == ""):
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
    def __init__(self, directories, target_dir, ui_callbacks, ratio_clean=1.0, threshold=0, border=0, step=32, tile_size=64):
        super().__init__()
        self.directories = directories
        self.target_dir = target_dir
        self.ui_callbacks = ui_callbacks  # dict of functions for UI updates
        self.step = step
        self.tile_size = tile_size
        self.ratio_clean = ratio_clean
        self.border = border
        self.threshold = threshold
        self._stop = False
  
    def process_a_file(self,file_index,file_path,occurances):
                #print("File : ",file_path," index : ",file_index,end=" ")
                print("File : ",file_path,end=" ")
  
                # load tiles using loadImage if available else try a simple fallback
                tiles, tile_classes = loadImage(file_path, file_index, border=self.border, tile_size=self.tile_size, step=self.step, low_value_tile_threshold=self.threshold)
                print("Tiles : ",len(tiles)," ")
                # Create per-file output dir under target_dir -> maintain the same structure: one output root for everything
                outdir = os.path.abspath(self.target_dir)
                os.makedirs(outdir, exist_ok=True)

                # progress callback to update UI for each tile
                progress_cb = None
                #def progress_cb(info):
                #    wx.CallAfter(self.ui_callbacks['on_progress_update'], info)

                occurances, cleanThreshold = dump_dataset_to_keras_data_loader(tiles, tile_classes, occurances, self.ratio_clean, outdir, progress_callback=progress_cb)

                #Empty memory occupied by tiles 
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
                if entry.lower().endswith(('.jpg', '.png', '.jpeg', '.pnm')) and os.path.exists(entry_path + '.json'):
                    file_list.append(entry_path)

            # report start of dataset
            #wx.CallAfter(self.ui_callbacks['on_dataset_start'], dataset_dir, len(file_list), dir_index, total_dirs)

            files_processed = 0
            for file_index, file_path in enumerate(file_list, start=1):
                if self._stop:
                    break
 
                self.process_a_file(file_index,file_path,occurances)

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




        refresh_btn = wx.Button(panel, label="Refresh list")
        refresh_btn.Bind(wx.EVT_BUTTON, self.on_refresh)
        left_sizer.Add(refresh_btn, 0, wx.ALIGN_RIGHT | wx.RIGHT | wx.BOTTOM, 5)



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
        self.tile_size_ctrl = wx.SpinCtrl(panel, min=8, max=512, initial=32)
        grid.Add(self.tile_size_ctrl, 1, wx.EXPAND)

        # Step size control
        grid.Add(wx.StaticText(panel, label="Step size:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.step_size_ctrl = wx.SpinCtrl(panel, min=1, max=512, initial=8)
        grid.Add(self.step_size_ctrl, 1, wx.EXPAND)

        # Border size control
        grid.Add(wx.StaticText(panel, label="Border safe-zone size:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.border_size_ctrl = wx.SpinCtrl(panel, min=0, max=16, initial=8)
        grid.Add(self.border_size_ctrl, 1, wx.EXPAND)

        # Ratio control
        grid.Add(wx.StaticText(panel, label="Clean/Non-clean ratio:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.ratio_ctrl = wx.SpinCtrl(panel, min=1, max=32, initial=2)
        grid.Add(self.ratio_ctrl, 1, wx.EXPAND)

        # Threshold control
        grid.Add(wx.StaticText(panel, label="Threshold per tile:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.threshold_ctrl = wx.SpinCtrl(panel, min=0, max=100, initial=20)
        grid.Add(self.threshold_ctrl, 1, wx.EXPAND)

        # Filtering checkbox
        grid.Add(wx.StaticText(panel, label="Filtering:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.filter_checkbox = wx.CheckBox(panel, label="Enable filtering")
        self.filter_checkbox.SetValue(True)
        grid.Add(self.filter_checkbox, 1, wx.EXPAND)

        config_sizer.Add(grid, 0, wx.EXPAND | wx.ALL, 5)
        right_sizer.Add(config_sizer, 0, wx.EXPAND | wx.ALL, 5)

        # ---- Start/Stop buttons ----
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.start_btn = wx.Button(panel, label="Start Dump")
        self.start_btn.Bind(wx.EVT_BUTTON, self.on_start)
        self.stop_btn = wx.Button(panel, label="Stop")
        self.stop_btn.Bind(wx.EVT_BUTTON, self.on_stop)
        self.stop_btn.Disable()

        # ---- Merge buttons ----
        self.merge_btn = wx.Button(panel, label="Merge Output Classes")
        self.merge_btn.Bind(wx.EVT_BUTTON, self.on_merge)
        self.merge_btn.Disable()

        btn_sizer.Add(self.start_btn)
        btn_sizer.Add(self.stop_btn, 0, wx.LEFT, 8)
        btn_sizer.Add(self.merge_btn, 0,  wx.LEFT, 5)
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

    def on_dir_changed(self, event):
        path = self.dirpicker.GetPath()
        #wx.MessageBox(f"New directory selected:\n{path}", "Directory Changed")
        self.merge_btn.Enable()

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
            label = os.path.basename(p) + ("  [json]" if has_json else "")
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

    def on_merge(self, event):
        self.log("Executing merging application")
        os.system("python3 mergeDatasets.py %s" % self.dirpicker.GetPath())

    def on_refresh(self, event):
        self.populate_dataset_list()
        self.log("Refreshed dataset list")


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
                                         ratio_clean = self.ratio_ctrl.GetValue(), 
                                         threshold   = self.threshold_ctrl.GetValue(),
                                         border      = self.border_size_ctrl.GetValue(), 
                                         step        = self.step_size_ctrl.GetValue(), 
                                         tile_size   = self.tile_size_ctrl.GetValue())
        self.processor.start()
        self.log(f"Started processing {len(selected_dirs)} datasets -> {target_dir}")

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

    def on_all_done(self, target_dir):
        self.log(f"All done. Output written to: {target_dir}")
        self.start_btn.Enable()
        self.stop_btn.Disable()

        # ask to zip
        dlg = wx.MessageDialog(self, f"Processing finished. Do you want to zip the output folder {target_dir}?", "Zip output?", wx.YES_NO | wx.ICON_QUESTION)
        res = dlg.ShowModal()
        dlg.Destroy()
        if res == wx.ID_YES:
            try:
                base = os.path.abspath(target_dir)
                archive_name = shutil.make_archive(base, 'zip', base_dir=base)
                wx.MessageBox(f"Created archive: {archive_name}", "Zip created", wx.ICON_INFORMATION)
                self.log(f"Created archive: {archive_name}")
            except Exception as e:
                wx.MessageBox(f"Failed to create archive: {e}", "Error", wx.ICON_ERROR)
                self.log(f"Failed to create archive: {e}")

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

