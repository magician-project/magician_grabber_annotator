#!/usr/bin/python3

""" 
Author : "Ammar Qammaz"
Copyright : "2025 Foundation of Research and Technology, Computer Science Department Greece, See license.txt"
License : "FORTH" 
"""

"""
wx_tile_viewer.py

A simple wxPython + OpenCV tile viewer + small processing/PCA integration.

Features implemented:
 - scan a directory containing many PNG tiles
 - navigate with a slider (and next/prev buttons)
 - upscale selected tile to 200x200 and show 4 processing slots (dropdowns)
 - processing ops: None, Threshold >, Threshold <, Sobel X, Sobel Y, Canny
 - reads PNG Comment metadata (Pillow) and shows it
 - "Load PCA" button: tries to load a PCA saved file using the provided principleComponentAnalysis.PCA class (if present) and computes embeddings by flattening grayscale tiles (lazy batch processing)
 - if PCA embeddings are available, allows KNN search for nearest tiles to a chosen query image
 - thumbnails for K nearest results

Notes:
 - This is a simple desktop utility meant as a starting point. For 100k files you should run PCA/embeddings offline and keep them dumped to disk (the GUI will attempt batched processing but it can still be slow).
 - Dependencies: wxPython, opencv-python, pillow, numpy, matplotlib, scikit-learn (optional for some PCA variants). Install via pip if needed.

Usage: python wx_tile_viewer.py

"""

import wx
import wx.lib.scrolledpanel as scrolled
import os
import sys
import glob
import cv2
import numpy as np
from PIL import Image
import threading
import traceback
import json

try:
    import principleComponentAnalysis as pca_module
except Exception:
    pca_module = None

def get_png_comment(filename):
    from PIL import Image
    try:
        img = Image.open(filename)
        comment = img.info.get("Comment", "")
        img.close()
        return comment
    except Exception:
        return ""

PROCESSING_OPTIONS = ["None", "Threshold >", "Threshold <", "Sobel X", "Sobel Y", "Canny"]

def readTile(image_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if image is None:
        print("Error: Unable to read the image ",image_path,".")
        return None 

    return image

class TileDataset:
    def __init__(self, folder):
        self.folder = folder
        self.files = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.PNG"):
            self.files.extend(glob.glob(os.path.join(folder, ext)))
        self.files.sort()
    def __len__(self):
        return len(self.files)
    def get_path(self, idx):
        if 0 <= idx < len(self.files):
            return self.files[idx]
    def load_image(self, idx, as_gray=True):
        path = self.get_path(idx)
        if path is None:
            return None

        img = readTile(path)
        if img is None:
            return None
        if as_gray:
            if len(img.shape) == 4:
                img = np.mean(img, axis=2).astype(img.dtype)
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

class MainFrame(wx.Frame):
    def __init__(self,start_dir):
        super().__init__(None, title="Tile viewer", size=(1200,800))
        self.panel = wx.Panel(self)

        self.dataset       = None
        self.current_index = 0
        self.current_img   = None
        self.pca           = None
        self.embeddings    = None

        self.folder_picker = wx.DirPickerCtrl(self.panel, message="Select folder")
        if (start_dir!="./"):
             self.folder_picker.SetPath(start_dir)
        self.load_btn = wx.Button(self.panel, label="Load folder")
        self.load_btn.Bind(wx.EVT_BUTTON, self.on_load_folder)

        self.slider = wx.Slider(self.panel, style=wx.SL_HORIZONTAL)
        self.slider.Bind(wx.EVT_SLIDER, self.on_slider)

        self.prev_btn = wx.Button(self.panel, label="Prev")
        self.next_btn = wx.Button(self.panel, label="Next")
        self.prev_btn.Bind(wx.EVT_BUTTON, lambda e: self.change_index(-1))
        self.next_btn.Bind(wx.EVT_BUTTON, lambda e: self.change_index(1))

        self.image_ctrl = wx.StaticBitmap(self.panel)
        self.meta_txt = wx.TextCtrl(self.panel, style=wx.TE_MULTILINE|wx.TE_READONLY, size=(300,120))

        self.processing_checkboxes, self.processing_choices, self.processing_thresholds = [], [], []
        for i in range(4):
            cb = wx.CheckBox(self.panel, label=f"Slot {i+1}")
            choice = wx.Choice(self.panel, choices=PROCESSING_OPTIONS)
            choice.SetSelection(0)
            sld = wx.Slider(self.panel, minValue=0, maxValue=255, value=128)
            self.processing_checkboxes.append(cb)
            self.processing_choices.append(choice)
            self.processing_thresholds.append(sld)

        self.pca_btn = wx.Button(self.panel, label="Load PCA")
        self.pca_btn.Bind(wx.EVT_BUTTON, self.on_load_pca)
        self.compute_embeddings_btn = wx.Button(self.panel, label="Compute embeddings")
        self.compute_embeddings_btn.Bind(wx.EVT_BUTTON, self.on_compute_embeddings)

        self.save_embeddings_btn = wx.Button(self.panel, label="Save embeddings")
        self.save_embeddings_btn.Bind(wx.EVT_BUTTON, self.on_save_embeddings)
        self.load_embeddings_btn = wx.Button(self.panel, label="Load embeddings")
        self.load_embeddings_btn.Bind(wx.EVT_BUTTON, self.on_load_embeddings)

        self.knn_btn = wx.Button(self.panel, label="KNN search")
        self.knn_btn.Bind(wx.EVT_BUTTON, self.on_knn_search)
        self.knn_k = wx.SpinCtrl(self.panel, min=1, max=50, initial=5)

        self.results_panel = scrolled.ScrolledPanel(self.panel, size=(1000,150))
        self.results_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.results_panel.SetSizer(self.results_sizer)
        self.results_panel.SetupScrolling()

        top = wx.BoxSizer(wx.VERTICAL)
        row1 = wx.BoxSizer(wx.HORIZONTAL)
        row1.Add(self.folder_picker, 1, wx.EXPAND|wx.ALL,4)
        row1.Add(self.load_btn, 0, wx.ALL,4)
        top.Add(row1, 0, wx.EXPAND)

        nav = wx.BoxSizer(wx.HORIZONTAL)
        nav.Add(self.prev_btn, 0, wx.ALL,4)
        nav.Add(self.slider, 1, wx.EXPAND|wx.ALL,4)
        nav.Add(self.next_btn, 0, wx.ALL,4)
        top.Add(nav, 0, wx.EXPAND)

        disp = wx.BoxSizer(wx.HORIZONTAL)
        disp.Add(self.image_ctrl, 0, wx.ALL,8)
        right = wx.BoxSizer(wx.VERTICAL)
        right.Add(self.meta_txt, 0, wx.EXPAND|wx.ALL,4)

        grid = wx.GridSizer(rows=4, cols=3, vgap=4, hgap=4)
        for i in range(4):
            grid.Add(self.processing_checkboxes[i], 0)
            grid.Add(self.processing_choices[i], 0, wx.EXPAND)
            grid.Add(self.processing_thresholds[i], 0, wx.EXPAND)
        right.Add(grid, 0, wx.EXPAND|wx.ALL,4)

        pcarow = wx.BoxSizer(wx.HORIZONTAL)
        for w in (self.pca_btn, self.compute_embeddings_btn, self.save_embeddings_btn, self.load_embeddings_btn):
            pcarow.Add(w, 0, wx.ALL,4)
        pcarow.Add(wx.StaticText(self.panel, label="k:"), 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL,6)
        pcarow.Add(self.knn_k, 0, wx.ALL,4)
        pcarow.Add(self.knn_btn, 0, wx.ALL,4)
        right.Add(pcarow, 0, wx.EXPAND)

        disp.Add(right, 1, wx.EXPAND)
        top.Add(disp, 1, wx.EXPAND)

        top.Add(wx.StaticText(self.panel, label="KNN Results:"), 0, wx.ALL,4)
        top.Add(self.results_panel, 0, wx.EXPAND|wx.ALL,4)
        self.panel.SetSizer(top)
        self.update_ui()

        wx.MessageBox(f"Ammar: This utility is under construction!", "Under Construction")

    def update_ui(self):
        if self.dataset:
            self.slider.SetRange(0, max(0,len(self.dataset)-1))
            self.slider.SetValue(self.current_index)
        else:
            self.slider.SetRange(0,1)
            self.slider.SetValue(0)

    def on_load_folder(self, evt):
        path = self.folder_picker.GetPath()
        if not os.path.isdir(path):
            wx.MessageBox("Invalid folder","Error")
            return
        self.dataset = TileDataset(path)
        if len(self.dataset)==0:
            wx.MessageBox("No images found","Error")
            return
        self.current_index = 0
        self.update_ui()
        self.show_current()

    def on_slider(self, evt):
        self.current_index = self.slider.GetValue()
        self.show_current()

    def change_index(self, d):
        if not self.dataset: return
        self.current_index = max(0, min(len(self.dataset)-1, self.current_index+d))
        self.slider.SetValue(self.current_index)
        self.show_current()

    def show_current(self):
        if not self.dataset: return
        img = self.dataset.load_image(self.current_index, as_gray=True)
        if img is None: return
        self.current_img = img
        up = cv2.resize(img, (200,200), interpolation=cv2.INTER_NEAREST)
        proc = up.copy()
        for i in range(4):
            if not self.processing_checkboxes[i].GetValue():
                continue
            opt = self.processing_choices[i].GetStringSelection()
            val = self.processing_thresholds[i].GetValue()
            proc = self.apply_processing(proc,opt,val)
        bmp = wx.Bitmap.FromBuffer(proc.shape[1],proc.shape[0],cv2.cvtColor(proc,cv2.COLOR_GRAY2RGB))
        self.image_ctrl.SetBitmap(bmp)
        cmt = get_png_comment(self.dataset.get_path(self.current_index))
        self.meta_txt.SetValue(f"File: {os.path.basename(self.dataset.get_path(self.current_index))}\nIndex: {self.current_index}\nComment: {cmt}")
        self.panel.Layout()

    def apply_processing(self,img,opt,val):
        if opt=="Threshold >": _,o=cv2.threshold(img,val,255,cv2.THRESH_BINARY); return o
        if opt=="Threshold <": _,o=cv2.threshold(img,val,255,cv2.THRESH_BINARY_INV); return o
        if opt=="Sobel X": s=cv2.Sobel(img,cv2.CV_64F,1,0); s=np.abs(s); s=(s/s.max()*255).astype(np.uint8); return s
        if opt=="Sobel Y": s=cv2.Sobel(img,cv2.CV_64F,0,1); s=np.abs(s); s=(s/s.max()*255).astype(np.uint8); return s
        if opt=="Canny": return cv2.Canny(img,max(1,val-50),val+50)
        return img

    def getPathToPCAFile(self): 
            baseName = os.path.basename(self.dataset.folder)
            path = "%s/../%s.pca" % (self.dataset.folder,baseName)
            return path

    def on_load_pca(self, evt):
        if pca_module is None:
            wx.MessageBox("principleComponentAnalysis.py not found")
            return

        self.pca=pca_module.PCA(savedFile=self.getPathToPCAFile()) 
        #dlg=wx.FileDialog(self,"Open PCA",wildcard='*.pca;*.json',style=wx.FD_OPEN)
        #if dlg.ShowModal()==wx.ID_OK:
        #    try:
        #        self.pca=pca_module.PCA(savedFile=dlg.GetPath())
        #        wx.MessageBox("PCA loaded")
        #    except Exception as e:
        #        wx.MessageBox(str(e))
        #dlg.Destroy()

    def on_compute_embeddings(self, evt):
     if not self.dataset:
        wx.MessageBox("Load a dataset first", "Error")
        return

     def worker():
        try:
            n = len(self.dataset)
            if n == 0:
                return
            first = self.dataset.load_image(0, as_gray=True)
            D = first.size
            data = np.zeros((n, D), np.float32)
            for i in range(n):
                img = self.dataset.load_image(i, as_gray=True)
                if img is not None:
                    data[i, :] = img.flatten()
 
            baseName = os.path.basename(self.dataset.folder)

            # Create or use existing PCA
            if self.pca is None:
                print("Doing PCA Fitting")
                self.pca = pca_module.PCA(inputData=data)
                if not self.pca.ok():
                    self.pca.fit(data)
                print("Doing PCA Visualization")
                self.pca.visualize(data,saveToFile="%s.jpg"%self.getPathToPCAFile(),label=baseName,onlyScreePlotNDimensions=10)
                # ask user to save PCA file
                wx.CallAfter(self._save_pca_after_fit)
            else:
                # PCA already available — just transform
                pass

            # compute embeddings
            self.embeddings = self.pca.transform(data).real
            wx.CallAfter(wx.MessageBox, f"Embeddings shape: {self.embeddings.shape}", "Info")
        except Exception as e:
            traceback.print_exc()
            wx.CallAfter(wx.MessageBox, f"Error computing PCA/embeddings: {e}", "Error")

     threading.Thread(target=worker).start()


    def _save_pca_after_fit(self):
     """Prompt to save PCA model after fitting."""
     self.pca.save(self.getPathToPCAFile()) 
     #dlg = wx.FileDialog(self, "Save PCA model", wildcard="*.json", style=wx.FD_SAVE|wx.FD_OVERWRITE_PROMPT)
     #if dlg.ShowModal() == wx.ID_OK:
     #   fname = dlg.GetPath()
     #   try:
     #       self.pca.save(fname)
     #       wx.MessageBox(f"PCA saved to {fname}", "Info")
     #   except Exception as e:
     #       wx.MessageBox(f"Error saving PCA: {e}", "Error")
     #dlg.Destroy()
 

    def _compute_thread(self):
        try:
            n=len(self.dataset)
            f=self.dataset.load_image(0)
            D=f.size
            arr=[]
            for i in range(n):
                img=self.dataset.load_image(i)
                if img is None: continue
                arr.append(img.flatten().astype(np.float32))
            data=np.stack(arr)
            self.embeddings=self.pca.transform(data).real
            wx.CallAfter(wx.MessageBox,f"Embeddings shape: {self.embeddings.shape}")
        except Exception as e:
            traceback.print_exc()
            wx.CallAfter(wx.MessageBox,str(e))

    def on_save_embeddings(self, evt):
        if self.embeddings is None:
            wx.MessageBox("No embeddings to save")
            return
        dlg = wx.FileDialog(self, "Save embeddings", wildcard='*.npy;*.json', style=wx.FD_SAVE|wx.FD_OVERWRITE_PROMPT)
        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPath()
            try:
                if path.lower().endswith('.json'):
                    json.dump(self.embeddings.tolist(), open(path,'w'))
                else:
                    np.save(path, self.embeddings)
                wx.MessageBox("Embeddings saved")
            except Exception as e:
                wx.MessageBox(str(e))
        dlg.Destroy()

    def on_load_embeddings(self, evt):
        dlg = wx.FileDialog(self, "Load embeddings", wildcard='*.npy;*.json', style=wx.FD_OPEN)
        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPath()
            try:
                if path.lower().endswith('.json'):
                    self.embeddings = np.array(json.load(open(path)))
                else:
                    self.embeddings = np.load(path)
                wx.MessageBox(f"Embeddings loaded: {self.embeddings.shape}")
            except Exception as e:
                wx.MessageBox(str(e))
        dlg.Destroy()

    def on_knn_search(self, evt):
        if self.embeddings is None:
            wx.MessageBox("No embeddings")
            return
        print("Doing KNN Search")
        k=self.knn_k.GetValue()
        img=self.dataset.load_image(self.current_index,as_gray=True)
        #q=self.pca.transform(img.flatten().reshape(1,-1)).real
        q = self.pca.transform(img.astype(np.float32).flatten().reshape(1, -1)).real 
        d=np.linalg.norm(self.embeddings-q,axis=1)
        idxs=np.argsort(d)[:k+1]
        idxs=[int(i) for i in idxs if i!=self.current_index][:k]
        print("IDs : ",idxs)
        wx.CallAfter(self.show_knn_results,idxs)

    def show_knn_results(self,idxs):
        for c in self.results_panel.GetChildren(): c.Destroy()
        self.results_sizer.Clear(True)
        for i in idxs:
            im=self.dataset.load_image(i,as_gray=True)
            up=cv2.resize(im,(64,64))
            bmp=wx.Bitmap.FromBuffer(64,64,cv2.cvtColor(up,cv2.COLOR_GRAY2RGB))
            box=wx.BoxSizer(wx.VERTICAL)
            box.Add(wx.StaticBitmap(self.results_panel,bitmap=bmp))
            box.Add(wx.StaticText(self.results_panel,label=str(i)))
            self.results_sizer.Add(box,0,wx.ALL,4)
        self.results_panel.Layout()

if __name__=='__main__':

    start_dir = os.path.join(os.getcwd(), './')
    if len(sys.argv) > 1:
        start_dir = sys.argv[1] 

    app=wx.App(False)
    f=MainFrame(start_dir=start_dir)
    f.Show()
    app.MainLoop()

