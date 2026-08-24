#!/usr/bin/python3

""" 
Author : "Ammar Qammaz"
Copyright : "2025 Foundation of Research and Technology, Computer Science Department Greece, See license.txt"
License : "FORTH" 
"""

#pip install numpy opencv-python --user

import numpy as np
import os
import sys
import cv2
import json

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

# fs helpers are single-sourced in mga/core/read_data_annotator.py (Stage 3e of
# the wx_annotator refactor — these copies had drifted).
from mga.core.read_data_annotator import (checkIfFileExists, checkIfPathExists,
                                          checkIfPathIsDirectory,
                                          list_image_files)


class FolderStreamer():
  def __init__(self,
               path       = None,
               label      = "colorFrame_0_",
               width      = 0,
               height     = 0
              ):
      self.path        = path
      self.local_dir   = path
      self.label       = label
      self.frameNumber = 0
      #----------------------------------------------------
      self.width       = width
      self.height      = height
      #----------------------------------------------------
      self.should_stop = False
      #----------------------------------------------------
      self.metadata = None
      self.directoryList = None
      self.directoryListIndex = 0
      #----------------------------------------------------
      if (self.path is not None):
         self.loadNewDataset(path)
      #----------------------------------------------------

  def loadNewDataset(self,path):
      if (path!=""):
           pathIsDirectory = checkIfPathIsDirectory(path)
           if (pathIsDirectory):
             #----------------------------------------------------
             print("Loading image files from : ",path)
             self.path          = path
             self.local_dir     = path
             self.directoryList = list_image_files(path)
             self.directoryListIndex = 0
             #----------------------------------------------------
             if (checkIfFileExists(path)):
                with open(path) as json_data:
                   self.metadata = json.load(json_data)
      #----------------------------------------------------
  def current(self):
      return self.directoryListIndex

  def max(self):
      return len(self.directoryList)

  def next(self):
    print("Folder Stream Next..")
    if (self.directoryListIndex<len(self.directoryList)-1):
                   self.directoryListIndex = self.directoryListIndex + 1
    else:
                   self.directoryListIndex = 0

  def previous(self):
    print("Folder Stream Previous..")
    if (self.directoryListIndex>0):
                   self.directoryListIndex = self.directoryListIndex - 1
    else:
                   self.directoryListIndex = len(self.directoryList) - 1

  def select(self,item):
    print("Folder Stream Select..")
    self.directoryListIndex = item

  def getJSON(self):
    img_path = self.directoryList[self.directoryListIndex]
    stem = os.path.splitext(img_path)[0]

    candidates = [
        img_path + ".json",      # new style: image.ext.json
        stem + ".pnm.json",      # legacy
        stem + ".png.json",      # legacy
    ]

    for filepath in candidates:
        if checkIfFileExists(filepath):
            print("folderStream: There is a JSON file for item ", self.directoryListIndex, " -> ", filepath)
            return filepath

    print("folderStream: There is no JSON file for item ", self.directoryListIndex)
    return None

  def getImage(self):
    self.filepath = self.directoryList[self.directoryListIndex]
    print("Folder Stream item ",self.directoryListIndex," -> ",self.filepath)
    return self.filepath

  def getImageSimple(self):
    """Compatibility helper (alias for getImage)."""
    return self.getImage()

  def saveJSON(self):
    print("folderStream saveJSON (Doing nothing)..")
 
if __name__ == '__main__':
    print("Folder Stream tester..") 
    test = FolderStreamer(path="40-positive-class-a")
    test.getImage()
    test.getJSON()
    test.next()
    test.getImage()
    test.getJSON()
    test.next()
    test.getImage()
    test.getJSON()

