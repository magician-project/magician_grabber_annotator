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

"""
Check if a file exists
"""
def checkIfFileExists(filename):
    if (filename is None):  
       return False
    return os.path.isfile(filename) 

"""
Check if a path exists
"""
def checkIfPathExists(filename):
    if (filename is None):  
       return False
    return os.path.exists(filename) 

"""
Check if a path exists
"""
def checkIfPathIsDirectory(filename):
    if (filename is None):  
       return False
    return os.path.isdir(filename) 


def list_image_files(directory):
    """
    Retrieve a list of all files in the specified directory.

    Parameters:
    - directory (str): The path to the directory.

    Returns:
    - files (list): A list of file names in the directory.
    """

    image_extensions = ['.png', '.pnm', '.jpg', '.jpeg']
    image_files = []

    try:
        # Iterate over all files and directories in the specified directory
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)

            # Check if it's a file (not a directory) and has a valid image extension
            if os.path.isfile(filepath) and any(filename.lower().endswith(ext) for ext in image_extensions):
               if "foreground.png" in filepath:
                   print("Omitting ",filepath," since it is a foreground file!")
               else:
                   image_files.append(filepath) 

    except OSError as e:
        print(f"Error reading directory '{directory}': {e}")
    
    image_files.sort() # Always sort files 

    return image_files


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

