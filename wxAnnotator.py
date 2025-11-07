#!/usr/bin/python3
""" 
Author : "Ammar Qammaz"
Copyright : "2025 Foundation of Research and Technology, Computer Science Department Greece"
License : "FORTH" 
"""


"""

In a machine using Ubuntu 22.04.5 with Python 3.10 

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


"""
Configurations in one central place
"""

version         = "0.28"
useSAM          = False
useClassifier   = True #<- Switch classifier off
combineChannels = True
options         = ["Unknown", "Material Defect", "Positive Dent", "Negative Dent", "Deformation", "Seal", "Welding", "Suspicious", "Clean"]
severities      = ["Class A","Class B","Class C"]
directions      = ["Unknown","Bottom Left","Top Left","Top","Top Right", "Bottom Right", "Bottom"]
processors      = ["PolarizationRGB1","PolarizationRGB2","PolarizationRGB3", "Polarization_0_degree","Polarization_45_degree","Polarization_90_degree", "Polarization_135_degree", "Sobel","Visible","SAM"]


#classifier_relative_directory = "../classifier" #Old Name
classifier_relative_directory = "../magician_vision_classifier"
classifier_model_path         = "%s/last.pth"  % classifier_relative_directory
classifier_cfg_path           = "%s/last.json" % classifier_relative_directory


"""
from wxAcquisition import CameraSettingsDialog
"""

# Import the wxScrollBar module
import wx.lib.newevent
import sys
import os

from folderStream import FolderStreamer

# Add this line at the beginning of the file to define a new event
ScrollEvent, EVT_SCROLL_EVENT = wx.lib.newevent.NewCommandEvent()


#-------------------------------------------------------------------------------
# Make Classifier completely seperatable from the rest of the codebase
#-------------------------------------------------------------------------------
if useClassifier:
  parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), classifier_relative_directory))
  sys.path.append(parent_path)
  from liveClassifierTorch import ClassifierPnm
  from EnsembleClassifier  import EnsembleClassifierPnm
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


#-------------------------------------------------------------------------------
# Make SAM Processor completely seperatable from the rest of the codebase
#-------------------------------------------------------------------------------
if useSAM:
  from sam import SAMProcessor
else: 
  class SAMProcessorFoo:
    def __init__(self, sam_checkpoint, model_type, device="cuda"):
        self.masks = None
        self.image = None
        self.foregroundMask  = None
        self.foregroundImage = None
    def save_mask(self,path,mask):
        pass
#-------------------------------------------------------------------------------

"""
Do a CRC on data to prevent data corruption training errors
"""
def get_md5(file_path):
    # Construct the command
    command = f"md5sum {file_path}"
    
    # Execute the command and capture the output
    output = os.popen(command).read()
    
    # Parse the output to extract the MD5 hash
    md5_hash = output.split()[0]
    
    return md5_hash

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

def debayerPolarImage(image): 
 # Split the A, B, C, and D values into separate monochrome images
 polarization_90_deg   = image[0::2, 0::2]
 polarization_45_deg   = image[0::2, 1::2]
 polarization_135_deg  = image[1::2, 0::2]
 polarization_0_deg    = image[1::2, 1::2]
 return polarization_0_deg,polarization_45_deg,polarization_90_deg,polarization_135_deg      

def loadMoreClasses(filename,classes_dict):
    with open("%s.json"%filename) as json_data:
        data          = json.load(json_data)
        point_clicks  = data.get("pointClicks", [])
        point_classes = data.get("pointClasses", [])
        for cl in point_classes:
           #print("Add `",cl,"` class ")
           classes_dict[cl]=True 
    return classes_dict 

def detect_sobel_edges(image):
    """
    Detect Sobel edges in a 4-channel polarized image and return them in a 3-channel image.

    Parameters:
        image (numpy.ndarray): Input image with 4 channels (polarizations: 0°, 45°, 90°, 135°).

    Returns:
        numpy.ndarray: 3-channel image containing Sobel edge detections.
    """
    # Split the 4-channel image into individual polarization channels
    polar_0, polar_45, polar_90, polar_135 = debayerPolarImage(image)# cv2.split(image)

    # Initialize an empty list to store edge-detected channels
    edges = []

    # Apply Sobel edge detection for each channel
    for channel in [polar_0, polar_45, polar_90, polar_135]:
        # Compute Sobel gradients in x and y directions
        grad_x = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=3)

        # Compute gradient magnitude
        magnitude = cv2.magnitude(grad_x, grad_y)

        # Normalize the magnitude to the range [0, 255] and convert to uint8
        magnitude_normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Append the result to the edges list
        edges.append(magnitude_normalized)

    # Combine the first three edge-detected channels into a 3-channel image
    # (Choose polarizations 0°, 45°, 90° for the output channels)
    result = cv2.merge(edges[:4])

    return result



def tenengrad_focus_measure(image, ksize=3):
    """
    Compute the Tenengrad focus measure of an image.
    
    Parameters:
        image (numpy.ndarray): Input image (BGR or grayscale).
        ksize (int): Kernel size for Sobel operator (must be 1, 3, 5, or 7).
    
    Returns:
        float: Tenengrad focus measure value.
    """
    # Convert to grayscale if image is colored
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Apply Sobel operator in X and Y directions
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    
    # Compute gradient magnitude
    gxy = gx**2 + gy**2
    
    # Return mean gradient magnitude (Tenengrad measure)
    return np.mean(gxy)


def determine_intensity_region(image, threshold=0.1):
    """
    Determines the region of the image with the highest intensity values.

    Parameters:
        image (numpy.ndarray): A 4-channel image (H, W, 4) with intensity values.
        threshold (float): A value between 0 and 1 to decide if intensity changes are significant.

    Returns:
        str: One of "Unknown", "Bottom Left", "Top Left", "Top", "Top Right", "Bottom Right", or "Bottom".
    """
    #if image.shape[-1] != 4:
    #    raise ValueError("Input image must have 4 channels.")

    # Convert image to grayscale by summing up all channels
    gray_image = np.sum(image, axis=-1)

    # Get image dimensions
    height, width = gray_image.shape

    # Define regions
    top_left     = gray_image[:height//2, :width//2]
    top          = gray_image[:height//2, width//4:(3*width)//4]
    top_right    = gray_image[:height//2, width//2:]
    bottom_left  = gray_image[height//2:, :width//2]
    bottom       = gray_image[height//2:, width//4:(3*width)//4]
    bottom_right = gray_image[height//2:, width//2:]

    # Compute average intensities for each region
    regions = {
        "Top Left": np.mean(top_left),
        "Top": np.mean(top),
        "Top Right": np.mean(top_right),
        "Bottom Left": np.mean(bottom_left),
        "Bottom": np.mean(bottom),
        "Bottom Right": np.mean(bottom_right)
    }

    print("determine_intensity_region: ",regions)

    # Find the region with the highest intensity
    max_region = max(regions, key=regions.get)
    max_value  = regions[max_region]

    # Calculate the overall mean intensity
    overall_mean = np.mean(gray_image)

    # If the difference between the highest intensity and the mean is below the threshold, return "Unknown"
    if (max_value - overall_mean) / overall_mean < threshold:
        return "Unknown"

    return max_region



def adjust_contrast(image: np.ndarray, factor: float):
    """
    Adjusts the contrast of an RGB image.
    
    Parameters:
        image (np.ndarray): Input image as a NumPy array with shape (H, W, 3).
        factor (float): Contrast adjustment factor (1.0 = no change, >1 increases contrast, <1 decreases contrast).
    
    Returns:
        np.ndarray: Contrast-adjusted image.
    """
    # Convert to float for precision
    img_float = image.astype(np.float32) / 255.0
    
    # Compute mean intensity per channel
    mean      = np.mean(img_float, axis=(0, 1), keepdims=True)
    
    # Apply contrast adjustment
    adjusted  = mean + factor * (img_float - mean)
    
    # Clip values to valid range and convert back to uint8
    adjusted = np.clip(adjusted * 255, 0, 255).astype(np.uint8)
    
    return adjusted


def convertRGBCVMATToRGB(rgb_image,brightness=0,contrast=0):
    brightnessValue = 10* brightness
    contrastValue   = 1.0 + contrast/10
    rgb_image = adjust_contrast(rgb_image,contrastValue)
    return rgb_image


def convertPolarCVMATToRGB(image,way=0,brightness=0,contrast=0):
    if image is None:
        print("Error: Unable to read the image.")
        return None

    height, width, channels = image.shape
    #if channels == 3: 
    #    print("Casting RGB image as monochrome")
    #    image = image[:,:,0]
    image = image[:,:,0]

    # Split into polarization images
    #from readData import debayerPolarImage
    polarization_0_deg, polarization_45_deg, polarization_90_deg, polarization_135_deg = debayerPolarImage(image)

    # Create an RGB image
    rgb_image = np.zeros((int(height/2),int(width/2), 3), dtype=np.uint8)

    brightnessValue = 10* brightness
    contrastValue   = 1.0 + contrast/10

    # Assign each polarization image to a specific channel
    if (way==0):
      rgb_image[:, :, 0] = np.clip(polarization_0_deg.astype(np.float32)   + brightnessValue, 0, 255)
      rgb_image[:, :, 1] = np.clip(polarization_45_deg.astype(np.float32)  + brightnessValue, 0, 255)
      rgb_image[:, :, 2] = np.clip(polarization_90_deg.astype(np.float32)  + brightnessValue, 0, 255)
    elif (way==1):
      rgb_image[:, :, 0] = np.clip(polarization_45_deg.astype(np.float32)  + brightnessValue, 0, 255)
      rgb_image[:, :, 1] = np.clip(polarization_90_deg.astype(np.float32)  + brightnessValue, 0, 255)
      rgb_image[:, :, 2] = np.clip(polarization_135_deg.astype(np.float32) + brightnessValue, 0, 255)
    elif (way==2):
      rgb_image[:, :, 0] = np.clip( ( polarization_0_deg.astype(np.float32) +  polarization_45_deg.astype(np.float32) ) / 2   + brightnessValue, 0, 255)
      rgb_image[:, :, 1] = np.clip(polarization_45_deg  + brightnessValue, 0, 255)
      rgb_image[:, :, 2] = np.clip( ( polarization_90_deg.astype(np.float32) + polarization_135_deg.astype(np.float32) ) / 2  + brightnessValue, 0, 255)
    elif (way==4):
      #this needs some care so that values are not clipped
      sumMat = (polarization_0_deg.astype(np.float32) + 
                  polarization_45_deg.astype(np.float32) + 
                  polarization_90_deg.astype(np.float32) + 
                  polarization_135_deg.astype(np.float32)) / 4
      rgb_image[:, :, 0] = np.clip(sumMat.astype(np.float32) + brightnessValue, 0, 255)
      rgb_image[:, :, 1] = np.clip(sumMat.astype(np.float32) + brightnessValue, 0, 255)
      rgb_image[:, :, 2] = np.clip(sumMat.astype(np.float32) + brightnessValue, 0, 255)
    elif (way==5):
      rgb_image[:, :, 0] = np.clip(polarization_0_deg.astype(np.float32) + brightnessValue, 0, 255)
      rgb_image[:, :, 1] = np.clip(polarization_0_deg.astype(np.float32) + brightnessValue, 0, 255)
      rgb_image[:, :, 2] = np.clip(polarization_0_deg.astype(np.float32) + brightnessValue, 0, 255)
    elif (way==6):
      rgb_image[:, :, 0] = np.clip(polarization_45_deg.astype(np.float32) + brightnessValue, 0, 255)
      rgb_image[:, :, 1] = np.clip(polarization_45_deg.astype(np.float32) + brightnessValue, 0, 255)
      rgb_image[:, :, 2] = np.clip(polarization_45_deg.astype(np.float32) + brightnessValue, 0, 255)
    elif (way==7):
      rgb_image[:, :, 0] = np.clip(polarization_90_deg.astype(np.float32) + brightnessValue, 0, 255)
      rgb_image[:, :, 1] = np.clip(polarization_90_deg.astype(np.float32) + brightnessValue, 0, 255)
      rgb_image[:, :, 2] = np.clip(polarization_90_deg.astype(np.float32) + brightnessValue, 0, 255)
    elif (way==8):
      rgb_image[:, :, 0] = np.clip(polarization_135_deg.astype(np.float32) + brightnessValue, 0, 255)
      rgb_image[:, :, 1] = np.clip(polarization_135_deg.astype(np.float32) + brightnessValue, 0, 255)
      rgb_image[:, :, 2] = np.clip(polarization_135_deg.astype(np.float32) + brightnessValue, 0, 255)

    if (contrastValue!=0.0):
           rgb_image = adjust_contrast(rgb_image,contrastValue)

    return rgb_image

def slowPC():
    import socket
    try:
        hostname = socket.gethostname()
        if (hostname=="cvrldemo"):
          return True 
    except:
        return "Unable to retrieve hostname"
        return True
    return False



class UploadDialog(wx.Dialog):
    def __init__(self, parent, zip_path, dataset, credentials="server.json"):
        super().__init__(parent, title="Upload Annotations", size=(350, 200))
        self.zip_path = zip_path  # path to the zip file
        self.dataset  = dataset
        self.credentials = credentials

        # Try to load saved credentials
        saved_user, saved_pwd = self.load_credentials()

        vbox = wx.BoxSizer(wx.VERTICAL)

        # Username
        hbox1 = wx.BoxSizer(wx.HORIZONTAL)
        hbox1.Add(wx.StaticText(self, label="Username:"), 0, wx.ALL | wx.CENTER, 5)
        self.username = wx.TextCtrl(self, value=saved_user)
        hbox1.Add(self.username, 1, wx.ALL | wx.EXPAND, 5)
        vbox.Add(hbox1, 0, wx.EXPAND)

        # Password
        hbox2 = wx.BoxSizer(wx.HORIZONTAL)
        hbox2.Add(wx.StaticText(self, label="Password:"), 0, wx.ALL | wx.CENTER, 5)
        self.password = wx.TextCtrl(self, style=wx.TE_PASSWORD, value=saved_pwd)
        hbox2.Add(self.password, 1, wx.ALL | wx.EXPAND, 5)
        vbox.Add(hbox2, 0, wx.EXPAND)

        vbox.Add(wx.StaticText(self, label=" Contact ammarkov@ics.forth.gr for a new account"), 0, wx.EXPAND)

        # Buttons
        btns = self.CreateSeparatedButtonSizer(wx.OK | wx.CANCEL)
        vbox.Add(btns, 0, wx.EXPAND | wx.ALL, 10)

        self.SetSizer(vbox)

        # Override Upload (OK) behavior
        self.Bind(wx.EVT_BUTTON, self.onUpload, id=wx.ID_OK)

    def load_credentials(self):
        """Load username and password from config file if it exists."""
        if os.path.exists(self.credentials):
            try:
                with open(self.credentials, "r") as f:
                    data = json.load(f)
                    return data.get("username", ""), data.get("password", "")
            except Exception:
                pass
        return "", ""  # defaults

    def save_credentials(self, username, password):
        """Save username and password to config file."""
        try:
            with open(self.credentials, "w") as f:
                json.dump({"username": username, "password": password}, f)
        except Exception as e:
            wx.MessageBox(f"Failed to save credentials: {e}", "Warning", wx.OK | wx.ICON_WARNING)

    def onUpload(self, event):
        user     = self.username.GetValue().strip()
        pwd      = self.password.GetValue().strip()
        dataset  = self.dataset

        if not user or not pwd:
            wx.MessageBox("Please enter both username and password.", "Error", wx.OK | wx.ICON_ERROR)
            return  # don’t close yet

        # Command for curl file upload
        url = "http://ammar.gr/magician/upload.php"
        cmd = [
            "curl",
            "-s",  # silent mode
            "-F", f"username={user}",
            "-F", f"password={pwd}",
            "-F", f"dataset={dataset}",
            "-F", f"file=@{self.zip_path}",  # attach file
            url
        ]

        try:
            import subprocess
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            wx.MessageBox(f"Upload successful!\nServer response:\n{result.stdout}", 
                          "Success", wx.OK | wx.ICON_INFORMATION)

            # Save credentials only if successful
            self.save_credentials(user, pwd)

            self.EndModal(wx.ID_OK)
        except subprocess.CalledProcessError as e:
            wx.MessageBox(f"Upload failed!\n{e.stderr}", "Error", wx.OK | wx.ICON_ERROR)


class BatchProcessDialog(wx.Dialog):
    def __init__(self, parent, folderStreamer):
        super().__init__(parent, title="Batch Download of Dataset", size=(400, 200))
        self.folderStreamer = folderStreamer
        self.stop_requested = False  # flag for cancellation

        vbox = wx.BoxSizer(wx.VERTICAL)

        # Spin control for number of iterations
        hbox1 = wx.BoxSizer(wx.HORIZONTAL)
        hbox1.Add(wx.StaticText(self, label="Number of files to download:"), 0, wx.ALL | wx.CENTER, 5)
        self.spin = wx.SpinCtrl(self, min=1, max=100000, initial=self.folderStreamer.max())
        hbox1.Add(self.spin, 1, wx.ALL | wx.CENTER, 5)
        vbox.Add(hbox1, 0, wx.EXPAND)

        # Progress bar
        self.gauge = wx.Gauge(self, range=100, size=(250, 25))
        vbox.Add(self.gauge, 0, wx.ALL | wx.EXPAND, 10)

        # ETA label
        self.eta_label = wx.StaticText(self, label="Estimated time remaining: --")
        vbox.Add(self.eta_label, 0, wx.ALL | wx.CENTER, 5)

        # Buttons
        btns = self.CreateSeparatedButtonSizer(wx.OK | wx.CANCEL)
        vbox.Add(btns, 0, wx.EXPAND | wx.ALL, 10)

        self.SetSizer(vbox)

        # Get references to buttons
        self.okBtn = self.FindWindowById(wx.ID_OK)
        self.cancelBtn = self.FindWindowById(wx.ID_CANCEL)

        # Override button behavior
        self.okBtn.Bind(wx.EVT_BUTTON, self.onStart)
        self.cancelBtn.Bind(wx.EVT_BUTTON, self.onCancel)

    def onStart(self, event):
        count = self.spin.GetValue()
        self.okBtn.Disable()  # prevent starting twice
        self.stop_requested = False
        threading.Thread(target=self.runBatch, args=(count,), daemon=True).start()

    def onCancel(self, event):
        self.stop_requested = True  # signal thread to stop
        self.cancelBtn.Disable()    # prevent spamming cancel button

    def runBatch(self, count):
        times = []
        for i in range(count):
            if self.stop_requested:
                wx.CallAfter(self.eta_label.SetLabel, "Cancelled by user.")
                break

            start = time.perf_counter()

            self.folderStreamer.next()
            self.folderStreamer.getJSON()
            self.folderStreamer.getImageSimple()

            elapsed = time.perf_counter() - start
            times.append(elapsed)

            avg_time = sum(times) / len(times)
            remaining = avg_time * (count - i - 1) / 60

            # Update UI safely
            wx.CallAfter(self.gauge.SetValue, int((i + 1) / count * 100))
            wx.CallAfter(self.eta_label.SetLabel, f"Estimated time remaining: {remaining:.1f} mins")

        wx.CallAfter(self.EndModal, wx.ID_OK)



class MagnifierFrame(wx.Frame):
    def __init__(self, parent, zoom=3, size=(300, 300), win_size=(400, 400)):
        super().__init__(parent, title="Magnifier", size=win_size)
        self.panel = wx.Panel(self)
        self.zoom = zoom
        self.size = size
        self.original_img = None
        self.last_x, self.last_y = 0, 0  # store last cursor pos
        self.show_crosshair = True

        # Layout: image + controls
        vbox = wx.BoxSizer(wx.VERTICAL)

        # Image display
        self.imageCtrl = wx.StaticBitmap(self.panel, size=size)
        vbox.Add(self.imageCtrl, 1, wx.EXPAND | wx.ALL, 5)

        # Zoom controls row
        hbox = wx.BoxSizer(wx.HORIZONTAL)

        self.btnZoomOut = wx.Button(self.panel, label="–")
        self.btnZoomIn = wx.Button(self.panel, label="+")
        hbox.Add(self.btnZoomOut, 0, wx.ALL, 5)
        hbox.Add(self.btnZoomIn, 0, wx.ALL, 5)

        # Slider
        self.slider = wx.Slider(self.panel, value=self.zoom, minValue=1, maxValue=20,
                                style=wx.SL_HORIZONTAL | wx.SL_LABELS)
        hbox.Add(self.slider, 1, wx.ALL | wx.EXPAND, 5)

        # Text field for zoom value
        self.txtZoom = wx.TextCtrl(self.panel, value=str(self.zoom), size=(50, -1),
                                   style=wx.TE_PROCESS_ENTER)
        hbox.Add(self.txtZoom, 0, wx.ALL, 5)

        # Checkbox for crosshair
        self.crosshairCheckbox = wx.CheckBox(self.panel, label="Show Crosshair")
        self.crosshairCheckbox.SetValue(self.show_crosshair)
        hbox.Add(self.crosshairCheckbox, 0, wx.ALL | wx.CENTER, 5)


        vbox.Add(hbox, 0, wx.EXPAND)

        self.panel.SetSizer(vbox)

        # Bind events
        self.btnZoomIn.Bind(wx.EVT_BUTTON, self.onZoomIn)
        self.btnZoomOut.Bind(wx.EVT_BUTTON, self.onZoomOut)
        self.slider.Bind(wx.EVT_SLIDER, self.onSliderChange)
        self.txtZoom.Bind(wx.EVT_TEXT_ENTER, self.onTextEnter)
        self.crosshairCheckbox.Bind(wx.EVT_CHECKBOX, self.onCrosshairToggle)

    def setImage(self, img):
        """Store original image (as wx.Image) for magnification."""
        self.original_img = img

    def updateMagnifier(self, x, y):
        if not self.original_img:
            return

        self.last_x, self.last_y = x, y  # store last position

        half_w = self.size[0] // (2 * self.zoom)
        half_h = self.size[1] // (2 * self.zoom)

        # Crop region around cursor
        crop_x1 = max(0, x - half_w)
        crop_y1 = max(0, y - half_h)
        crop_x2 = min(self.original_img.GetWidth(),  x + half_w)
        crop_y2 = min(self.original_img.GetHeight(), y + half_h)

        sub_img = self.original_img.GetSubImage(
            (crop_x1, crop_y1, crop_x2 - crop_x1, crop_y2 - crop_y1)
        )
        sub_img = sub_img.Scale(self.size[0], self.size[1], wx.IMAGE_QUALITY_HIGH)

        # Draw crosshair if enabled
        if self.show_crosshair:
            dc = wx.MemoryDC()
            bmp = wx.Bitmap(sub_img)
            dc.SelectObject(bmp)
            w, h = bmp.GetWidth(), bmp.GetHeight()
            pen = wx.Pen(wx.Colour(255, 0, 0), 1)  # red crosshair
            dc.SetPen(pen)
            # Horizontal line
            dc.DrawLine(0, h//2, w, h//2)
            # Vertical line
            dc.DrawLine(w//2, 0, w//2, h)
            dc.SelectObject(wx.NullBitmap)
            self.imageCtrl.SetBitmap(bmp)
        else:
            self.imageCtrl.SetBitmap(wx.Bitmap(sub_img))

        self.panel.Layout()

    def refreshZoom(self):
        """Update slider + text + refresh view."""
        self.slider.SetValue(self.zoom)
        self.txtZoom.SetValue(str(self.zoom))
        self.updateMagnifier(self.last_x, self.last_y)

    def onZoomIn(self, event):
        self.zoom = min(self.zoom + 1, 20)
        self.refreshZoom()

    def onZoomOut(self, event):
        self.zoom = max(self.zoom - 1, 1)
        self.refreshZoom()

    def onSliderChange(self, event):
        self.zoom = self.slider.GetValue()
        self.refreshZoom()

    def onTextEnter(self, event):
        try:
            val = int(self.txtZoom.GetValue())
            if 1 <= val <= 20:
                self.zoom = val
                self.refreshZoom()
        except ValueError:
            pass  # ignore invalid input

    def onCrosshairToggle(self, event):
        self.show_crosshair = self.crosshairCheckbox.GetValue()
        self.updateMagnifier(self.last_x, self.last_y)



class PhotoCtrl(wx.App):
   def __init__(self, redirect=False, filename=None):
        if (useSAM):
          if (slowPC):
            self.sam_processor = SAMProcessor(sam_checkpoint="sam_vit_b_01ec64.pth", model_type="vit_b", device="cpu")
          else:
            self.sam_processor = SAMProcessor(sam_checkpoint="sam_vit_l_0b3195.pth", model_type="vit_l", device="cuda")
        else:
            self.sam_processor = SAMProcessorFoo(sam_checkpoint="foo.pth", model_type="vit_l", device="cuda")

        self.ClassifierPnm = ClassifierPnm(model_path=classifier_model_path,cfg_path=classifier_cfg_path)
        self.EnsembleClassifierPnm = EnsembleClassifierPnm(
                                                            initial_model_cfg = ("../magician_vision_classifier/binary_custom.pth","../magician_vision_classifier/binary_custom.json"),
                                                            model_cfg_list=[("../magician_vision_classifier/allclass_custom.pth","../magician_vision_classifier/allclass_custom.json"),
                                                                            ("../magician_vision_classifier/allclass_resnet18.pth","../magician_vision_classifier/allclass_resnet18.json"),
                                                                            ("../magician_vision_classifier/allclass_resnext50.pth","../magician_vision_classifier/allclass_resnext50.json"),
                                                                            ("../magician_vision_classifier/allclass_efficientnet_v2_s.pth","../magician_vision_classifier/allclass_efficientnet_v2_s.json"),
                                                                            ("../magician_vision_classifier/allclass_convnext_tiny.pth","../magician_vision_classifier/allclass_convnext_tiny.json")])
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
         
        windowTitle = 'Segmentation Control Annotator Tool v%s'%version
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

        self.viewedImageFullWidth  = 0
        self.viewedImageFullHeight = 0
        self.viewedImageViewWidth  = 0
        self.viewedImageViewHeight = 0           
        self.processingWay = 0
        self.brightness_offset = 0
        self.contrast_offset = 0
        self.scrollStep  = 10

        self.local_base_path = "./"
        self.maintainPoints = False  # Initial state
        self.controlsData = []

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


   def createWidgets(self):
    # ----- Menus (unchanged) -------------------------------------------------
    menuBar = wx.MenuBar()

    fileMenu = wx.Menu()
    itemOpen    = fileMenu.Append(wx.ID_OPEN, "&Open Image", "Open an image file")
    itemOpenDir = fileMenu.Append(wx.ID_OPEN, "Open &Directory", "Open a directory")
    itemOpenNet = fileMenu.Append(wx.ID_OPEN, "Open &Network", "Open network server")
    itemUpload  = fileMenu.Append(wx.ID_UP, "Upload &Annotations", "Upload annotations to server")
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
    itemMagnify       = toolsMenu.Append(wx.ID_ZOOM_IN, "&Magnifier", "Magnifier")
    itemCreateDataset = toolsMenu.Append(wx.ID_EDIT, "&Create Dataset", "Create Dataset")
    itemTileExplorer  = toolsMenu.Append(wx.ID_FIND, "&Tile Explorer", "Tile Explorer")
    itemStreamer      = toolsMenu.Append(wx.ID_FORWARD, "&Stream To Shared Memory", "Stream To Shared Memory")
    self.Bind(wx.EVT_MENU, self.onOpenMagnifier,itemMagnify)
    self.Bind(wx.EVT_MENU, self.onCreateDataset,itemCreateDataset)
    self.Bind(wx.EVT_MENU, self.onTileExplorer,itemTileExplorer)
    self.Bind(wx.EVT_MENU, self.onStreamer,itemStreamer)
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
    self.rescanBtn = wx.Button(self.panel, label='Rescan')
    self.rescanBtn.Bind(wx.EVT_BUTTON, self.onRescan)

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
    self.nextBtn = wx.Button(self.panel, label='>')
    self.nextBtn.Bind(wx.EVT_BUTTON, self.onNext)
    self.cameraSettingsBtn = wx.Button(self.panel, label='Camera')
    self.cameraSettingsBtn.Bind(wx.EVT_BUTTON, self.onCameraSettings)

    global processors
    self.ProcessorComboBox = wx.ComboBox(self.panel, choices=processors, style=wx.CB_DROPDOWN)
    self.ProcessorComboBox.Bind(wx.EVT_COMBOBOX, self.onProcessorComboBoxSelect)
    self.ProcessorComboBox.SetValue(processors[0])

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
    controlsPanel = wx.Panel(self.rightBook)
    self._buildControlsTab(controlsPanel)
    self.rightBook.AddPage(controlsPanel, "Sensor")



    # Add notebook to the right side
    self.sizer.Add(self.rightBook, 1, wx.ALL | wx.EXPAND, 5)

    # Add top row to main
    self.mainSizer.Add(self.sizer, 1, wx.ALL | wx.EXPAND, 5)

    # Under-image controls row
    self.underImage.Add(self.prevBtn, 0, wx.ALL, 5)
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

    # Action buttons
    self.autoBtn = wx.Button(parent, label='Auto')
    self.autoBtn.Bind(wx.EVT_BUTTON, self.onAuto)
    self.saveBtn = wx.Button(parent, label='Save')
    self.saveBtn.Bind(wx.EVT_BUTTON, self.onSave)
    self.deleteMetadataBtn = wx.Button(parent, label='Delete')
    self.deleteMetadataBtn.Bind(wx.EVT_BUTTON, self.ondeleteMetadata)

    comboButtons = wx.BoxSizer(wx.HORIZONTAL)
    comboButtons.Add(self.autoBtn, 0, wx.ALL, 5)
    comboButtons.Add(self.saveBtn, 0, wx.ALL, 5)
    comboButtons.Add(self.deleteMetadataBtn, 0, wx.ALL, 5)

    # Checkboxes (up to Guess lighting direction)
    self.maintainPointsCheckbox = wx.CheckBox(parent, label="Maintain Points for next image")
    self.maintainPoints = False  # Initial state
    self.maintainPointsCheckbox.SetValue(self.maintainPoints)

    self.incrementFrameAfterAnAdditionCheckbox = wx.CheckBox(parent, label="Increment frame after defect annotation")        
    self.incrementFrameAfterAnAddition=True
    self.incrementFrameAfterAnAdditionCheckbox.SetValue(self.incrementFrameAfterAnAddition)
    self.guessLightingCheckbox = wx.CheckBox(parent, label="Guess lighting direction")
    self.guessLightingCheckbox.SetValue(True)

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
    s.Add(self.removePointBtn, 0, wx.ALL, 5)

    s.Add(comboButtons, 0, wx.ALL, 5)

    s.Add(self.maintainPointsCheckbox, 0, wx.ALL, 5)
    s.Add(self.incrementFrameAfterAnAdditionCheckbox, 0, wx.ALL, 5)
    s.Add(self.guessLightingCheckbox, 0, wx.ALL, 5)

    parent.SetSizer(s)



   def _buildClassifierTab(self, parent):
    """Builds the Classifier tab with model select, threshold, majority voting,
       tile size (4..128), and two-stage classification toggle."""
    s = wx.BoxSizer(wx.VERTICAL)

    # --- 1. Get available models from directory ---
    model_dir = classifier_relative_directory
    available_models = self.ClassifierPnm.model_scan(model_dir)
    if not available_models:
        available_models = ["Default"]

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
        if useClassifier:
            success = self.ClassifierPnm.reload_model(model_dir, model_name)
            if success:
                #wx.MessageBox(f"Successfully reloaded model: {model_name}", "Model Reloaded", wx.OK | wx.ICON_INFORMATION)
                print(f"Successfully reloaded model: {model_name}")
            else:
                wx.MessageBox(f"Failed to reload model: {model_name}", "Error", wx.OK | wx.ICON_ERROR)
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

    # --- 7. Two-stage classification checkbox ---
    self.classifierTwoStage = wx.CheckBox(parent, label="Enable two-stage classification")
    self.parallellTwoStage  = wx.CheckBox(parent, label="Two-stage parallelism (VRAM intensive)")

    # --- 8. "Use NN Classifier" checkbox ---
    self.useClassifierCheckbox = wx.CheckBox(parent, label="Use NN Classifier")
    self.useClassifierCheckbox.SetValue(True)

    # --- 9. Layout ---
    s.Add(modelRow, 0, wx.ALL | wx.EXPAND, 10)
    s.Add(thrRow, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM | wx.EXPAND, 10)
    s.Add(self.classifierMajorityVoting, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)
    s.Add(tileRow, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM | wx.EXPAND, 10)
    s.Add(self.classifierTwoStage, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)
    s.Add(self.parallellTwoStage, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)
    s.Add(self.useClassifierCheckbox, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)


    self.classifierInfo = wx.StaticText(parent, label="No classifier run yet.")
    s.Add(self.classifierInfo, 0, wx.ALL | wx.EXPAND, 5)


    s.AddStretchSpacer(1)

    parent.SetSizer(s)

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

    # --- CSV Info ---
    self.csvInfo = wx.StaticText(parent, label="No CSV loaded.")
    s.Add(self.csvInfo, 0, wx.ALL | wx.EXPAND, 5)

    parent.SetSizer(s)

 
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

   def onAuto(self, event): 
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


      print(f"Total new clean tiles: {len(new_points)}")

      if (self.incrementFrameAfterAnAdditionCheckbox.GetValue()):
          print(f"Auto incrementing due to checkbox")
          self.onNext(event)


      return new_points, new_classes
        

   def onSave(self, event):
        print("Save")
        if (len(self.regions_of_interest)>0):
           self.sam_processor.save_mask("%s_foreground.png"%self.filepath , self.sam_processor.foregroundMask )
        else:
           print("Not producing a foreground file since no foreground was selected") 

        allData = dict()
        allData["width"]   = self.width #self.sam_processor.image.shape[1]
        allData["height"]  = self.height #self.sam_processor.image.shape[0] 
        allData["md5hash"] = self.filehash

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

        if (self.lightComboBox.GetValue()!="Unknown"):
              allData["lightDirection"] = self.lightComboBox.GetValue()

        with open("%s.json" % self.filepath, "w") as outfile:
            json.dump(allData, outfile, sort_keys=False)

        self.folderStreamer.saveJSON()


   def cleanThisFrameMetaData(self):
               self.pointList.Clear()
               self.regionList.Clear()
               self.points_classes      = []
               self.points_severities   = []
               self.regions_of_interest = []
               self.points_of_interest  = []
               self.lightComboBox.SetValue("Unknown")


   def onProcessNewImageSample(self,filepath):
           if (self.maintainPointsCheckbox.GetValue()):
               print("Maintaining previous point lists")
           else:
               self.cleanThisFrameMetaData()

           #if (checkIfFileExists("%s.json"%filepath)):
           #    print("There are saved data that need to be restored here")
           #    self.restoreFromJSON("%s.json" % filepath)
           jsonPath = self.folderStreamer.getJSON()
           if (checkIfFileExists(jsonPath)):
               print("There are saved data that need to be restored here")
               self.restoreFromJSON(jsonPath)
           

           if hasattr(self, 'controlsData'):
                   frame_idx = self.scrollBar.GetValue()
                   if 0 <= frame_idx < len(self.controlsData):
                       self.updateControlsTab(self.controlsData[frame_idx])


           #self.filehash = get_md5(filepath) 
           #img = wx.Image(self.filepath, wx.BITMAP_TYPE_ANY)
           #img = self.rescaleBitmap(img)
           #self.imageCtrl.SetBitmap(wx.Bitmap(img))

           # Process the image with SAM and update the second StaticBitmap
           global combineChannels
           imgCV  = cv2.imread(filepath) #,cv2.IMREAD_UNCHANGED
           imgPNM = cv2.imread(filepath, cv2.IMREAD_UNCHANGED) #This is to be used without changes by the classifier

           if ((imgCV is None) or (imgPNM is None)):
                  print("Could not load ",filepath)
                  return 

           print("Raw image dims for ",filepath," ",imgCV.shape)
           self.viewedImageFullWidth  = imgCV.shape[1]
           self.viewedImageFullHeight = imgCV.shape[0] 

           self.tenengrad_focus_measure = tenengrad_focus_measure(imgCV)
           print("Focus : ",self.tenengrad_focus_measure)
           
           processingString = self.ProcessorComboBox.GetValue()
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
           elif (processingString=="Sobel"):
               self.processingWay=3
           elif (processingString=="Visible"):
               self.processingWay=4
           

           global useSAM
           if useSAM:
             processed_img = self.sam_processor.process_image(imgCV)
           else:
             if combineChannels:
                print("Image CV Combining all channels to one")
                if (self.processingWay==3):
                    imgCV = detect_sobel_edges(imgCV)  
                    self.processingWay=0
                #import ipdb; ipdb.set_trace()
                #resize imgpnm 1/2

                imgCV = self.rescaleCVMAT(convertPolarCVMATToRGB(imgCV,way=self.processingWay,brightness=self.brightness_offset, contrast=self.contrast_offset))

                if app.photoTxt.GetValue() != "default": #<- Don't trigger in logo on boot 
                  if self.useClassifierCheckbox.GetValue():

                    if self.classifierTwoStage.GetValue():
                       print("Route through 2-stage classifier here")
                       self.EnsembleClassifierPnm.step = self.classifierTileSize.GetValue()
                       self.EnsembleClassifierPnm.maxProbabilityThreshold = float(self.classifierThreshold.GetValue() / 100.0)
                       imgRGBFromClassifier,occupancy, self.AIAnnotations = self.EnsembleClassifierPnm.forward(imgPNM, majorityVote=self.classifierMajorityVoting.GetValue(), parallel=self.parallellTwoStage.GetValue(), multimodel=self.parallellTwoStage.GetValue())
                       imgRGBFromClassifier = self.rescaleCVMAT(convertRGBCVMATToRGB(imgRGBFromClassifier,brightness=self.brightness_offset, contrast=self.contrast_offset))
                       processed_img = imgRGBFromClassifier
                       self.sam_processor.image = imgRGBFromClassifier
                       self.classifierInfo.SetLabel("2-stage: %0.2f Hz" % self.EnsembleClassifierPnm.hz)
                    else:
                       print("Regular 1-stage classifier here")
                       self.ClassifierPnm.step = self.classifierTileSize.GetValue()
                       self.ClassifierPnm.maxProbabilityThreshold = float(self.classifierThreshold.GetValue() / 100.0)
                       imgRGBFromClassifier,occupancy, self.AIAnnotations = self.ClassifierPnm.forward(imgPNM, majorityVote=self.classifierMajorityVoting.GetValue())
                       imgRGBFromClassifier = self.rescaleCVMAT(convertRGBCVMATToRGB(imgRGBFromClassifier,brightness=self.brightness_offset, contrast=self.contrast_offset))
                       processed_img = imgRGBFromClassifier
                       self.sam_processor.image = imgRGBFromClassifier
                       self.classifierInfo.SetLabel("1-stage: %0.2f Hz" % self.ClassifierPnm.hz)
                else:
                  #If we didn't trigger then show the raw image as processed image
                  processed_img                  = imgCV
                  self.sam_processor.image       = imgCV


                
                if (self.lightComboBox.GetValue()=="Unknown"): #If we don't have a light orientation set
                 print("We don't know Light Direction")
                 if (self.guessLightingCheckbox.GetValue()):   #If we are ok with guessing 
                   print("We will try to guess light direction")
                   self.lightComboBox.SetValue(determine_intensity_region(imgCV, threshold=0.1))

                if self.useClassifierCheckbox.GetValue():
                    #processed_img = imgRGBFromClassifier
                    #self.sam_processor.image = imgRGBFromClassifier
                    pass
                else:
                    processed_img                  = imgCV
                    self.sam_processor.image       = imgCV
                self.sam_processor.foregroundImage = imgCV 
             else:
                processed_img                      = imgCV
                self.sam_processor.image           = imgCV
                self.sam_processor.foregroundImage = imgCV
  
   
           self.width  = processed_img.shape[1]
           self.height = processed_img.shape[0]
           self.viewedImageViewWidth  = self.width
           self.viewedImageViewHeight = self.height
           print("Cast to WxBitmap width / height for ",filepath," ",self.width,"x",self.height)

           self.clickRatioX = self.viewedImageFullWidth / self.viewedImageViewWidth 
           self.clickRatioY = self.viewedImageFullHeight / self.viewedImageViewHeight 

           self.imageCtrl.SetBitmap(wx.Bitmap.FromBuffer(self.width, self.height, processed_img))


           if (len(self.regions_of_interest)>0):
              print("Play back of selected regions")
              for x,y in self.regions_of_interest:
                     self.sam_processor.select_area(int(x),int(y))

           self.onView()

           bmp = self.imageCtrl.GetBitmap()
           if hasattr(self, 'magnifier') and self.magnifier and bmp.IsOk():
               img = bmp.ConvertToImage()
               self.magnifier.setImage(img)
               self.magnifier.refreshZoom()


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
        wx.MessageBox("Made by Ammar Qammaz a.k.a. AmmarkoV\nhttp://ammar.gr/\nVersion %s"%version, "About", wx.OK | wx.ICON_INFORMATION)

   def onRescan(self, newPath):
        self.onProcessNewImageSample(self.filepath)

   def populateMetaData(self,path):
         self.metadata = None
         if (checkIfFileExists(path)):
              with open(path) as json_data:
                   self.metadata = json.load(json_data)

              metadata = list()
              for k in self.metadata.keys():
                  metadata.append("%s: %s"%(k,self.metadata[k]))

              print("Dataset metadata is : ",metadata)
              self.datasetList.Set(metadata)
         else:
              print("Failed opening meta data from ",path)
 
         return self.metadata


   def onNewInputPath(self, newPath):
        print("\n\n\n\nNew Input Path Received : ",newPath)
        self.filepath = newPath
        if (self.filepath!=""):
           self.filePathIsDirectory = checkIfPathIsDirectory(self.filepath)
           if (self.filePathIsDirectory):
               self.folderStreamer.loadNewDataset(self.filepath)
               self.directoryList = list_image_files(self.filepath)
               #self.directoryListIndex = 0
               self.folderStreamer.select(0)
               print("Directory mode")
               #print("Directory mode : ",self.directoryList)
               self.populateMetaData("%s/info.json" % self.filepath)
               self.loadControlsCSV("%s/controller.csv" % self.filepath)
               #self.filepath = self.directoryList[self.directoryListIndex]
               self.filepath = self.folderStreamer.getImage()
               self.updateMinMaxSlider()


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
            self.folderStreamer = HTTPFolderStreamer(base_url=dlg.selectedDataset, local_dir=selectedDirectory, retrieve_zip=dlg.replaceAnnotations)
            self.populateMetaData("%s/info.json" % selectedDirectory)
            self.loadControlsCSV("%s/controller.csv" % selectedDirectory)
            self.onNext(event)
            self.onPrevious(event)
            app.photoTxt.SetValue(dlg.selectedDirectory)
        dlg.Destroy()

   def onBrowse(self, event):
        wildcard = "JPEG files (*.jpg)|*.jpg"
        dialog   = wx.FileDialog(None, "Choose a file",wildcard=wildcard,style=wx.FD_OPEN)
        if dialog.ShowModal() == wx.ID_OK:
            self.photoTxt.SetValue(dialog.GetPath())
        dialog.Destroy()
        self.onNewInputPath(self.photoTxt.GetValue())

   def processImageWithSAM(self, img):
        global useSAM
        if not useSAM:
          print("Deactivated SAM")
          self.sam_processor.foregroundImage = img
          return img

        # Process the image with SAM
        # For example, assuming self.sam_processor.process() is a method that takes an image and returns the processed image
        processed_img = self.sam_processor.process(img)
        
        # Convert the processed image to wx.Image
        h, w = processed_img.shape[:2]
        wx_processed_img = wx.Image(w, h, processed_img.tobytes())
        
        return wx_processed_img


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
 
   def onView(self):
        processed_img = self.sam_processor.foregroundImage
        processed_img = self.rescaleCVMAT(processed_img)

        #Don't recalculate this
        #self.clickRatioX = self.sam_processor.foregroundImage.shape[1] / processed_img.shape[1]
        #self.clickRatioY = self.sam_processor.foregroundImage.shape[0] / processed_img.shape[0]

        h, w = processed_img.shape[:2]
        #processed_img = self.ClassifierPnm.forward(self.sam_processor.foregroundImage)
        wxbmp = wx.Bitmap.FromBuffer(w, h, processed_img)
        
        if len(self.points_of_interest)>0:
          #img_copy = wx.Image(w, h)
          #img_copy.SetData(wxbmp.ConvertToImage().GetData())
          img_copy = wx.Image(wxbmp.ConvertToImage())  # Preserve original image data
          # Draw red circles on the copy for points of interest
          temp_bmp = wx.Bitmap(img_copy)  # Create a temporary wx.Bitmap
          dc = wx.MemoryDC()
          dc.SelectObject(temp_bmp)  # Select the temporary bitmap into the memory DC
          dc.SetBrush(wx.TRANSPARENT_BRUSH)  # Make sure circles are not filled with white

          numberOfPointsToDraw = len(self.points_of_interest)
          for pointID in range(numberOfPointsToDraw):
              x = self.points_of_interest[pointID][0]
              y = self.points_of_interest[pointID][1]
              pClass = self.points_classes[pointID]
              pSever = self.points_severities[pointID]

              if pClass == "Suspicious":
                #This is not a defect we annotate it to make sure it is included
                dc.SetPen(wx.Pen(wx.GREEN, 2))
                dc.DrawCircle(int(x/self.clickRatioX), int(y/self.clickRatioY), 17)
              elif pClass == "Clean":
                #This is not a defect we annotate it to make sure it is included
                dc.SetPen(wx.Pen(wx.GREEN, 2))
                dc.DrawCircle(int(x/self.clickRatioX), int(y/self.clickRatioY), 17)
              else:
                #Regular defect here 
                if   "Class A" in pSever:  
                   dc.SetPen(wx.Pen(wx.YELLOW, 2))
                elif "Class B" in pSever: 
                   wxColor = wx.NamedColour("orange")
                   dc.SetPen(wx.Pen(wxColor, 2)) #wx.ORANGE
                elif "Class C" in pSever: 
                   dc.SetPen(wx.Pen(wx.BLACK, 2))
                elif "AI" in pSever: 
                   dc.SetPen(wx.Pen(wx.WHITE, 2))
                else:
                   print("Weird severity encountered (",pSever,")")
                   dc.SetPen(wx.Pen(wx.BLUE, 2))
                dc.DrawCircle(int(x/self.clickRatioX), int(y/self.clickRatioY), 19)

                dc.SetPen(wx.Pen(wx.RED, 2))
                dc.DrawCircle(int(x/self.clickRatioX), int(y/self.clickRatioY), 17)
                dc.DrawCircle(int(x/self.clickRatioX), int(y/self.clickRatioY), 21)

          dc.SelectObject(wx.NullBitmap)  # Deselect the bitmap from the memory DC

          self.secondaryImageCtrl.SetBitmap(temp_bmp)  # Set the bitmap with the drawn circles

        else:
          self.secondaryImageCtrl.SetBitmap(wxbmp)

        self.panel.Refresh()

   def onSelectPoint(self, event):
        selected_index = self.pointList.GetSelection()
        #if selected_index != -1:
        #    wx.MessageBox(f"Selected Point: {self.points_of_interest[selected_index]}")

   def updateMinMaxSlider(self):
        cur      = self.folderStreamer.current()
        maxim    = self.folderStreamer.max()
        percent  = 100.0 * (cur/maxim) 

        self.scrollBar.SetValue(cur)
        self.scrollBar.SetMax(maxim)
        print("setScrollbar Value/Max ( ",cur,",",maxim,")")
        self.instructLbl.SetLabel("Sample %u/%u - %0.2f%%  - Focus %0.2f" % (cur,maxim,percent,self.tenengrad_focus_measure) )

   def onScroll(self, event):
        # Handle scroll events here
        #scroll_position = self.scrollBar.GetThumbPosition()
        scroll_position = self.scrollBar.GetValue()
        scroll_max      = self.scrollBar.GetMax()
        print("Scroll Position:", scroll_position,"/",scroll_max)

        self.folderStreamer.select(scroll_position)

        if (self.filePathIsDirectory):
               self.onSave(None)
               self.filepath = self.folderStreamer.getImage()
               self.onProcessNewImageSample(self.filepath)
               self.updateMinMaxSlider()
               self.onView()
              

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
        #Next 2 lines work but are a lazy solution ->
        self.onNext(event) 
        self.onPrevious(event)

   def updateControlsTab(self, data_row):
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


   def onPrevious(self, event):
        print("Previous")
        self.folderStreamer.previous()
        if (self.filePathIsDirectory):
               self.onSave(None)
               self.updateMinMaxSlider()
               self.filepath = self.folderStreamer.getImage()
               self.onProcessNewImageSample(self.filepath)
               self.onView()

   def onNext(self, event):
        print("Next")
        self.folderStreamer.next()
        if (self.filePathIsDirectory):
               self.onSave(None)
               self.updateMinMaxSlider()
               self.filepath = self.folderStreamer.getImage()
               self.onProcessNewImageSample(self.filepath)
               self.onView()

   def onRemovePoint(self, event):
        selected_index = self.pointList.GetSelection()
        if selected_index != -1:
            del self.points_of_interest[selected_index]
            del self.points_classes[selected_index]
            del self.points_severities[selected_index]
            self.updatePointList()

   def onRemoveRegion(self, event):
        selected_index = self.regionList.GetSelection()
        if selected_index != -1:
            del self.regions_of_interest[selected_index]
            self.updateRegionList()

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
       if self.photoTxt.GetValue() != "default": #<- Don't trigger in logo on boot 
        self.x, self.y = event.GetPosition()
        self.points_of_interest.append((self.x * self.clickRatioX, self.y * self.clickRatioY))
        selected_option = self.defectComboBox.GetValue()
        self.points_classes.append(selected_option)
        selected_option = self.severityComboBox.GetValue()
        self.points_severities.append(selected_option)

        self.updatePointList() 

        if (self.incrementFrameAfterAnAdditionCheckbox.GetValue()):
               print("Auto Incrementing")
               self.onNext(event)
        else:
               print("Forcing Redraw")
               self.onRedrawData(event)

   def onRightDown(self, event):
      if self.photoTxt.GetValue() != "default": #<- Don't trigger in logo on boot 
        self.x, self.y = event.GetPosition()
        self.regions_of_interest.append((self.x * self.clickRatioX, self.y * self.clickRatioY))
        self.updateRegionList()
        print("Click ",self.x,",",self.y)
        print("Rescaled Click ",int(self.x*self.clickRatioX),",",int(self.y*self.clickRatioY))
        try:
          self.sam_processor.select_area(int(self.x*self.clickRatioX),int(self.y*self.clickRatioY))
        except Exception as e:
          print("Error ",e)
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
        if keycode == wx.WXK_LEFT:
            self.onPrevious(event)
        elif keycode == wx.WXK_RIGHT:
            self.onNext(event)
        elif keycode == wx.WXK_ESCAPE:
            self.onExit(event)
        else:
            event.Skip()

   def onUploadAnnotations(self, event):
      print("Local Dir: ",self.folderStreamer.local_dir)
      zip_path = "./upload.zip"  # replace with your real file path
      zipCommand = "zip %s -b %s %s/color*.json "% (zip_path, self.local_base_path, self.folderStreamer.local_dir) 
      print("Zip command : ",zipCommand)
      os.system(zipCommand)
      dlg = UploadDialog(self.frame, zip_path, self.folderStreamer.local_dir)
      dlg.ShowModal()
      dlg.Destroy()
      os.system("rm upload.zip")

   def onRunBatch(self, event):
        dlg = BatchProcessDialog(self.frame, self.folderStreamer)
        dlg.ShowModal()
        dlg.Destroy()

   def onOpenMagnifier(self, event):
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

   def onCreateDataset(self,event):
       os.system("python3 datasetCreator.py %s" % self.local_base_path) #<- Lazy

   def onTileExplorer(self,event):
       os.system("python3 tileExplorer.py %s" % self.local_base_path) #<- Lazy

   def onStreamer(self,event):
       selectedDirectory = self.folderStreamer.local_dir
       print("Streamer set directory : ",selectedDirectory)
       os.system("python3 streamDataset.py %s" % selectedDirectory) #<- Lazy


   def onMouseMoveMagnifier(self, event):
     if hasattr(self, 'magnifier') and self.magnifier and self.magnifier.IsShown():
        x, y = event.GetX(), event.GetY()
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
               app.useClassifierCheckbox.SetValue(True) 
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

