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
import json
import os
import sys
import numpy as np
import time
import threading

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

version = "0.17"
useSAM  = False
useClassifier  = False #<- Switch classifier off
combineChannels = True
options    = ["Unknown", "Material Defect", "Positive Dent", "Negative Dent", "Deformation", "Seal", "Welding", "Suspicious"]
severities = ["Class A","Class B","Class C"]
processors = ["PolarizationRGB1","PolarizationRGB2","PolarizationRGB3", "Polarization_0_degree","Polarization_45_degree","Polarization_90_degree", "Polarization_135_degree", "Sobel","Visible","SAM"]



# Go up two directories
if useClassifier:
  parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
  sys.path.append(parent_path)
  from classifier.liveClassifierTorch import ClassifierPnm
else:
  class ClassifierPnm:
    def __init__(self, model_path='foo', tile_classes=['foo'],tile_size=64, step=16):
        pass
    def load_model(self):
        return None
    def forward(self, image):
        return None
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
#-------------------------------------------------------------------------------

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
    def __init__(self, parent, zip_path, dataset):
        super().__init__(parent, title="Upload Annotations", size=(350, 200))
        self.zip_path = zip_path  # path to the zip file
        self.dataset  = dataset

        vbox = wx.BoxSizer(wx.VERTICAL)

        # Username
        hbox1 = wx.BoxSizer(wx.HORIZONTAL)
        hbox1.Add(wx.StaticText(self, label="Username:"), 0, wx.ALL | wx.CENTER, 5)
        self.username = wx.TextCtrl(self)
        hbox1.Add(self.username, 1, wx.ALL | wx.EXPAND, 5)
        vbox.Add(hbox1, 0, wx.EXPAND)

        # Password
        hbox2 = wx.BoxSizer(wx.HORIZONTAL)
        hbox2.Add(wx.StaticText(self, label="Password:"), 0, wx.ALL | wx.CENTER, 5)
        self.password = wx.TextCtrl(self, style=wx.TE_PASSWORD)
        hbox2.Add(self.password, 1, wx.ALL | wx.EXPAND, 5)
        vbox.Add(hbox2, 0, wx.EXPAND)


 
        vbox.Add(wx.StaticText(self, label=" Contact ammarkov@ics.forth.gr for a new account"), 0, wx.EXPAND)

        # Buttons
        btns = self.CreateSeparatedButtonSizer(wx.OK | wx.CANCEL)
        vbox.Add(btns, 0, wx.EXPAND | wx.ALL, 10)

        self.SetSizer(vbox)

        # Override Upload (OK) behavior
        self.Bind(wx.EVT_BUTTON, self.onUpload, id=wx.ID_OK)

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
        self.show_crosshair = False

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
        self.ClassifierPnm = ClassifierPnm()
        wx.App.__init__(self, redirect, filename)


        screen = wx.Display(0)  # Get the primary display
        screen_width, screen_height = screen.GetGeometry().GetSize()
        #screen_width = 1900
        print("Screen Resolution: {}x{}".format(screen_width, screen_height))

        if (screen_width>=1920):
          self.PhotoMaxSizeWidth   = 800
          self.PhotoMaxSizeHeight  = 650
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
        # Create the File menu-----------------------------------------------------
        menuBar = wx.MenuBar()

        fileMenu = wx.Menu()

        # Add items to the File menu
        itemOpen    = fileMenu.Append(wx.ID_OPEN, "&Open Image", "Open an image file")
        itemOpenDir = fileMenu.Append(wx.ID_OPEN, "Open &Directory", "Open a directory")
        itemOpenNet = fileMenu.Append(wx.ID_OPEN, "Open &Network", "Open network server")
        itemUpload = fileMenu.Append(wx.ID_UP, "Upload &Annotations", "Upload annotations to server")
        self.Bind(wx.EVT_MENU, self.onUploadAnnotations, itemUpload)
        itemBatch = fileMenu.Append(wx.ID_DOWN, "Download &All", "Process multiple files automatically")
        self.Bind(wx.EVT_MENU, self.onRunBatch, itemBatch)

        itemSave    = fileMenu.Append(wx.ID_SAVE, "&Save", "Save the current file")
        fileMenu.AppendSeparator()
        itemGen     = fileMenu.Append(wx.ID_NEW, "&Generate JSON", "Generate JSON for all files")
        itemDebug   = fileMenu.Append(wx.ID_MORE, "Debug", "Debug GUI")
        fileMenu.AppendSeparator()
        itemExit    = fileMenu.Append(wx.ID_EXIT, "E&xit", "Exit the application")

        # Bind events to the menu items
        self.Bind(wx.EVT_MENU, self.onBrowse, itemOpen)
        self.Bind(wx.EVT_MENU, self.onOpenDirectory, itemOpenDir)  # Bind to the new option
        self.Bind(wx.EVT_MENU, self.onOpenNetwork, itemOpenNet)  # Bind to the new option 
        self.Bind(wx.EVT_MENU, self.onGenerateJSON, itemGen)  # Bind to the new option
        self.Bind(wx.EVT_MENU, self.onSave, itemSave)
        self.Bind(wx.EVT_MENU, self.onDebug, itemDebug)
        self.Bind(wx.EVT_MENU, self.onExit, itemExit)


        # Add the File menu to the menu bar
        menuBar.Append(fileMenu, "&File")


        toolsMenu = wx.Menu()
        itemMagnify    = toolsMenu.Append(wx.ID_ZOOM_IN, "&Magnifier", "Magnifier")
        self.Bind(wx.EVT_MENU, self.onOpenMagnifier,itemMagnify)

        # Add the File menu to the menu bar
        menuBar.Append(toolsMenu, "&Tools")

        # Create the Help menu-----------------------------------------------------
        helpMenu = wx.Menu()

        # Add items to the Help menu
        itemAbout = helpMenu.Append(wx.ID_ABOUT, "&About", "Information about this application")

        # Bind events to the menu items in the Help menu
        self.Bind(wx.EVT_MENU, self.onAbout, itemAbout)

        # Add the Help menu to the menu bar
        menuBar.Append(helpMenu, "&Help")

        # Set the menu bar for the frame
        self.frame.SetMenuBar(menuBar)

        img = wx.Image(self.PhotoMaxSizeWidth,self.PhotoMaxSizeHeight)
        self.imageCtrl = wx.StaticBitmap(self.panel, wx.ID_ANY,wx.Bitmap(img))

        # Add a secondary StaticBitmap
        self.secondaryImageCtrl = wx.StaticBitmap(self.panel, wx.ID_ANY, wx.Bitmap(img))

        self.instructLbl   = wx.StaticText(self.panel, label='Magician Annotator')
        self.photoTxt = wx.TextCtrl(self.panel, size=(200, -1),style=wx.TE_PROCESS_ENTER)
        self.photoTxt.Bind(wx.EVT_TEXT_ENTER, self.onPhotoTxtEnter)

        browseBtn = wx.Button(self.panel, label='Browse')
        browseBtn.Bind(wx.EVT_BUTTON, self.onBrowse)
        self.rescanBtn = wx.Button(self.panel, label='Rescan')
        self.rescanBtn.Bind(wx.EVT_BUTTON, self.onRescan)

 
        #self.scrollBar = wx.ScrollBar(self.panel, size=(600, -1),style=wx.SB_HORIZONTAL)
        #self.scrollBar.SetScrollbar(0, 100, 1500, 99 )  # Set the range from 0 to 1000 with 100 visible
        #self.scrollBar.Bind(wx.EVT_SCROLL, self.onScroll)
        #self.scrollBar.Bind(wx.EVT_SCROLL_THUMBTRACK, self.onScroll) #<- ? is this needed ? 
        #self.scrollBar.Bind(wx.EVT_SCROLL_CHANGED, self.onScroll)    #<- ? is this needed ?
        
        # Replace ScrollBar with Slider
        self.scrollBar = wx.Slider(self.panel, value=0, minValue=0, maxValue=1000, size=(400, -1), style=wx.SL_HORIZONTAL)
        self.scrollBar.SetTickFreq(50) 	
        self.scrollBar.Bind(wx.EVT_SLIDER, self.onScroll)


        #Brightness Control 
        #-------------------------------------------------------------------------------------------
        # Minus button
        self.minusButton = wx.Button(self.panel, label="-", size=(40, 30))
        self.minusButton.Bind(wx.EVT_BUTTON, self.decrease_brightness)
        # Text field
        self.brightnessLabel = wx.StaticText(self.panel, label = "Br.")
        self.brightnessText = wx.TextCtrl(self.panel, value="0", size=(50, 30), style=wx.TE_CENTER)
        #self.brightnessText.SetEditable(False)  # Make it read-only      
        self.brightnessText.Bind(wx.EVT_TEXT, self.on_brightness_change)
        # Plus button
        self.plusButton = wx.Button(self.panel, label="+", size=(40, 30))
        self.plusButton.Bind(wx.EVT_BUTTON, self.increase_brightness)
        #-------------------------------------------------------------------------------------------

        #Brightness Control 
        #-------------------------------------------------------------------------------------------
        # Minus button
        self.minusContrastButton = wx.Button(self.panel, label="-", size=(40, 30))
        self.minusContrastButton.Bind(wx.EVT_BUTTON, self.decrease_contrast)
        # Text field
        self.contrastLabel = wx.StaticText(self.panel, label = "Co.")
        self.contrastText = wx.TextCtrl(self.panel, value="0", size=(50, 30), style=wx.TE_CENTER)
        #self.brightnessText.SetEditable(False)  # Make it read-only      
        self.contrastText.Bind(wx.EVT_TEXT, self.on_contrast_change)
        # Plus button
        self.plusContrastButton = wx.Button(self.panel, label="+", size=(40, 30))
        self.plusContrastButton.Bind(wx.EVT_BUTTON, self.increase_contrast)
        #-------------------------------------------------------------------------------------------

 
        # Add a checkbox
        self.maintainPointsCheckbox = wx.CheckBox(self.panel, label="Maintain Points for next image")
        self.maintainPoints = False  # Initial state
        self.maintainPointsCheckbox.SetValue(self.maintainPoints)   
        
        # Add a checkbox
        self.incrementFrameAfterAnAdditionCheckbox = wx.CheckBox(self.panel, label="Increment frame after defect annotation")
        self.incrementFrameAfterAnAddition=True
        self.incrementFrameAfterAnAdditionCheckbox.SetValue(self.incrementFrameAfterAnAddition)
        
        #Add a checkbox for classifier
        self.useClassifierCheckbox = wx.CheckBox(self.panel, label="Use NN Classifier")
        self.useClassifierCheckbox.SetValue(False)

        # Add a checkbox
        self.guessLightingCheckbox = wx.CheckBox(self.panel, label="Guess lighting direction")
        self.guessLightingCheckbox.SetValue(True)   


        global processors
        self.ProcessorComboBox = wx.ComboBox(self.panel, choices=processors, style=wx.CB_DROPDOWN)
        self.ProcessorComboBox.Bind(wx.EVT_COMBOBOX, self.onProcessorComboBoxSelect)
        self.ProcessorComboBox.SetValue(processors[0])


        # Add a wxComboBox
        self.defectLabel = wx.StaticText(self.panel, label = "Defect Classification")
        global options
        self.defectComboBox = wx.ComboBox(self.panel, choices=options, style=wx.CB_DROPDOWN)
        self.defectComboBox.Append("Add Custom Option")
        self.defectComboBox.Bind(wx.EVT_COMBOBOX, self.onComboBoxSelect)
        self.defectComboBox.SetValue(options[0])

        self.severityLabel = wx.StaticText(self.panel, label = "Defect Severity")
        global severities
        self.severityComboBox = wx.ComboBox(self.panel, choices=severities, style=wx.CB_DROPDOWN)

        self.lightLabel = wx.StaticText(self.panel, label = "Light Direction")
        self.lightComboBox = wx.ComboBox(self.panel, choices=["Unknown","Bottom Left","Top Left","Top","Top Right", "Bottom Right", "Bottom"], style=wx.CB_DROPDOWN)

        self.datasetLabel = wx.StaticText(self.panel, label = "Dataset Information")

        datasetListSize = wx.Size(-1,80) 
        self.datasetList  = wx.ListBox(self.panel, size=datasetListSize, choices=[], style=wx.LB_SINGLE)

        #Side buttons
        self.regionLabel = wx.StaticText(self.panel, label = "Image Regions")
        regionListSize = wx.Size(-1,40) 
        self.regionList = wx.ListBox(self.panel, size=regionListSize, choices=[], style=wx.LB_SINGLE)
        self.regionList.Bind(wx.EVT_LISTBOX, self.onSelectPoint)
        self.removeRegionBtn = wx.Button(self.panel, label='Remove Selected Point')
        self.removeRegionBtn.Bind(wx.EVT_BUTTON, self.onRemovePoint)
        #---------------------------------------------------------------------------
        self.line = wx.StaticLine(self.panel)
        #---------------------------------------------------------------------------
        self.pointLabel = wx.StaticText(self.panel, label = "Image Points")
        self.pointList = wx.ListBox(self.panel, choices=[], style=wx.LB_SINGLE)
        self.pointList.Bind(wx.EVT_LISTBOX, self.onSelectPoint)
        self.removePointBtn = wx.Button(self.panel, label='Remove Selected Point')
        self.removePointBtn.Bind(wx.EVT_BUTTON, self.onRemovePoint)
        #---------------------------------------------------------------------------
        self.saveBtn = wx.Button(self.panel, label='Save')
        self.saveBtn.Bind(wx.EVT_BUTTON, self.onSave)
        self.deleteMetadataBtn = wx.Button(self.panel, label='Delete Metadata')
        self.deleteMetadataBtn.Bind(wx.EVT_BUTTON, self.ondeleteMetadata)

        #Under Buttons
        self.prevBtn = wx.Button(self.panel, label='<')
        self.prevBtn.Bind(wx.EVT_BUTTON, self.onPrevious)
        self.nextBtn = wx.Button(self.panel, label='>')
        self.nextBtn.Bind(wx.EVT_BUTTON, self.onNext)


        # Add a button for camera settings
        self.cameraSettingsBtn = wx.Button(self.panel, label='Camera')
        self.cameraSettingsBtn.Bind(wx.EVT_BUTTON, self.onCameraSettings)


        self.mainSizer  = wx.BoxSizer(wx.VERTICAL)
        self.sizer      = wx.BoxSizer(wx.HORIZONTAL)
        self.sideSizer  = wx.BoxSizer(wx.VERTICAL)
        self.underImage = wx.BoxSizer(wx.HORIZONTAL)

        self.mainSizer.Add(wx.StaticLine(self.panel, wx.ID_ANY), 0, wx.ALL | wx.EXPAND, 5)
        self.mainSizer.Add(self.instructLbl, 0, wx.ALL, 5)


        # Add both StaticBitmaps to a horizontal sizer
        self.sizer.Add(self.imageCtrl, 0, wx.ALL, 5)
        self.sizer.Add(self.secondaryImageCtrl, 0, wx.ALL, 5)

        self.mainSizer.Add(self.sizer, 0, wx.ALL, 5)
        self.sizer.Add(self.sideSizer, 0, wx.ALL, 5)

        #Side bar
        self.sideSizer.Add(self.datasetLabel, 0, wx.ALL | wx.EXPAND, 5)
        self.sideSizer.Add(self.datasetList, 0, wx.ALL | wx.EXPAND, 5)

        self.sideSizer.Add(self.regionLabel, 0, wx.ALL | wx.EXPAND, 5)
        self.sideSizer.Add(self.regionList, 0, wx.ALL | wx.EXPAND, 5)
        self.sideSizer.Add(self.removeRegionBtn, 0, wx.ALL, 5)
        #------------------------------------------------------------
        self.sideSizer.Add(self.line , 0, wx.ALL, 5)
        #------------------------------------------------------------
        self.sideSizer.Add(self.defectLabel, 0, wx.ALL, 5)
        self.sideSizer.Add(self.defectComboBox, 0, wx.ALL, 5)

        self.sideSizer.Add(self.severityLabel, 0, wx.ALL, 5)
        self.sideSizer.Add(self.severityComboBox, 0, wx.ALL, 5) 

        self.sideSizer.Add(self.lightLabel, 0, wx.ALL, 5)
        self.sideSizer.Add(self.lightComboBox, 0, wx.ALL, 5)  

        self.sideSizer.Add(self.pointLabel, 0, wx.ALL | wx.EXPAND, 5)
        self.sideSizer.Add(self.pointList, 0, wx.ALL | wx.EXPAND, 5)
        self.sideSizer.Add(self.removePointBtn, 0, wx.ALL, 5)
        #------------------------------------------------------------
        comboButtons      = wx.BoxSizer(wx.HORIZONTAL)  
        comboButtons.Add(self.saveBtn, 0, wx.ALL, 5)
        comboButtons.Add(self.deleteMetadataBtn, 0, wx.ALL, 5)
        self.sideSizer.Add(comboButtons)

        self.sideSizer.Add(self.maintainPointsCheckbox, 0, wx.ALL, 5)
        self.sideSizer.Add(self.incrementFrameAfterAnAdditionCheckbox, 0, wx.ALL, 5)
        self.sideSizer.Add(self.useClassifierCheckbox, 0, wx.ALL, 5)
        self.sideSizer.Add(self.guessLightingCheckbox, 0, wx.ALL, 5)
        #------------------------------------------------------------

        #Under Image
        self.underImage.Add(self.prevBtn, 0, wx.ALL, 5)
        self.underImage.Add(self.nextBtn, 0, wx.ALL, 5)
        self.underImage.Add(self.photoTxt, 0, wx.ALL, 5)
        self.underImage.Add(browseBtn, 0, wx.ALL, 5)
        self.underImage.Add(self.rescanBtn, 0, wx.ALL, 5)
        self.underImage.Add(self.scrollBar, 0, wx.ALL | wx.EXPAND, 5)
        self.underImage.Add(self.cameraSettingsBtn, 0, wx.ALL | wx.EXPAND, 5)
        self.underImage.Add(self.ProcessorComboBox, 0, wx.ALL | wx.EXPAND, 5)

        #self.underImage.Add(self.brightnessScrollBar, 0, wx.ALL | wx.EXPAND, 5) 
        # Add to sizer
        self.underImage.Add(self.minusButton, 0, wx.ALL, 5) 
        self.underImage.Add(self.brightnessLabel, 0, wx.ALL, 5)
        self.underImage.Add(self.brightnessText, 0, wx.ALL, 5)
        self.underImage.Add(self.plusButton, 0, wx.ALL, 5)

        self.underImage.Add(self.minusContrastButton, 0, wx.ALL, 5) 
        self.underImage.Add(self.contrastLabel, 0, wx.ALL, 5)
        self.underImage.Add(self.contrastText, 0, wx.ALL, 5)
        self.underImage.Add(self.plusContrastButton, 0, wx.ALL, 5)

        self.mainSizer.Add(self.underImage)       

        self.panel.SetSizer(self.mainSizer)
        self.mainSizer.Fit(self.frame)

        self.imageCtrl.Bind(wx.EVT_LEFT_DOWN, self.onLeftDown)  # Bind to image control
        self.secondaryImageCtrl.Bind(wx.EVT_LEFT_DOWN, self.onLeftDown)  # Bind to image control

        self.imageCtrl.Bind(wx.EVT_MIDDLE_DOWN, self.onMiddleDown)  # Bind to image control
        self.secondaryImageCtrl.Bind(wx.EVT_MIDDLE_DOWN, self.onMiddleDown)  # Bind to image control

        self.imageCtrl.Bind(wx.EVT_RIGHT_DOWN, self.onRightDown)  # Bind to image control
        self.secondaryImageCtrl.Bind(wx.EVT_RIGHT_DOWN, self.onRightDown)  # Bind to image control

        # Bind the mouse wheel event to the panel
        self.panel.Bind(wx.EVT_MOUSEWHEEL, self.onMouseWheel)

        # Bind left and right arrow keys to onNext and onPrevious methods
        self.frame.Bind(wx.EVT_CHAR_HOOK, self.onKeyPress)

        self.panel.Layout()
 
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
           

           #self.filehash = get_md5(filepath) 
           #img = wx.Image(self.filepath, wx.BITMAP_TYPE_ANY)
           #img = self.rescaleBitmap(img)
           #self.imageCtrl.SetBitmap(wx.Bitmap(img))

           # Process the image with SAM and update the second StaticBitmap
           global combineChannels
           imgCV  = cv2.imread(filepath) #,cv2.IMREAD_UNCHANGED
           imgPNM = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
           print("Raw image dims for ",filepath," ",imgCV.shape)
           self.viewedImageFullWidth  = imgCV.shape[1]
           self.viewedImageFullHeight = imgCV.shape[0] 

           
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
                if self.useClassifierCheckbox.GetValue():
                    imgClassifier = self.ClassifierPnm.forward(imgPNM)
                    imgClassifier = self.rescaleCVMAT(convertPolarCVMATToRGB(imgClassifier,way=self.processingWay,brightness=self.brightness_offset, contrast=self.contrast_offset))
                imgCV = self.rescaleCVMAT(convertPolarCVMATToRGB(imgCV,way=self.processingWay,brightness=self.brightness_offset, contrast=self.contrast_offset))
                
                if (self.lightComboBox.GetValue()=="Unknown"): #If we don't have a light orientation set
                 print("We don't know Light Direction")
                 if (self.guessLightingCheckbox.GetValue()):   #If we are ok with guessing 
                   print("We will try to guess light direction")
                   self.lightComboBox.SetValue(determine_intensity_region(imgCV, threshold=0.1))

                if self.useClassifierCheckbox.GetValue():
                    processed_img = imgClassifier
                    self.sam_processor.image = imgClassifier
                else:
                    processed_img                  = imgCV
                    self.sam_processor.image       = imgCV
                self.sam_processor.foregroundImage = imgCV 
             else:
                processed_img                      = imgCV
                self.sam_processor.image           = imgCV
                self.sam_processor.foregroundImage = imgCV
  
           self.tenengrad_focus_measure = tenengrad_focus_measure(processed_img)
           print("Focus : ",self.tenengrad_focus_measure)
   
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
        wx.MessageBox("Made by Ammar Qammaz a.k.a. AmmarkoV\n\nVersion %s"%version, "About", wx.OK | wx.ICON_INFORMATION)

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
               self.populateMetaData("%s/info.json"%self.filepath)
               #self.filepath = self.directoryList[self.directoryListIndex]
               self.filepath = self.folderStreamer.getImage()
               self.updateMinMaxSlider()


           self.onProcessNewImageSample(self.filepath)

    def onPhotoTxtEnter(self, event):
        self.onNewInputPath(self.photoTxt.GetValue())

    def onExit(self, event):
        sys.exit(0)

    def onProcessorComboBoxSelect(self, event):
        print("Combo box select changed")
        self.onRedrawData(event)

    def onComboBoxSelect(self, event):
      selected_option = self.defectComboBox.GetValue()
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
        dlg = DatasetSelector()
        if dlg.ShowModal() == wx.ID_OK:
            print("Selected Dataset:",  dlg.selectedDataset)
            print("Caching Directory:", dlg.selectedDirectory)
            # You can pass this to your HTTPFolderStreamer
            #self.onNewInputPath(dlg.selectedDataset)
            from HTTPStream import HTTPFolderStreamer 
            self.folderStreamer = HTTPFolderStreamer(base_url=dlg.selectedDataset, local_dir=dlg.selectedDirectory)
            self.onNext(event)
            self.onPrevious(event)
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
              else:
                #Regular defect here 
                if   "Class A" in pSever:  
                   dc.SetPen(wx.Pen(wx.YELLOW, 2))
                elif "Class B" in pSever: 
                   dc.SetPen(wx.Pen(wx.ORANGE, 2))
                elif "Class C" in pSever: 
                   dc.SetPen(wx.Pen(wx.BLACK, 2))
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
        self.x, self.y = event.GetPosition()
        self.regions_of_interest.append((self.x * self.clickRatioX, self.y * self.clickRatioY))
        self.updateRegionList()
        print("Click ",self.x,",",self.y)
        print("Rescaled Click ",int(self.x*self.clickRatioX),",",int(self.y*self.clickRatioY))
        self.sam_processor.select_area(int(self.x*self.clickRatioX),int(self.y*self.clickRatioY))
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
      os.system("zip upload.zip  %s/color*.json "%self.folderStreamer.local_dir)
      zip_path = "upload.zip"  # replace with your real file path
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
           if (sys.argv[i]=="--from"):
               loadDataset=sys.argv[i+1]

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
               inputIsSet = False
 

    if not inputIsSet:
               app.photoTxt.SetValue("default")
               app.onNewInputPath("default")


    app.MainLoop()

