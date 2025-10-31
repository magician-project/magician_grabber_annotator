import wx
import wx.adv
import threading
import os
import sys
import cv2
import numpy as np
import sys

try:
  parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../classifier'))
  sys.path.append(parent_path)
except:
  print("Could not find classifier ..")
from SharedMemoryManager import SharedMemoryManager

# ----------------- Helper functions -----------------
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def checkIfFileExists(filename):
    return os.path.isfile(filename)

def resize_with_padding(img, target_width, target_height):
    h, w = img.shape[:2]
    scale = min(target_width/w, target_height/h)
    new_w, new_h = int(w*scale), int(h*scale)
    resized = cv2.resize(img, (new_w, new_h))
    top = (target_height - new_h)//2
    bottom = target_height - new_h - top
    left = (target_width - new_w)//2
    right = target_width - new_w - left
    return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0,0,0])

# ----------------- FolderStreamer -----------------
class FolderStreamer():
    def __init__(self, path="./", label="colorFrame_0_", width=0, height=0, loop=True):
        self.path        = path
        self.label       = label
        self.frameNumber = 0
        self.width       = width
        self.height      = height
        self.should_stop = False
        self.loop        = loop

        self.available_frames = sorted([
            int(f.replace(label, "").split(".")[0])
            for f in os.listdir(path)
            if f.startswith(label) and (f.endswith(".jpg") or f.endswith(".png") or f.endswith(".pnm"))
        ])

        if not self.available_frames:
            eprint("No frames found in folder:", path)
        else:
            eprint(f"Found {len(self.available_frames)} frames, looping = {self.loop}")

    def isOpened(self):
        return not self.should_stop

    def release(self):
        self.should_stop = True 

    def read(self):
        if self.frameNumber >= len(self.available_frames):
            if self.loop:
                self.frameNumber = 0
            else:
                self.should_stop = True
                return False, None

        filenameJPG = f"{self.path}/{self.label}{self.frameNumber:05d}.jpg"
        filenamePNG = f"{self.path}/{self.label}{self.frameNumber:05d}.png"
        filenamePNM = f"{self.path}/{self.label}{self.frameNumber:05d}.pnm"

        try:
            if checkIfFileExists(filenameJPG):
                self.img = cv2.imread(filenameJPG, cv2.IMREAD_UNCHANGED)
            elif checkIfFileExists(filenamePNG):
                self.img = cv2.imread(filenamePNG, cv2.IMREAD_UNCHANGED)
            elif checkIfFileExists(filenamePNM):
                self.img = cv2.imread(filenamePNM, cv2.IMREAD_UNCHANGED)
            else:
                eprint("Could not find ", filenameJPG, filenamePNG, filenamePNM)
                self.img = None
        except Exception as e:
            eprint("Failed to open frame:", e)
            self.img = None

        if self.img is not None:
            #if self.width != 0 and self.height != 0:
            #    self.img = resize_with_padding(self.img, self.width, self.height)
            success = True
            self.frameNumber += 1
        else:
            success = False
            self.should_stop = True
        return success, self.img

# ----------------- wxPython GUI -----------------
class StreamerFrame(wx.Frame):
    def __init__(self, path=None):
        super().__init__(None, title="Folder Streamer", size=(900,700))
        self.panel = wx.Panel(self)

        self.streamer = None
        self.smm = None
        self.thread = None
        self.timer = wx.Timer(self)

        self.current_frame = None
        self.total_frames = 0
        self.start_dir = path
        # ---- GUI Elements ----
        vbox = wx.BoxSizer(wx.VERTICAL)

        # Directory and Stream Name
        hbox1 = wx.BoxSizer(wx.HORIZONTAL)
        self.dir_picker = wx.DirPickerCtrl(self.panel, message="Select source folder", path=self.start_dir)
        hbox1.Add(wx.StaticText(self.panel, label="Source Folder:"), 0, wx.ALL|wx.CENTER, 5)
        hbox1.Add(self.dir_picker, 1, wx.ALL|wx.EXPAND, 5)
        vbox.Add(hbox1, 0, wx.EXPAND)

        hbox2 = wx.BoxSizer(wx.HORIZONTAL)
        hbox2.Add(wx.StaticText(self.panel, label="Stream Name:"), 0, wx.ALL|wx.CENTER, 5)
        self.stream_name_ctrl = wx.TextCtrl(self.panel, value="stream1")
        hbox2.Add(self.stream_name_ctrl, 1, wx.ALL|wx.EXPAND, 5)
        vbox.Add(hbox2, 0, wx.EXPAND)

        # Buttons
        hbox3 = wx.BoxSizer(wx.HORIZONTAL)
        self.start_btn = wx.Button(self.panel, label="Start")
        self.stop_btn  = wx.Button(self.panel, label="Stop")
        hbox3.Add(self.start_btn, 0, wx.ALL, 5)
        hbox3.Add(self.stop_btn, 0, wx.ALL, 5)
        vbox.Add(hbox3, 0, wx.CENTER)

        # Progress bar
        self.gauge = wx.Gauge(self.panel, range=100, style=wx.GA_HORIZONTAL)
        vbox.Add(self.gauge, 0, wx.ALL|wx.EXPAND, 10)

        # Image preview
        self.bmp = wx.StaticBitmap(self.panel, size=(640,480))
        vbox.Add(self.bmp, 0, wx.ALL|wx.CENTER, 5)

        self.panel.SetSizer(vbox)

        # Bind events
        self.Bind(wx.EVT_TIMER, self.on_timer, self.timer)
        self.start_btn.Bind(wx.EVT_BUTTON, self.on_start)
        self.stop_btn.Bind(wx.EVT_BUTTON, self.on_stop)
        self.Bind(wx.EVT_CLOSE, self.on_close)

    def __del__(self):
        print("Stream Dataset Destructor called")
        #os.system("ps -A | grep server")
        #os.system("killall server")
        #os.system("sleep 1 && ps -A | grep server")

    def on_start(self, event):
        folder = self.dir_picker.GetPath()
        stream_name = self.stream_name_ctrl.GetValue()
        if not folder or not stream_name:
            wx.MessageBox("Please select folder and set stream name", "Error", wx.ICON_ERROR)
            return

        self.streamer = FolderStreamer(path=folder)
        if not self.streamer.available_frames:
            wx.MessageBox("No frames found in folder", "Error", wx.ICON_ERROR)
            return

        first_frame_success, frame = self.streamer.read()
        if not first_frame_success:
            wx.MessageBox("Cannot read first frame", "Error", wx.ICON_ERROR)
            return

        width  = frame.shape[1]
        height = frame.shape[0]
        channels = 1 if len(frame.shape) == 2 else frame.shape[2]

    
        #os.system("ln -s ../classifier/libSharedMemoryVideoBuffers.so")
        #os.system("git clone https://github.com/AmmarkoV/SharedMemoryVideoBuffers")
        #os.system("cd SharedMemoryVideoBuffers && make && cd .. && SharedMemoryVideoBuffers/server --nokb&")

        self.smm = SharedMemoryManager("libSharedMemoryVideoBuffers.so", 
                                       descriptor="video_frames.shm", 
                                       frameName=stream_name, 
                                       width=width,
                                       height=height,
                                       channels=channels)

        self.total_frames = len(self.streamer.available_frames)
        self.streamer.frameNumber = 0
        self.streamer.should_stop = False
        self.timer.Start(50)  # 20 FPS update

        # Start background thread
        self.thread = threading.Thread(target=self.stream_loop)
        self.thread.start()

    def stream_loop(self):
        while not self.streamer.should_stop:
            ret, frame = self.streamer.read()
            if not ret:
                break

            self.smm.copy_numpy_to_shared_memory(frame)

            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = frame

            self.current_frame = frame_rgb

            wx.CallAfter(self.update_gui)
            cv2.waitKey(10)

  # ---------- Timer Trigger ----------
    def on_timer(self, event):
        if not self.streamer or self.streamer.should_stop:
            self.timer.Stop()
            return

        ok, frame = self.streamer.read()
        if not ok:
            self.timer.Stop()
            self.running = False
            return

        try:
            self.smm.copy_numpy_to_shared_memory(frame_rgb)
        except Exception as e:
            print("Shared memory write error:", e)
            self.timer.Stop()
            return

        # Ensure proper format
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # Match shape
        if frame.shape[1] != self.smm.width or frame.shape[0] != self.smm.height:
            frame = cv2.resize(frame, (self.smm.width, self.smm.height))

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


        self.current_frame = frame_rgb
        self.update_gui()

    def update_gui(self):
        # Update progress bar
        if self.streamer:
            progress = int(100 * self.streamer.frameNumber / max(1, self.total_frames))
            self.gauge.SetValue(progress)

        # Update image preview
        if self.current_frame is not None:
            try:
              img = cv2.cvtColor(self.current_frame, cv2.COLOR_RGB2BGR)
              print("Image shape ",img.shape)
              imgS = cv2.resize(img, (640, 480))
              h, w = imgS.shape[:2]
              bmp = wx.Bitmap.FromBuffer(w, h, imgS)
              self.bmp.SetBitmap(bmp)
            except Exception as e:
              print("Image Preview encountered as error ",e)

    def on_stop(self, event):
        if self.streamer:
            self.streamer.release()
        if self.timer.IsRunning():
            self.timer.Stop()

    def on_close(self, event):
        if self.streamer:
            self.streamer.release()
        self.Destroy()

# ----------------- Main -----------------
if __name__ == '__main__':
    start_dir = os.path.join(os.getcwd(), './')
    if len(sys.argv) > 1:
        start_dir = sys.argv[1]

    app = wx.App()
    frame = StreamerFrame(path=start_dir)
    frame.Show()
    app.MainLoop()

