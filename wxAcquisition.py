import wx
import cv2
import json
import os
import sys
import datetime

class CameraSettingsDialog(wx.Dialog):
    def __init__(self, parent, title):
        wx.Dialog.__init__(self, parent, title=title, size=(300, 600))

        self.exposureCtrl = wx.TextCtrl(self,value="25000", style=wx.TE_PROCESS_ENTER)
        self.framesCtrl = wx.TextCtrl(self,value="100", style=wx.TE_PROCESS_ENTER)
        self.gainCtrl = wx.TextCtrl(self,value="1.0", style=wx.TE_PROCESS_ENTER)
        self.framerateCtrl = wx.TextCtrl(self,value="10", style=wx.TE_PROCESS_ENTER)

        self.acquireBtn = wx.Button(self, label='Acquire', size=(100, 30))
        self.cancelBtn = wx.Button(self, label='Cancel', size=(100, 30))

        self.Bind(wx.EVT_BUTTON, self.onAcquire, self.acquireBtn)
        self.Bind(wx.EVT_BUTTON, self.onCancel, self.cancelBtn)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(wx.StaticText(self, label='Exposure:'), 0, wx.ALL, 5)
        sizer.Add(self.exposureCtrl, 0, wx.ALL | wx.EXPAND, 5)
        sizer.Add(wx.StaticText(self, label='Number of Frames:'), 0, wx.ALL, 5)
        sizer.Add(self.framesCtrl, 0, wx.ALL | wx.EXPAND, 5)
        sizer.Add(wx.StaticText(self, label='Gain:'), 0, wx.ALL, 5)
        sizer.Add(self.gainCtrl, 0, wx.ALL | wx.EXPAND, 5)
        sizer.Add(wx.StaticText(self, label='Framerate:'), 0, wx.ALL, 5)
        sizer.Add(self.framerateCtrl, 0, wx.ALL | wx.EXPAND, 5)
        sizer.Add(wx.StaticLine(self, wx.ID_ANY), 0, wx.ALL | wx.EXPAND, 5)
        sizer.Add(self.acquireBtn, 0, wx.ALL | wx.CENTER, 5)
        sizer.Add(self.cancelBtn, 0, wx.ALL | wx.CENTER, 5)
        self.filename = ""
        self.SetSizer(sizer)

    def onAcquire(self, event):
        exposure = int(self.exposureCtrl.GetValue())
        num_frames = int(self.framesCtrl.GetValue())
        gain = float(self.gainCtrl.GetValue())
        framerate = float(self.framerateCtrl.GetValue())


        os.system("/home/ammar/Documents/3dParty/aravis-c-examples/build/06-grabber -o tmp --exposure %u --fps %0.2f --gain %0.2f --maxFrames %u" % (exposure,framerate,gain,num_frames))
        os.system("python3 averageimages.py tmp/ pnm" )
        
        # Get the current date and time
        current_datetime = datetime.datetime.now()
        # Format the date and time as a string for the filename
        formatted_datetime = current_datetime.strftime("%Y%m%d_%H%M%S")
        # Use the formatted datetime string in the filename
        filename = f"image_{formatted_datetime}.pnm"
        print(filename)
        os.system("mv tmp/average_image.pnm %s"%filename)
        os.system("rm tmp/*")
        os.system("rmdir tmp/")

        self.filename = filename
        self.Close()

    def onCancel(self, event):
        self.Close()


