import os
import threading
import time
import wx

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

    def updateImage(self, bitmap):
        self.original_bitmap = bitmap
        self.Refresh()

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
