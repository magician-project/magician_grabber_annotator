import os
import threading
import time
import wx

# Import conversion helper from compressDataset.py
# (this uses cv2 + numpy internally)
from compressDataset import write_polar_png_from_pnm  # :contentReference[oaicite:1]{index=1}


class BatchProcessDialog(wx.Dialog):
    def __init__(self, parent, folderStreamer):
        super().__init__(parent, title="Batch Download of Dataset", size=(420, 240))
        self.folderStreamer = folderStreamer
        self.stop_requested = False  # flag for cancellation

        vbox = wx.BoxSizer(wx.VERTICAL)

        # Spin control for number of iterations
        hbox1 = wx.BoxSizer(wx.HORIZONTAL)
        hbox1.Add(wx.StaticText(self, label="Number of files to download:"), 0, wx.ALL | wx.CENTER, 5)
        self.spin = wx.SpinCtrl(self, min=1, max=100000, initial=self.folderStreamer.max())
        hbox1.Add(self.spin, 1, wx.ALL | wx.CENTER, 5)
        vbox.Add(hbox1, 0, wx.EXPAND)

        # NEW: Compression checkbox
        self.cb_compress_pnm = wx.CheckBox(self, label="Compress .pnm to .png while downloading")
        self.cb_compress_pnm.SetValue(False)
        vbox.Add(self.cb_compress_pnm, 0, wx.LEFT | wx.RIGHT | wx.TOP, 10)

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

    def _maybe_compress_pnm(self, image_path: str):
        """
        If checkbox is enabled and image_path ends with .pnm:
          - convert to .png (same base name)
          - delete .pnm after successful write
        """
        if not image_path:
            return

        if not self.cb_compress_pnm.GetValue():
            return

        if not image_path.lower().endswith(".pnm"):
            return

        png_path = os.path.splitext(image_path)[0] + ".png"

        try:
            # Convert pnm->png (polar RGBA) using helper from compressDataset.py
            write_polar_png_from_pnm(image_path, png_path)

            # Remove original pnm to actually "compress" storage usage
            try:
                os.remove(image_path)
            except OSError:
                pass

        except Exception as e:
            # Don’t kill the batch thread; just inform the UI
            wx.CallAfter(self.eta_label.SetLabel, f"Compression failed: {e}")

    def runBatch(self, count):
        times = []
        for i in range(count):
            if self.stop_requested:
                wx.CallAfter(self.eta_label.SetLabel, "Cancelled by user.")
                break

            start = time.perf_counter()

            self.folderStreamer.next()
            self.folderStreamer.getJSON()

            image_path = self.folderStreamer.getImageSimple()
            self._maybe_compress_pnm(image_path)

            elapsed = time.perf_counter() - start
            times.append(elapsed)

            avg_time = sum(times) / len(times)
            remaining = avg_time * (count - i - 1) / 60

            # Update UI safely
            wx.CallAfter(self.gauge.SetValue, int((i + 1) / count * 100))
            wx.CallAfter(self.eta_label.SetLabel, f"Estimated time remaining: {remaining:.1f} mins")

        wx.CallAfter(self.EndModal, wx.ID_OK)
