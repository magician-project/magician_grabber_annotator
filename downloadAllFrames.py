import os
import threading
import time
import wx

from compressDataset import write_polar_png_from_pnm  # if you use the compression checkbox


class BatchProcessDialog(wx.Dialog):
    def __init__(self, parent, folderStreamer):
        super().__init__(parent, title="Batch Download of Dataset", size=(420, 260))
        self.folderStreamer = folderStreamer

        self.stop_requested = False
        self._modal_closed = False  # <- NEW: prevents UI updates after closing

        vbox = wx.BoxSizer(wx.VERTICAL)

        # Spin control
        hbox1 = wx.BoxSizer(wx.HORIZONTAL)
        hbox1.Add(wx.StaticText(self, label="Number of files to download:"), 0, wx.ALL | wx.CENTER, 5)
        self.spin = wx.SpinCtrl(self, min=1, max=100000, initial=self.folderStreamer.max())
        hbox1.Add(self.spin, 1, wx.ALL | wx.CENTER, 5)
        vbox.Add(hbox1, 0, wx.EXPAND)

        # Optional: compression checkbox (keep if you already added it)
        self.cb_compress_pnm = wx.CheckBox(self, label="Use image compression to occupy less space")
        self.cb_compress_pnm.SetValue(True)
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

        self.okBtn = self.FindWindowById(wx.ID_OK)
        self.cancelBtn = self.FindWindowById(wx.ID_CANCEL)

        self.okBtn.Bind(wx.EVT_BUTTON, self.onStart)
        self.cancelBtn.Bind(wx.EVT_BUTTON, self.onCancel)

        # If user closes window via X, treat as cancel too
        self.Bind(wx.EVT_CLOSE, self.onClose)

    def onStart(self, event):
        count = self.spin.GetValue()
        self.okBtn.Disable()
        self.stop_requested = False
        threading.Thread(target=self.runBatch, args=(count,), daemon=True).start()

    def onCancel(self, event):
        # IMPORTANT: end modal immediately so ShowModal() returns
        self.stop_requested = True
        self.cancelBtn.Disable()
        self._end_modal_safe(wx.ID_CANCEL)

    def onClose(self, event):
        # X button: same as cancel
        self.stop_requested = True
        self._end_modal_safe(wx.ID_CANCEL)

    def _end_modal_safe(self, code):
        """EndModal safely exactly once."""
        if self._modal_closed:
            return
        self._modal_closed = True

        # If dialog is modal, EndModal; else Destroy.
        try:
            if self.IsModal():
                self.EndModal(code)
            else:
                self.Destroy()
        except Exception:
            # If already torn down, ignore
            pass

    def _ui_call(self, fn, *args, **kwargs):
        """Call UI updates only if dialog is still alive."""
        if self._modal_closed:
            return
        try:
            wx.CallAfter(fn, *args, **kwargs)
        except Exception:
            pass

    def _maybe_compress_pnm(self, image_path: str):
        if not image_path:
            return
        if self._modal_closed or self.stop_requested:
            return
        if not self.cb_compress_pnm.GetValue():
            return
        if not image_path.lower().endswith(".pnm"):
            return

        png_path = os.path.splitext(image_path)[0] + ".png"
        try:
            write_polar_png_from_pnm(image_path, png_path)
            try:
                os.remove(image_path)
            except OSError:
                pass
        except Exception as e:
            self._ui_call(self.eta_label.SetLabel, f"Compression failed: {e}")

    def runBatch(self, count):
        times = []
        for i in range(count):
            if self.stop_requested or self._modal_closed:
                break

            start = time.perf_counter()

            # These may block; cancel will close the dialog immediately,
            # and the thread will just finish whenever these return.
            self.folderStreamer.next()
            if self.stop_requested or self._modal_closed:
                break

            self.folderStreamer.getJSON()
            if self.stop_requested or self._modal_closed:
                break

            image_path = self.folderStreamer.getImageSimple()
            self._maybe_compress_pnm(image_path)

            elapsed = time.perf_counter() - start
            times.append(elapsed)

            avg_time = sum(times) / len(times)
            remaining = avg_time * (count - i - 1) / 60.0

            self._ui_call(self.gauge.SetValue, int((i + 1) / count * 100))
            self._ui_call(self.eta_label.SetLabel, f"Estimated time remaining: {remaining:.1f} mins")

        # Only close with OK if user didn’t cancel and dialog is still open
        if (not self.stop_requested) and (not self._modal_closed):
            self._end_modal_safe(wx.ID_OK)
