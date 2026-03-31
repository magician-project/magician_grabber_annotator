import wx
import wx.lib.newevent
import subprocess
import threading

(ProcessOutputEvent, EVT_PROCESS_OUTPUT) = wx.lib.newevent.NewEvent()


class MagicianGrabberFrame(wx.Frame):
    def __init__(self):
        super().__init__(None, title="Magician Grabber GUI", size=(750, 1000))
        panel = wx.Panel(self)
        main = wx.BoxSizer(wx.VERTICAL)

        # ===================== OUTPUT DIRECTORY =====================
        sb_output = wx.StaticBox(panel, label="Output")
        s_output = wx.StaticBoxSizer(sb_output, wx.VERTICAL)

        hdir = wx.BoxSizer(wx.HORIZONTAL)
        hdir.Add(wx.StaticText(panel, label="Directory:"), 0,
                 wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 8)
        self.dirCtrl = wx.TextCtrl(panel)
        hdir.Add(self.dirCtrl, 1, wx.EXPAND | wx.RIGHT, 8)

        bBrowse = wx.Button(panel, label="Browse…")
        bBrowse.Bind(wx.EVT_BUTTON, self.onBrowse)
        hdir.Add(bBrowse, 0)

        s_output.Add(hdir, 0, wx.EXPAND | wx.ALL, 5)
        main.Add(s_output, 0, wx.EXPAND | wx.ALL, 10)

        # ===================== CHECKBOX OPTIONS =====================
        sb_flags = wx.StaticBox(panel, label="General Options")
        s_flags = wx.StaticBoxSizer(sb_flags, wx.VERTICAL)

        grid_flags = wx.GridSizer(3, 3, 5, 5)

        self.chkSim = wx.CheckBox(panel, label="--simulate")
        self.chkCamera = wx.CheckBox(panel, label="--camera")
        self.chkRAM = wx.CheckBox(panel, label="--ram")
        self.chkSpeak = wx.CheckBox(panel, label="--speak")
        self.chkViewer = wx.CheckBox(panel, label="--view")
        self.chkForever = wx.CheckBox(panel, label="--forever")
        self.chkNoOut = wx.CheckBox(panel, label="--nooutput")
        self.chkStream = wx.CheckBox(panel, label="--stream")
        self.chkSilent = wx.CheckBox(panel, label="--silent")

        checkboxes = [
            self.chkSim, self.chkCamera, self.chkRAM,
            self.chkSpeak, self.chkViewer, self.chkForever,
            self.chkNoOut, self.chkStream, self.chkSilent,
        ]
        for c in checkboxes:
            grid_flags.Add(c, 0, wx.ALL, 2)

        s_flags.Add(grid_flags, 0, wx.ALL, 8)
        main.Add(s_flags, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

        # ===================== DEVICES =====================
        sb_dev = wx.StaticBox(panel, label="Hardware Devices")
        s_dev = wx.StaticBoxSizer(sb_dev, wx.VERTICAL)
        #grid_dev = wx.GridSizer(3, 3, 5, 5)
        grid_dev = wx.FlexGridSizer(0, 3, 5, 5)


        self.chkArduino = wx.CheckBox(panel, label="--distance (Arduino)")
        self.chkTeensy = wx.CheckBox(panel, label="--accelerometer (Teensy)")
        self.chkForce = wx.CheckBox(panel, label="--force")
        self.chkFeatures = wx.CheckBox(panel, label="--features")
        self.chkDLight = wx.CheckBox(panel, label="--dlight")
        self.chkRLight = wx.CheckBox(panel, label="--rlight")
        self.chkTPattern = wx.CheckBox(panel, label="--tlight")

        for c in (self.chkArduino, self.chkTeensy, self.chkForce,
                  self.chkFeatures, self.chkDLight, self.chkRLight,
                  self.chkTPattern):
            grid_dev.Add(c, 0, wx.ALL, 2)

        s_dev.Add(grid_dev, 0, wx.ALL, 8)
        main.Add(s_dev, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

        # ===================== CAMERA SETTINGS =====================
        sb_cam = wx.StaticBox(panel, label="Camera Settings")
        s_cam = wx.StaticBoxSizer(sb_cam, wx.VERTICAL)

        #grid_cam = wx.FlexGridSizer(5, 2, 8, 10)
        grid_cam = wx.FlexGridSizer(0, 2, 8, 10)  # auto rows, 2 columns
        grid_cam.AddGrowableCol(1, 1)

        def labeled(label, default=""):
            grid_cam.Add(wx.StaticText(panel, label=label),
                         0, wx.ALIGN_CENTER_VERTICAL)
            ctrl = wx.TextCtrl(panel, value=str(default))
            grid_cam.Add(ctrl, 1, wx.EXPAND)
            return ctrl

        self.txtWidth = labeled("Width:", "640")
        self.txtHeight = labeled("Height:", "480")
        self.txtExposure = labeled("Exposure (µs):", "10000")
        self.txtGain = labeled("Gain:", "0")
        self.txtFPS = labeled("FPS:", "10")
        self.txtBlack = labeled("Black Level:", "0")
        self.txtDuration = labeled("Duration (s):", "5")
        self.txtCountdown = labeled("Countdown (s):", "0")

        s_cam.Add(grid_cam, 0, wx.EXPAND | wx.ALL, 8)
        main.Add(s_cam, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

        # ===================== START BUTTON =====================
        self.btnStart = wx.Button(panel, label="Start")
        self.btnStart.Bind(wx.EVT_BUTTON, self.onStart)
        main.Add(self.btnStart, 0, wx.ALIGN_CENTER | wx.ALL, 10)

        # ===================== CONSOLE OUTPUT =====================
        sb_console = wx.StaticBox(panel, label="Console Output")
        s_console = wx.StaticBoxSizer(sb_console, wx.VERTICAL)

        self.console = wx.TextCtrl(panel,
                                   style=wx.TE_MULTILINE |
                                         wx.TE_READONLY |
                                         wx.TE_RICH2)
        s_console.Add(self.console, 1, wx.EXPAND | wx.ALL, 5)
        main.Add(s_console, 1, wx.EXPAND | wx.ALL, 10)

        panel.SetSizer(main)
        self.Center()

    # =============================================================
    # Handlers
    # =============================================================

    def onBrowse(self, _):
        dlg = wx.DirDialog(self, "Choose output directory")
        if dlg.ShowModal() == wx.ID_OK:
            self.dirCtrl.SetValue(dlg.GetPath())
        dlg.Destroy()

    def onStart(self, _):
        # Build the command
        cmd = ["./magician_grabber"]

        # Output directory
        if self.dirCtrl.GetValue().strip():
            cmd += ["--output", self.dirCtrl.GetValue().strip()]

        # Flags
        def add(flag, ctrl):
            if ctrl.GetValue():
                cmd.append(flag)

        add("--simulate", self.chkSim)
        add("--camera", self.chkCamera)
        add("--ram", self.chkRAM)
        add("--speak", self.chkSpeak)
        add("--view", self.chkViewer)
        add("--forever", self.chkForever)
        add("--nooutput", self.chkNoOut)
        add("--silent", self.chkSilent)
        add("--stream", self.chkStream)

        # Devices
        add("--distance", self.chkArduino)
        add("--accelerometer", self.chkTeensy)
        add("--force", self.chkForce)
        add("--features", self.chkFeatures)
        add("--dlight", self.chkDLight)
        add("--rlight", self.chkRLight)
        add("--tlight", self.chkTPattern)

        # Camera params
        def addn(flag, txt):
            v = txt.GetValue().strip()
            if v:
                cmd += [flag, v]

        addn("--size", f"{self.txtWidth.GetValue()} {self.txtHeight.GetValue()}")
        addn("--exposure", self.txtExposure)
        addn("--gain", self.txtGain)
        addn("--fps", self.txtFPS)
        addn("--blacklevel", self.txtBlack)
        addn("--duration", self.txtDuration)
        addn("--countdown", self.txtCountdown)

        # Lock GUI
        self.enableControls(False)
        self.console.SetValue("")
        self.console.AppendText("Running: " + " ".join(cmd) + "\n")

        # Start process thread
        threading.Thread(target=self.runProcess, args=(cmd,), daemon=True).start()

    def enableControls(self, en):
        for child in self.GetChildren():
            if isinstance(child, wx.Button) or isinstance(child, wx.CheckBox) or isinstance(child, wx.TextCtrl):
                child.Enable(en)

    def runProcess(self, cmd):
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True)

            for line in proc.stdout:
                wx.PostEvent(self, ProcessOutputEvent(text=line))

            proc.wait()
            wx.PostEvent(self, ProcessOutputEvent(text="\nProcess finished.\n"))

        except Exception as e:
            wx.PostEvent(self, ProcessOutputEvent(text=f"ERROR: {e}\n"))

        wx.CallAfter(self.enableControls, True)

    def onProcessOutput(self, evt):
        self.console.AppendText(evt.text)


class App(wx.App):
    def OnInit(self):
        frame = MagicianGrabberFrame()
        frame.Show()
        return True


if __name__ == "__main__":
    app = App(False)
    app.MainLoop()

