import os
import wx
import requests

class DatasetSelector(wx.Dialog):
    def __init__(self,  local_base_path="./", parent=None):
        super().__init__(parent, title="Dataset Selector", size=(500, 200))

        self.selectedDataset = None  # this will hold the final choice
        self.replaceAnnotations = False  # <-- flag for the checkbox
        self.local_base_path = local_base_path  # local directory base to check

        panel = wx.Panel(self)
        vbox = wx.BoxSizer(wx.VERTICAL)

        # URL input
        hbox1 = wx.BoxSizer(wx.HORIZONTAL)
        self.url_label = wx.StaticText(panel, label="Server URL:")
        self.url_input = wx.TextCtrl(panel, value="http://ammar.gr/magician/datasets/")
        hbox1.Add(self.url_label, flag=wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, border=8)
        hbox1.Add(self.url_input, proportion=1)
        vbox.Add(hbox1, flag=wx.EXPAND | wx.ALL, border=10)

        # Dropdown for directories
        hbox2 = wx.BoxSizer(wx.HORIZONTAL)
        self.dropdown = wx.Choice(panel)
        hbox2.Add(wx.StaticText(panel, label="Select dataset:"), 
                  flag=wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, border=8)
        hbox2.Add(self.dropdown, proportion=1)
        vbox.Add(hbox2, flag=wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, border=10)

        # Checkbox for annotation replacement
        self.checkbox = wx.CheckBox(panel, label="Replace annotations with live snapshot")
        vbox.Add(self.checkbox, flag=wx.LEFT | wx.RIGHT | wx.TOP, border=10)


        # Buttons
        hbox3 = wx.BoxSizer(wx.HORIZONTAL)
        self.fetch_button  = wx.Button(panel, label="Connect To Server")
        self.select_button = wx.Button(panel, label="OK")
        self.cancel_button = wx.Button(panel, label="Cancel")
        hbox3.Add(self.fetch_button, flag=wx.RIGHT, border=5)
        hbox3.Add(self.select_button, flag=wx.RIGHT, border=5)
        hbox3.Add(self.cancel_button)
        vbox.Add(hbox3, flag=wx.ALIGN_CENTER | wx.ALL, border=10)

        panel.SetSizer(vbox)

        # Bind events
        self.fetch_button.Bind(wx.EVT_BUTTON, self.onFetchDirectories)
        self.select_button.Bind(wx.EVT_BUTTON, self.onSelectDataset)
        self.cancel_button.Bind(wx.EVT_BUTTON, lambda evt: self.EndModal(wx.ID_CANCEL))
        self.dropdown.Bind(wx.EVT_CHOICE, self.onDropdownSelection)

    def onDropdownSelection(self, event):
        selection = self.dropdown.GetStringSelection()
        self.updateCheckboxState(selection)

    def updateCheckboxState(self, selection):
        local_path = os.path.join(self.local_base_path, selection)
        if os.path.exists(local_path):
            #self.checkbox.Disable()
            self.checkbox.SetValue(False)
        else:
            #self.checkbox.Enable()
            self.checkbox.SetValue(True)


    def onFetchDirectories(self, event):
        base_url = self.url_input.GetValue().strip()
        if not base_url.endswith("/"):
            base_url += "/"
        try:
            resp = requests.get(base_url)
            resp.raise_for_status()
            dirs = []
            for line in resp.text.splitlines():
                if "<a href=" in line:
                    start = line.find('<a href="') + 9
                    end = line.find('"', start)
                    if start > 8 and end > start:
                        fname = line[start:end]
                        if fname.endswith("/") and not fname.startswith("../"):
                            dirs.append(fname.strip("/"))
            dirs.sort()
            self.dropdown.SetItems(dirs)
            if dirs:
                self.dropdown.SetSelection(0)
        except Exception as e:
            wx.MessageBox(f"Error fetching directories: {e}", "Error", wx.OK | wx.ICON_ERROR)

    def onSelectDataset(self, event):
        selection = self.dropdown.GetStringSelection()
        if selection:
            base_url = self.url_input.GetValue().strip()
            if not base_url.endswith("/"):
                base_url += "/"
            self.selectedDataset   = base_url + selection + "/"
            self.selectedDirectory = selection
            self.replaceAnnotations = self.checkbox.GetValue()  # <-- read checkbox state
            self.EndModal(wx.ID_OK)
        else:
            wx.MessageBox("No dataset selected", "Info", wx.OK | wx.ICON_INFORMATION)

