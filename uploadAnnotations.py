import os
import threading
import time
import wx



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


