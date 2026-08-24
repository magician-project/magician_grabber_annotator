import os
import threading
import time
import json
import wx

from mga.paths import repo_root


"""
Check if a file exists
"""
def checkIfFileExists(filename):
    if (filename is None):  
       return False
    return os.path.isfile(filename) 

class UploadDialog(wx.Dialog):
    def __init__(self, parent, zip_path, dataset, credentials=os.path.join(repo_root(), "server.json")):
        super().__init__(parent, title="Upload Annotations", size=(350, 200))
        self.zip_path = zip_path  # path to the zip file
        self.dataset  = dataset
        self.credentials = credentials

        if (not checkIfFileExists(zip_path)):
            wx.MessageBox(f"Could not find zip file %s with annotations :(" % zip_path, "Error", wx.OK | wx.ICON_ERROR)
            return

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




def upload_dataset_annotations(parent, base_dir, local_dir):
    """Zip the per-frame annotation JSONs + info.json of `local_dir` and open the
    UploadDialog (the body of wx_annotator.onUploadAnnotations).

    base_dir is the datasets ROOT shared by every dataset; the zip lands there, so
    the name carries our PID: two annotator instances uploading different datasets
    at the same time would otherwise build and rm -f the same upload.zip."""
    print("Local Dir: ", local_dir)

    # e.g. /media/ammar/games2/Datasets/Magician
    zip_path = os.path.join(base_dir, "upload_%d.zip" % os.getpid())
    rel_dir  = os.path.basename(local_dir.rstrip("/"))
    # rel_dir should be "AltinayKapoDefect"

    # zip APPENDS to an existing archive — start fresh, otherwise previously-uploaded
    # datasets accumulate in the same upload_<pid>.zip (the name is stable for the
    # lifetime of this instance, so a second upload from it would pile up).
    try:
        if os.path.isfile(zip_path):
            os.remove(zip_path)
    except Exception as e:
        print("Could not remove stale zip:", zip_path, e)

    # Include the per-frame annotation JSONs AND the (finalized) info.json for this dataset only.
    zipCommand = (
        f'cd "{base_dir}" && '
        f'zip "{zip_path}" -b "{base_dir}" "{rel_dir}"/color*.json "{rel_dir}"/info.json'
    )

    print("Zip command : ", zipCommand)
    os.system(zipCommand)

    dlg = UploadDialog(parent, zip_path, local_dir)
    dlg.ShowModal()
    dlg.Destroy()
    os.system(f'rm -f "{zip_path}"')
