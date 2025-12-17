#!/usr/bin/python3

""" 
Author : "Ammar Qammaz"
Copyright : "2022 Foundation of Research and Technology, Computer Science Department Greece, See license.txt"
License : "FORTH" 
"""

import os
import wx
import requests
import json
from urllib.parse import urlencode, urlparse, parse_qs, urlunparse

class DatasetSelector(wx.Dialog):
    def __init__(self, local_base_path="./", parent=None, credentials="server.json"):
        super().__init__(parent, title="Dataset Selector", size=(500, 280))

        self.selectedDataset  = None
        self.selectedProvider = "?"
        self.replaceAnnotations = False
        self.local_base_path = local_base_path
        self.credentials = credentials

        # Load saved credentials
        saved_user, saved_pwd = self.load_credentials()

        panel = wx.Panel(self)
        vbox = wx.BoxSizer(wx.VERTICAL)

        # ---- Server URL ----
        hbox_url = wx.BoxSizer(wx.HORIZONTAL)
        hbox_url.Add(wx.StaticText(panel, label="Server URL:"), 0,
                     wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, 8)
        self.url_input = wx.TextCtrl(
            panel, value="http://ammar.gr/magician/download.php"
        )
        hbox_url.Add(self.url_input, 1)
        vbox.Add(hbox_url, 0, wx.EXPAND | wx.ALL, 10)

        # ---- Username ----
        hbox_user = wx.BoxSizer(wx.HORIZONTAL)
        hbox_user.Add(wx.StaticText(panel, label="Username:"), 0,
                      wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, 8)
        self.username = wx.TextCtrl(panel, value=saved_user)
        hbox_user.Add(self.username, 1)
        vbox.Add(hbox_user, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, 5)

        # ---- Password ----
        hbox_pwd = wx.BoxSizer(wx.HORIZONTAL)
        hbox_pwd.Add(wx.StaticText(panel, label="Password:"), 0,
                     wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, 8)
        self.password = wx.TextCtrl(panel, style=wx.TE_PASSWORD, value=saved_pwd)
        hbox_pwd.Add(self.password, 1)
        vbox.Add(hbox_pwd, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, 5)

        # ---- Dataset dropdown ----
        hbox_ds = wx.BoxSizer(wx.HORIZONTAL)
        hbox_ds.Add(wx.StaticText(panel, label="Select dataset:"), 0,
                    wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, 8)
        self.dropdown = wx.Choice(panel)
        hbox_ds.Add(self.dropdown, 1)
        vbox.Add(hbox_ds, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, 10)

        # ---- Checkbox ----
        self.checkbox = wx.CheckBox(
            panel,
            label="Overwrite/Replace local annotations with server annotations"
        )
        vbox.Add(self.checkbox, 0, wx.LEFT | wx.RIGHT | wx.TOP, 10)

        # ---- Buttons ----
        hbox_btn = wx.BoxSizer(wx.HORIZONTAL)
        self.fetch_button = wx.Button(panel, label="Connect To Server")
        self.select_button = wx.Button(panel, label="OK")
        self.cancel_button = wx.Button(panel, label="Cancel")
        hbox_btn.Add(self.fetch_button, 0, wx.RIGHT, 5)
        hbox_btn.Add(self.select_button, 0, wx.RIGHT, 5)
        hbox_btn.Add(self.cancel_button, 0)
        vbox.Add(hbox_btn, 0, wx.ALIGN_CENTER | wx.ALL, 10)

        panel.SetSizer(vbox)

        # Bind events
        self.fetch_button.Bind(wx.EVT_BUTTON, self.onFetchDirectories)
        self.select_button.Bind(wx.EVT_BUTTON, self.onSelectDataset)
        self.cancel_button.Bind(wx.EVT_BUTTON, lambda evt: self.EndModal(wx.ID_CANCEL))
        self.dropdown.Bind(wx.EVT_CHOICE, self.onDropdownSelection)

    # ---------- Credentials ----------
    def load_credentials(self):
        if os.path.exists(self.credentials):
            try:
                with open(self.credentials, "r") as f:
                    data = json.load(f)
                    return data.get("username", ""), data.get("password", "")
            except Exception:
                pass
        return "", ""

    def save_credentials(self, username, password):
        try:
            with open(self.credentials, "w") as f:
                json.dump({"username": username, "password": password}, f)
        except Exception:
            pass

    # ---------- Helpers ----------
    def build_url_with_auth(self, base_url):
        user = self.username.GetValue().strip()
        pwd = self.password.GetValue().strip()

        if not user or not pwd:
            raise ValueError("Username and password required")

        parsed = urlparse(base_url)
        query = parse_qs(parsed.query)
        query["username"] = user
        query["password"] = pwd

        new_query = urlencode(query, doseq=True)
        return urlunparse(parsed._replace(query=new_query))

    def onDropdownSelection(self, event):
        selection = self.dropdown.GetStringSelection()
        local_path = os.path.join(self.local_base_path, selection)
        self.checkbox.SetValue(not os.path.exists(local_path))

    # ---------- Network ----------
    def onFetchDirectories(self, event):
        try:
            base_url = self.url_input.GetValue().strip()
            url = self.build_url_with_auth(base_url)

            resp = requests.get(url)
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

            # Save credentials after successful connection
            self.save_credentials(
                self.username.GetValue().strip(),
                self.password.GetValue().strip()
            )

        except Exception as e:
            wx.MessageBox(f"Error connecting to server:\n{e}",
                          "Error", wx.OK | wx.ICON_ERROR)

    def onSelectDataset(self, event):
        selection = self.dropdown.GetStringSelection()
        if not selection:
            wx.MessageBox("No dataset selected", "Info",
                          wx.OK | wx.ICON_INFORMATION)
            return

        base_url = self.url_input.GetValue().strip()
        self.selectedProvider = base_url
        url = self.build_url_with_auth(base_url)
        if not url.endswith("/"):
            url += "/"

        self.selectedDataset = selection
        self.selectedDirectory = selection
        self.replaceAnnotations = self.checkbox.GetValue()
        self.EndModal(wx.ID_OK)
