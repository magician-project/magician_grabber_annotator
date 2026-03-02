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
        super().__init__(parent, title="Dataset Selector", size=(560, 420))

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
            panel, value="http://ammar.gr/magician/download2.php"
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
        self.check_button = wx.Button(panel, label="Check Annotations")
        self.select_button = wx.Button(panel, label="OK")
        self.cancel_button = wx.Button(panel, label="Cancel")
        hbox_btn.Add(self.fetch_button, 0, wx.RIGHT, 5)
        hbox_btn.Add(self.check_button, 0, wx.RIGHT, 5)
        hbox_btn.Add(self.select_button, 0, wx.RIGHT, 5)
        hbox_btn.Add(self.cancel_button, 0)
        vbox.Add(hbox_btn, 0, wx.ALIGN_CENTER | wx.ALL, 10)

        # ---- Results area ----
        vbox.Add(wx.StaticText(panel, label="Annotation status:"),
                 0, wx.LEFT | wx.RIGHT | wx.TOP, 10)
        self.results_box = wx.TextCtrl(
            panel,
            style=wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_RICH2,
            size=(-1, 80)
        )
        vbox.Add(self.results_box, 1, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

        panel.SetSizer(vbox)

        # Bind events
        self.fetch_button.Bind(wx.EVT_BUTTON, self.onFetchDirectories)
        self.check_button.Bind(wx.EVT_BUTTON, self.onCheckAnnotations)
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

    def build_url_with_auth_and_dataset(self, base_url, dataset):
        """Append username/password and dataset=... query params to provider URL."""
        user = self.username.GetValue().strip()
        pwd = self.password.GetValue().strip()

        if not user or not pwd:
            raise ValueError("Username and password required")

        parsed = urlparse(base_url)
        query = parse_qs(parsed.query)
        query["username"] = user
        query["password"] = pwd
        query["dataset"] = dataset

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

    # ---------- Annotation checks ----------
    def _parse_anchor_filenames(self, html_text):
        files = []
        for line in html_text.splitlines():
            if "<a href=" not in line:
                continue
            start = line.find('<a href="') + 9
            end = line.find('"', start)
            if start <= 8 or end <= start:
                continue
            fname = line[start:end]
            if fname and not fname.startswith("../"):
                files.append(fname)
        return files

    def _annotation_status_for_dataset(self, provider_url, dataset):
        """
        A dataset is considered annotated if:
          - it has .json annotations alongside the color frames in the dataset listing, OR
          - it has at least one PUBLIC annotation ZIP in /magician/uploads/
            (see HTTPStream.retrieve_annotation_zips)

        Returns (has_any_annotations, annotated_count, total_images, has_zip, image_formats).
        annotated_count is computed by matching per-frame JSONs either in the dataset listing
        OR inside the latest matching annotation ZIP.
        """
        dataset_clean = str(dataset).strip().rstrip("/")

        # --- 1) List dataset folder contents from provider (authenticated) ---
        url = self.build_url_with_auth_and_dataset(provider_url, dataset_clean)
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()

        files = self._parse_anchor_filenames(resp.text)
        file_set = set(files)

        image_exts = (".png", ".pnm", ".jpg", ".jpeg")
        images = [
            f for f in files
            if f.lower().endswith(image_exts) and ("foreground.png" not in f.lower())
        ]
        total_images = len(images)

        # Image formats present in dataset (png/jpg/pnm/etc)
        image_formats = sorted({os.path.splitext(f)[1].lstrip(".").lower() for f in images})

        # --- 2) Find matching annotation ZIPs (public uploads) ---
        has_zip = False
        zip_json_set = set()

        try:
            from urllib.parse import urlparse
            import zipfile
            from io import BytesIO

            parsed = urlparse(provider_url)
            uploads_base = f"{parsed.scheme}://{parsed.netloc}/magician/uploads/"

            uresp = requests.get(uploads_base, timeout=30)
            if uresp.status_code == 200:
                up_files = self._parse_anchor_filenames(uresp.text)

                matches = []
                for fname in up_files:
                    if not fname.lower().endswith(".zip"):
                        continue
                    parts = fname[:-4].split("_")
                    # upload_user_dataset_date_time.zip  -> len >= 5
                    if len(parts) < 5 or parts[0] != "upload":
                        continue
                    dset = "_".join(parts[2:-2])
                    if dset != dataset_clean:
                        continue
                    dt = parts[-2] + "_" + parts[-1]  # YYYYMMDD_HHMMSS (string-sortable)
                    matches.append((dt, fname))

                if matches:
                    has_zip = True

                    # Download only the latest zip to avoid excessive downloads
                    matches.sort(key=lambda x: x[0], reverse=True)
                    latest_zip = matches[0][1]
                    zip_url = uploads_base.rstrip("/") + "/" + latest_zip

                    zresp = requests.get(zip_url, timeout=60)
                    if zresp.status_code == 200:
                        with zipfile.ZipFile(BytesIO(zresp.content)) as zf:
                            for zi in zf.infolist():
                                if zi.filename.lower().endswith(".json"):
                                    zip_json_set.add(os.path.basename(zi.filename))
        except Exception:
            # If uploads check fails, we still can count from in-folder JSONs
            pass

        # --- 3) Count annotated images (union of in-folder json + latest zip json) ---
        annotated = 0
        for img in images:
            stem, ext = os.path.splitext(img)

            # Patterns we consider annotated:
            #  - frame.png.json (i.e. original filename + ".json")
            #  - frame.json
            #  - frame.pnm.json (legacy, regardless of image extension)
            candidates = {
                img + ".json",
                stem + ".json",
                stem + ".pnm.json",
            }

            if (candidates & file_set) or (set(map(os.path.basename, candidates)) & zip_json_set):
                annotated += 1

        # "Annotated" per your rule: any per-frame json OR any matching zip exists
        has_any = (annotated > 0) or has_zip
        return has_any, annotated, total_images, has_zip, image_formats


    def onCheckAnnotations(self, event):
     """Check all datasets in the dropdown for presence of annotations.

     A dataset is considered annotated if:
      - it has per-frame .json annotations alongside color frames, OR
      - it has an annotation .zip available (per HTTPStream naming rules).
     """
     try:
        provider_url = self.url_input.GetValue().strip()
        datasets = list(self.dropdown.GetItems())

        if not datasets:
            wx.MessageBox(
                "No datasets loaded yet. Click 'Connect To Server' first.",
                "Info",
                wx.OK | wx.ICON_INFORMATION
            )
            return

        # UX: show something immediately
        self.results_box.SetValue("Checking annotations...\n")
        self.results_box.Update()

        lines = []

        # Frame-level totals
        total_frames = 0
        total_annotated_frames = 0

        # Dataset-level totals
        total_datasets = len(datasets)
        annotated_datasets = 0
        not_annotated_datasets = 0

        for dset in datasets:
            has_any, ann, tot, has_zip, fmts = self._annotation_status_for_dataset(provider_url, dset)

            fmt_note = "unknown" if not fmts else (fmts[0] if len(fmts) == 1 else "mixed[" + ",".join(fmts) + "]")

            # Frame totals
            total_annotated_frames += int(ann)
            total_frames += int(tot)

            # Dataset totals (uses your rule: per-frame json OR zip exists)
            dataset_is_annotated = bool(has_any)
            if dataset_is_annotated:
                annotated_datasets += 1
            else:
                not_annotated_datasets += 1

            # Per-dataset percentage (frame completion)
            pct_str = "N/A" if tot == 0 else f"{(100.0 * ann / tot):.1f}%"

            mark = "✅" if dataset_is_annotated else "❌"
            zip_note = " (zip)" if (has_zip and ann == 0) else ""
            lines.append(f"{mark} {dset}{zip_note}  —  {pct_str} ({ann}/{tot})  —  images: {fmt_note}")

        # Overall frame completion
        overall_frames_pct = "N/A" if total_frames == 0 else f"{(100.0 * total_annotated_frames / total_frames):.1f}%"

        # Overall dataset completion
        overall_dsets_pct = "N/A" if total_datasets == 0 else f"{(100.0 * annotated_datasets / total_datasets):.1f}%"

        lines.append("")
        lines.append(f"Overall frames: {overall_frames_pct} ({total_annotated_frames}/{total_frames})")
        lines.append(
            f"Overall datasets: {overall_dsets_pct} ({annotated_datasets}/{total_datasets})"
            f" — {not_annotated_datasets} not annotated"
        )

        self.results_box.SetValue("\n".join(lines))

     except Exception as e:
        wx.MessageBox(
            f"Error checking annotations:\n{e}",
            "Error",
            wx.OK | wx.ICON_ERROR
        )

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
