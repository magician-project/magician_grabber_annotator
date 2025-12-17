#!/usr/bin/python3

""" 
Author : "Ammar Qammaz"
Copyright : "2025 Foundation of Research and Technology, Computer Science Department Greece, See license.txt"
License : "FORTH" 
"""


# pip install requests --user

import os
import sys
import json
import requests
import threading
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse


# ------------------ Utilities ------------------

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def load_credentials(credentials="server.json"):
    if os.path.exists(credentials):
        try:
            with open(credentials, "r") as f:
                data = json.load(f)
                return data.get("username", ""), data.get("password", "")
        except Exception:
            pass
    return "", ""


def auth_urlWHY(url, username, password):
    if not username or not password:
        return url

    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    query["username"] = username
    query["password"] = password
    return urlunparse(parsed._replace(query=urlencode(query, doseq=True)))


def auth_url(provider, dataset, username, password):
    if not username or not password:
        return provider

    url = "%s?username=%s&password=%s&dataset=%s" % (provider,username,password,dataset)
    #print("auth_url = ",url)
    return url



# ------------------ ZIP Retrieval ------------------
def retrieve_annotation_zips(base_url, datasetnameFull, local_dir="downloads"):
    """
    Retrieves all PUBLIC annotation zip files from base_url
    that match the given dataset name.
    """
    os.makedirs(local_dir, exist_ok=True)

    datasetname = os.path.basename(datasetnameFull)
    print(f"Trying to retrieve extra annotation zips from {base_url}")

    # 🔓 PUBLIC access (no authentication)
    resp = requests.get(base_url)
    if resp.status_code != 200:
        raise RuntimeError(f"Cannot access {base_url}: {resp.status_code}")

    recoveredAnnotations = False
    html = resp.text

    for line in html.splitlines():
        if "<a href=" not in line:
            continue

        start = line.find('<a href="') + 9
        end = line.find('"', start)
        if start <= 8 or end <= start:
            continue

        fname = line[start:end]
        if not fname.endswith(".zip"):
            continue

        parts = fname[:-4].split("_")
        if len(parts) < 5:
            continue

        # upload_user_dataset_date_time.zip
        dset = "_".join(parts[2:-2])
        if dset != datasetname:
            continue

        zip_url = base_url.rstrip("/") + "/" + fname
        print(f"Downloading annotation ZIP: {zip_url}")

        zresp = requests.get(zip_url)
        if zresp.status_code != 200:
            print(f"Failed to download {zip_url}")
            continue

        import zipfile
        from io import BytesIO

        with zipfile.ZipFile(BytesIO(zresp.content)) as zf:
            zf.extractall(local_dir)

        os.system(
            f"mv {local_dir}/{datasetname}/*.json {local_dir}/ 2>/dev/null && "
            f"rmdir {local_dir}/{datasetname}/ 2>/dev/null"
        )

        recoveredAnnotations = True

    return recoveredAnnotations



# ------------------ HTTP Folder Streamer ------------------

class HTTPFolderStreamer:
    def __init__(self, provider=None, dataset=None, local_dir="http_cache",
                 label="colorFrame_0_", retrieve_zip=False,
                 credentials="server.json"):

        self.username, self.password = load_credentials(credentials)

        self.provider = provider
        self.dataset  = dataset + "/"

        self.local_dir = local_dir
        self.label = label
        self.frameNumber = 0
        self.should_stop = False
        self.metadata = None
        self.file_list = []
        self.index = 0
        self._prefetch_thread = None

        os.makedirs(self.local_dir, exist_ok=True)

        if self.provider:
            self.loadNewDataset(self.dataset)
            self.getInfo()
            self.getControllerInfo()
            self.getCameraInfo()
            self.getTactileData()

            if retrieve_zip:
                retrieve_annotation_zips(
                    "http://ammar.gr/magician/uploads/",
                    self.local_dir,
                    self.local_dir
                )

    # ---------- Dataset loading ----------

    def loadNewDataset(self, dataset):

        url = auth_url(self.provider, dataset, self.username, self.password)
        print(f"Loading file list from: {url}")
        resp = requests.get(url)
        if resp.status_code != 200:
            raise RuntimeError(f"Cannot access {url}: {resp.status_code}")

        html = resp.text
        image_extensions = ['.png', '.pnm', '.jpg', '.jpeg']
        files = []

        for line in html.splitlines():
            if "<a href=" not in line:
                continue

            start = line.find('<a href="') + 9
            end = line.find('"', start)
            if start <= 8 or end <= start:
                continue

            fname = line[start:end]
            if any(fname.lower().endswith(ext) for ext in image_extensions):
                if "foreground.png" not in fname:
                    files.append(fname)

        files.sort()
        self.file_list = files
        self.index = 0

    # ---------- Navigation ----------

    def current(self):
        return self.index

    def max(self):
        return len(self.file_list)

    def next(self):
        self.index = (self.index + 1) % len(self.file_list)

    def previous(self):
        self.index = (self.index - 1) % len(self.file_list)

    def select(self, item):
        if 0 <= item < len(self.file_list):
            self.index = item

    # ---------- Downloads ----------

    def _download_file(self, remote_name, overwrite=False):
        local_path = os.path.join(self.local_dir, remote_name)

        if not overwrite and os.path.isfile(local_path):
            return local_path

        url = auth_url(self.provider, self.dataset + remote_name,  self.username, self.password)
        print(f"Downloading: {url}")

        resp = requests.get(url)
        if resp.status_code != 200:
            raise RuntimeError(f"Failed to download {url}")

        with open(local_path, "wb") as f:
            f.write(resp.content)

        return local_path

    def _prefetch_next(self):
        next_index = (self.index + 1) % len(self.file_list)
        next_img = self.file_list[next_index]
        self._download_file(next_img)

        json_name = os.path.splitext(next_img)[0] + ".pnm.json"
        try:
            self._download_file(json_name)
        except RuntimeError:
            pass

    # ---------- Metadata ----------

    def getJSON(self):
        img_name = self.file_list[self.index]
        json_name =  os.path.splitext(img_name)[0] + ".pnm.json"
        try:
            return self._download_file(json_name)
        except RuntimeError:
            print(f"There is no JSON file for item {self.index}")
            return None

    def getInfo(self):
        try:
            return self._download_file("info.json")
        except RuntimeError:
            print("There is no Info file for dataset")
            return None

    def getControllerInfo(self):
        try:
            return self._download_file("controller.csv")
        except RuntimeError:
            print("There is no Controller file for dataset")
            return None

    def getCameraInfo(self):
        try:
            return self._download_file("camera.csv")
        except RuntimeError:
            print("There is no Camera file for dataset")
            return None

    def getTactileData(self):
        tactile_files = [
            "acceleration_psd.csv",
            "acceleration_spikeness.csv",
            "accelerometer.csv",
            "force.csv",
            "force_psd.csv",
            "friction.csv"
        ]

        tactile_base_url = auth_url( self.provider, self.dataset+"/tactile/", self.username, self.password )

        tactile_local_dir = os.path.join(self.local_dir, "tactile")
        os.makedirs(tactile_local_dir, exist_ok=True)

        downloaded = []

        for fname in tactile_files:
            local_path = os.path.join(tactile_local_dir, fname)
            if os.path.isfile(local_path):
                downloaded.append(local_path)
                continue

            url = tactile_base_url + fname
            print(f"Downloading tactile file: {url}")
            resp = requests.get(url)

            if resp.status_code == 200:
                with open(local_path, "wb") as f:
                    f.write(resp.content)
                downloaded.append(local_path)
            else:
                print(f"Warning: could not download {url}")

        if not downloaded:
            os.system(f"rmdir {tactile_local_dir} 2>/dev/null")
        else:
            print(f"Downloaded {len(downloaded)} tactile files")

        return downloaded

    # ---------- Images ----------

    def getImageSimple(self):
        return self._download_file(self.file_list[self.index])

    def getImage(self):
        img_name = self.file_list[self.index]
        img_path = self._download_file(img_name)

        if self._prefetch_thread is None or not self._prefetch_thread.is_alive():
            self._prefetch_thread = threading.Thread(
                target=self._prefetch_next, daemon=True
            )
            self._prefetch_thread.start()

        return img_path

    def saveJSON(self):
        print("Updated JSON (Doing nothing)..")


# ------------------ Test ------------------

if __name__ == '__main__':
    print("HTTP Folder Stream tester..")

    test = HTTPFolderStreamer(
        "http://ammar.gr/magician/datasets/40-positive-class-a/"
    )

    print("Max files:", test.max())
    print("Image:", test.getImage())
    print("JSON:", test.getJSON())
