# pip install requests --user

import os
import sys
import json
import requests
import threading

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def retrieve_annotation_zips(base_url, datasetname, local_dir="downloads"):
    """
    Retrieves all zip files from base_url that match the given datasetname.
    Downloads and unzips them into local_dir.
    """
    os.makedirs(local_dir, exist_ok=True)

    print(f"Trying to retrieve extra annotation zips from {base_url}")
    resp = requests.get(base_url)
    if resp.status_code != 200:
        raise RuntimeError(f"Cannot access {base_url}: {resp.status_code}")

    html = resp.text
    for line in html.splitlines():
        if "<a href=" in line:
            start = line.find('<a href="') + 9
            end = line.find('"', start)
            if start > 8 and end > start:
                fname = line[start:end]
                if not fname.endswith(".zip"):
                    continue

                print("Found ZIP entry:", fname)

                parts = fname[:-4].split("_")  # strip .zip first
                if len(parts) < 5:
                    continue

                # format: upload_user_datasetname_date_time
                user = parts[1]
                date = parts[-2]
                time = parts[-1]
                dset = "_".join(parts[2:-2])  # <-- everything between user and date

                if dset == datasetname:
                    zip_url = base_url.rstrip("/") + "/" + fname
                    print(f"Downloading zip for dataset '{datasetname}': {zip_url}")
                    zresp = requests.get(zip_url)
                    if zresp.status_code != 200:
                        print(f"Failed to download {zip_url}")
                        continue
                    import zipfile
                    from io import BytesIO
                    with zipfile.ZipFile(BytesIO(zresp.content)) as zf:
                        zf.extractall(local_dir)
                    print(f"Extracted {fname} into {local_dir}")
                    os.system(f"mv {local_dir}/{local_dir}/*.json {local_dir}/ && rmdir {local_dir}/{local_dir}/")




class HTTPFolderStreamer:
    def __init__(self, base_url=None, local_dir="http_cache", label="colorFrame_0_", retrieve_zip=False):
        self.base_url = base_url.rstrip("/") + "/" if base_url else None
        self.local_dir = local_dir
        self.label = label
        self.frameNumber = 0
        self.should_stop = False
        self.metadata = None
        self.file_list = []
        self.index = 0
        os.makedirs(self.local_dir, exist_ok=True)
        self._prefetch_thread = None
        if self.base_url:
            self.loadNewDataset(self.base_url)
            self.getInfo()
            if (retrieve_zip):
               retrieve_annotation_zips("http://ammar.gr/magician/uploads/", local_dir, local_dir)

    def loadNewDataset(self, url):
        print(f"Loading file list from: {url}")
        resp = requests.get(url)
        if resp.status_code != 200:
            raise RuntimeError(f"Cannot access {url}: {resp.status_code}")
        html = resp.text
        image_extensions = ['.png', '.pnm', '.jpg', '.jpeg']
        files = []
        for line in html.splitlines():
            if "<a href=" in line:
                start = line.find('<a href="') + 9
                end = line.find('"', start)
                if start > 8 and end > start:
                    fname = line[start:end]
                    if any(fname.lower().endswith(ext) for ext in image_extensions):
                        if "foreground.png" not in fname:
                            files.append(fname)
        files.sort()
        self.file_list = files
        self.index = 0



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

    def _download_file(self, remote_name):
        local_path = os.path.join(self.local_dir, remote_name)
        if not os.path.isfile(local_path):
            url = self.base_url + remote_name
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
            pass  # No JSON is fine


    def getJSON(self):
        img_name = self.file_list[self.index]
        json_name = os.path.splitext(img_name)[0] + ".pnm.json"
        try:
            return self._download_file(json_name)
        except RuntimeError:
            print(f"There is no JSON file for item {self.index}")
            return None


    def getInfo(self):
        try:
            return self._download_file("info.json")
        except RuntimeError:
            print(f"There is no Info file for dataset (this is weird)")
            return None

    def getImageSimple(self):
        img_name = self.file_list[self.index]
        return self._download_file(img_name)


    def getImage(self):
        img_name = self.file_list[self.index]
        img_path = self._download_file(img_name)

        # Prefetch in background
        if self._prefetch_thread is None or not self._prefetch_thread.is_alive():
            self._prefetch_thread = threading.Thread(target=self._prefetch_next, daemon=True)
            self._prefetch_thread.start()

        return img_path

    def saveJSON(self):
        print("Updated JSON (Doing nothing)..")

if __name__ == '__main__':
    print("HTTP Folder Stream tester..")
    test = HTTPFolderStreamer("http://ammar.gr/magician/datasets/40-positive-class-a/")
    print("Max files:", test.max())

    img_path = test.getImage()
    print("Downloaded image to:", img_path)

    json_path = test.getJSON()
    print("Downloaded JSON to:", json_path)

