# pip install requests --user

import os
import sys
import json
import requests
import threading

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

class HTTPFolderStreamer:
    def __init__(self, base_url=None, local_dir="http_cache", label="colorFrame_0_"):
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

