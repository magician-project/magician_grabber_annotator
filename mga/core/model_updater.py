#!/usr/bin/python3
"""
modelUpdater.py - Check and download classifier models from the online repository.
"""

import glob
import os
import re
import threading
import urllib.request
import urllib.error
import wx


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_size_str(s):
    """Parse Apache-style size strings ('45M', '1.2G', '512K', '1234') to bytes."""
    s = s.strip()
    if not s or s == '-':
        return None
    mult = {'K': 1024, 'M': 1024 ** 2, 'G': 1024 ** 3, 'T': 1024 ** 4}
    if s[-1].upper() in mult:
        try:
            return int(float(s[:-1]) * mult[s[-1].upper()])
        except ValueError:
            return None
    try:
        return int(s)
    except ValueError:
        return None


def _format_size(n):
    """Format byte count to human-readable string."""
    if n is None:
        return "?"
    for unit, threshold in [('GB', 1 << 30), ('MB', 1 << 20), ('KB', 1 << 10)]:
        if n >= threshold:
            return f"{n / threshold:.1f} {unit}"
    return f"{n} B"


# ---------------------------------------------------------------------------
# Repository helpers
# ---------------------------------------------------------------------------

def fetch_repository_index(base_url):
    """
    Fetch the directory listing at base_url and return
    ``(dict[filename -> size_bytes_or_None], error_str_or_None)``.

    Parses standard Apache / nginx autoindex HTML for .zip entries -- the
    repository packs each model as one flat {model}_{timestamp}.zip (see
    mvc.inference.model_download), not bare .pth/.json files.
    """
    if not base_url.endswith('/'):
        base_url += '/'
    try:
        with urllib.request.urlopen(base_url, timeout=15) as resp:
            html = resp.read().decode('utf-8', errors='replace')
    except Exception as e:
        return None, str(e)

    files = {}

    # Step 1 — collect every href that ends in .zip (no sub-paths)
    for m in re.finditer(r'href="([^"/?][^"]*\.zip)"', html, re.IGNORECASE):
        fname = m.group(1)
        if '/' not in fname:
            files[fname] = None

    # Step 2 — parse sizes from the text around each link. Tolerant of both a
    # flat Apache listing (link then plain text: "...</a>  2025-01-15 14:23  45M")
    # and a table-based nginx autoindex (link then "</a></td><td>...date...</td>
    # <td>...size...</td>") by stripping tags from a window after the link
    # before matching the date + size.
    for fname in list(files.keys()):
        m = re.search(re.escape(fname) + r'[^<]*</a>', html, re.IGNORECASE)
        if not m:
            continue
        window = re.sub(r'<[^>]+>', ' ', html[m.end():m.end() + 300])
        sm = re.search(
            r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})\s+([\d.]+[KMGkmg]?|-)',
            window, re.IGNORECASE,
        )
        if sm:
            files[fname] = _parse_size_str(sm.group(2))

    return files, None


def find_local_model(local_dir, name):
    """Where `name`'s .pth/.json actually live locally: ``(pth, json)`` or None.

    The classifier repo has two layouts. A DEPLOYED box keeps models flat in
    local_dir; a TRAINING box files each finished run under
    experiments/<campaign>/<run>/ beside its config and plots. Flat wins on a tie,
    matching ClassifierPnm.model_scan()/model_locate() -- which is the canonical
    implementation, deliberately not imported here so this module stays usable
    without torch and without wxAnnotator (which imports it).
    """
    flat = (os.path.join(local_dir, f"{name}.pth"),
            os.path.join(local_dir, f"{name}.json"))
    if os.path.isfile(flat[0]) and os.path.isfile(flat[1]):
        return flat
    for pth in sorted(glob.glob(os.path.join(local_dir, 'experiments',
                                             '*', '*', f'{name}.pth'))):
        cfg = os.path.join(os.path.dirname(pth), f"{name}.json")
        if os.path.isfile(cfg):
            return (pth, cfg)
    return None


def _is_flat(local_dir, path):
    """True if `path` sits directly in the deployment directory, not in a run dir."""
    return os.path.dirname(os.path.abspath(path)) == os.path.abspath(local_dir)


def _is_valid_pth(path):
    """Quick structural check: is the file a readable zip or legacy pickle?"""
    try:
        import zipfile
        if zipfile.is_zipfile(path):
            return True
        with open(path, 'rb') as f:
            header = f.read(2)
        return header == b'\x80\x02'
    except Exception:
        return False


def check_for_updates(base_url, local_dir):
    """
    Compare the remote model repository with the local classifier directory,
    including runs filed under experiments/<campaign>/<run>/ (see find_local_model).

    The repository packs each model as one {name}_{timestamp}.zip (pth+json+
    plots), so picking "which zip is current for this model" delegates to
    mvc.inference.model_download.newest_zip_for -- the same helper the app's
    own auto-download path (ensure_model) uses -- instead of re-deriving it here.

    Returns ``(list[model_info_dict], error_str_or_None)``.

    Each dict contains:
        name           – model base name (no extension)
        remote_zip     – newest {name}_{timestamp}.zip filename on the server
        remote_size    – that zip's size in bytes, or None if unknown
        local_pth      – path of the local .pth, or None
        local_pth_size – int or None
        status         – 'new' | 'corrupted' | 'filed' | 'current'

    'filed' means the only local copy was trained here and sits in a run directory
    rather than in the deployment directory. It is present, so it is NOT 'new'.

    There is no 'updated' status: the remote size is a compressed zip bundling
    pth+json+plots, which has no byte-for-byte relationship to the locally
    extracted .pth, so it can't signal staleness. ensure_model() (which the
    rest of the app's auto-download path uses) makes the same call --
    presence-only, no staleness check.
    """
    remote_files, err = fetch_repository_index(base_url)
    if remote_files is None:
        return [], err

    from mvc.inference.model_download import newest_zip_for
    remote_zips = list(remote_files.keys())

    model_names = set()
    for z in remote_zips:
        m = re.match(r"(.+)_\d{8}_\d{6}\.zip$", z)
        if m:
            model_names.add(m.group(1))

    results = []
    for name in sorted(model_names):
        remote_zip  = newest_zip_for(name, remote_zips)
        remote_size = remote_files.get(remote_zip)
        found       = find_local_model(local_dir, name)
        local_pth   = found[0] if found else None
        local_size  = os.path.getsize(local_pth) if found else None

        if found is None:
            status = 'new'
        elif not _is_valid_pth(local_pth):
            status = 'corrupted'
        elif not _is_flat(local_dir, local_pth):
            status = 'filed'
        else:
            status = 'current'

        results.append({
            'name':           name,
            'remote_zip':     remote_zip,
            'remote_size':    remote_size,
            'local_pth':      local_pth,
            'local_pth_size': local_size,
            'status':         status,
        })

    return results, None


# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------

class ModelUpdaterDialog(wx.Dialog):
    """
    Checks the online repository for new or updated classifier models and
    lets the user selectively download them (with .json configs) to the
    local classifier directory.

    Downloads always land FLAT in local_dir: that is the deployment layout, and it is
    the copy ClassifierPnm.model_locate() prefers. A model that only exists in a
    run directory is reported as 'filed' and left unchecked -- it is already usable,
    and fetching it would just shadow the local run with the server's copy.
    """

    # Statuses "Select New / Updated" ticks, i.e. the ones a download actually fixes.
    FETCH_STATUSES = ('new', 'corrupted')

    def __init__(self, parent, base_url, local_dir):
        super().__init__(parent, title="Check for Model Updates", size=(660, 500))
        self.base_url  = base_url if base_url.endswith('/') else base_url + '/'
        self.local_dir = local_dir
        self._stop             = False
        self._modal_closed     = False
        self._model_data       = []
        self._post_check_hook  = None   # optional callable(results, err) set by caller

        # ------------------------------------------------------------------ UI
        vbox = wx.BoxSizer(wx.VERTICAL)

        self.status_lbl = wx.StaticText(self, label="Contacting repository, please wait…")
        vbox.Add(self.status_lbl, 0, wx.ALL | wx.EXPAND, 8)

        self.list_ctrl = wx.CheckListBox(self, style=wx.LB_SINGLE)
        vbox.Add(self.list_ctrl, 1, wx.LEFT | wx.RIGHT | wx.BOTTOM | wx.EXPAND, 8)

        self.gauge = wx.Gauge(self, range=100)
        vbox.Add(self.gauge, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM | wx.EXPAND, 8)

        # Buttons
        btn_row = wx.BoxSizer(wx.HORIZONTAL)
        self.select_btn  = wx.Button(self, label="Select New / Updated")
        self.download_btn = wx.Button(self, label="Download Selected")
        self.close_btn   = wx.Button(self, wx.ID_CANCEL, label="Close")
        self.select_btn.Disable()
        self.download_btn.Disable()
        btn_row.Add(self.select_btn,   0, wx.RIGHT, 8)
        btn_row.Add(self.download_btn, 0, wx.RIGHT, 8)
        btn_row.AddStretchSpacer()
        btn_row.Add(self.close_btn, 0)
        vbox.Add(btn_row, 0, wx.ALL | wx.EXPAND, 8)

        self.SetSizer(vbox)

        self.select_btn.Bind(wx.EVT_BUTTON, self._on_select_new_updated)
        self.download_btn.Bind(wx.EVT_BUTTON, self._on_download)
        self.Bind(wx.EVT_BUTTON, self._on_close, id=wx.ID_CANCEL)
        self.Bind(wx.EVT_CLOSE,  self._on_window_close)

        # Kick off background check immediately
        threading.Thread(target=self._check_thread, daemon=True).start()

    # ------------------------------------------------------------------ thread helpers

    def _ui(self, fn, *args, **kwargs):
        if not self._modal_closed:
            try:
                wx.CallAfter(fn, *args, **kwargs)
            except Exception:
                pass

    # ------------------------------------------------------------------ check

    def _check_thread(self):
        results, err = check_for_updates(self.base_url, self.local_dir)
        self._ui(self._on_check_done, results, err)

    def _on_check_done(self, results, err):
        if err:
            self.status_lbl.SetLabel(f"Error reaching repository: {err}")
            return

        self._model_data = results
        self.list_ctrl.Clear()

        new_count     = sum(1 for r in results if r['status'] == 'new')
        filed_count   = sum(1 for r in results if r['status'] == 'filed')

        for i, r in enumerate(results):
            remote_sz = _format_size(r['remote_size'])
            if r['status'] == 'new':
                tag = '[NEW]'
            elif r['status'] == 'corrupted':
                tag = f"[CORRUPTED  local={_format_size(r['local_pth_size'])}]"
            elif r['status'] == 'filed':
                where = os.path.relpath(os.path.dirname(r['local_pth']), self.local_dir)
                tag = f"[trained here: {where}]"
            else:
                tag = '[current]'
            label = f"{r['name']}   remote: {remote_sz}   {tag}"
            self.list_ctrl.Append(label)
            if r['status'] in self.FETCH_STATUSES:
                self.list_ctrl.Check(i, True)

        summary = f"{new_count} new"
        if filed_count:
            summary += f", {filed_count} trained here"
        self.status_lbl.SetLabel(
            f"Repository: {len(results)} models found — {summary}."
        )
        self.select_btn.Enable()
        self.download_btn.Enable()

        if self._post_check_hook:
            self._post_check_hook(results, err)

    # ------------------------------------------------------------------ download

    def _download_thread(self, to_download):
        from mvc.inference.model_download import download_model
        total = len(to_download)   # one zip archive (pth+json+plots) per model
        done  = 0

        for r in to_download:
            if self._stop:
                break
            self._ui(self.status_lbl.SetLabel, f"Downloading {r['remote_zip'] or r['name']}…")
            try:
                download_model(r['name'], self.local_dir, include_plots=True, base_url=self.base_url)
            except Exception as e:
                self._ui(self.status_lbl.SetLabel, f"Failed {r['name']}: {e}")
            done += 1
            self._ui(self.gauge.SetValue, int(done / total * 100))

        if not self._stop:
            self._ui(self._on_download_done)

    def _on_download_done(self):
        self.status_lbl.SetLabel("Download complete. Refreshing list…")
        self.gauge.SetValue(100)
        self.download_btn.Disable()
        self.select_btn.Disable()
        # Re-check so statuses update
        threading.Thread(target=self._check_thread, daemon=True).start()

    # ------------------------------------------------------------------ events

    def _on_select_new_updated(self, evt):
        for i, r in enumerate(self._model_data):
            self.list_ctrl.Check(i, r['status'] in self.FETCH_STATUSES)

    def _on_download(self, evt):
        to_download = [
            r for i, r in enumerate(self._model_data)
            if i < self.list_ctrl.GetCount() and self.list_ctrl.IsChecked(i)
        ]
        if not to_download:
            wx.MessageBox("No models selected.", "Info", wx.OK | wx.ICON_INFORMATION)
            return
        self.download_btn.Disable()
        self.select_btn.Disable()
        self._stop = False
        threading.Thread(
            target=self._download_thread, args=(to_download,), daemon=True
        ).start()

    def _on_close(self, evt):
        self._stop = True
        self._end_modal_safe(wx.ID_CANCEL)

    def _on_window_close(self, evt):
        self._stop = True
        self._end_modal_safe(wx.ID_CANCEL)

    def _end_modal_safe(self, code):
        if self._modal_closed:
            return
        self._modal_closed = True
        try:
            if self.IsModal():
                self.EndModal(code)
            else:
                self.Destroy()
        except Exception:
            pass
