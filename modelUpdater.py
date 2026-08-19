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

    Parses standard Apache / nginx autoindex HTML for .pth and .json entries.
    """
    if not base_url.endswith('/'):
        base_url += '/'
    try:
        with urllib.request.urlopen(base_url, timeout=15) as resp:
            html = resp.read().decode('utf-8', errors='replace')
    except Exception as e:
        return None, str(e)

    files = {}

    # Step 1 — collect every href that ends in .pth or .json (no sub-paths)
    for m in re.finditer(r'href="([^"/?][^"]*\.(pth|json))"', html, re.IGNORECASE):
        fname = m.group(1)
        if '/' not in fname:
            files[fname] = None

    # Step 2 — try to parse sizes from the Apache listing text around each link.
    # Common Apache format:
    #   <a href="name.pth">name.pth</a>    2025-01-15 14:23   45M
    for fname in list(files.keys()):
        esc = re.escape(fname)
        m = re.search(
            esc + r'[^<]*</a>[^<\n]*'
            r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})\s+([\d.]+[KMGkmg]?|-)',
            html, re.IGNORECASE,
        )
        if m:
            files[fname] = _parse_size_str(m.group(2))

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
    Compare the remote repository with the local classifier directory, including
    runs filed under experiments/<campaign>/<run>/ (see find_local_model).

    Returns ``(list[model_info_dict], error_str_or_None)``.

    Each dict contains:
        name              – model base name (no extension)
        pth_remote_size   – remote .pth size in bytes or None
        json_remote_size  – remote .json size in bytes or None
        has_pth_local     – bool
        has_json_local    – bool
        local_pth         – path of the local .pth, or None
        local_pth_size    – int or None
        status            – 'new' | 'updated' | 'corrupted' | 'filed' | 'current'

    'filed' means the only local copy was trained here and sits in a run directory
    rather than in the deployment directory. It is present, so it is NOT 'new', and
    its size is not a staleness signal (it never came from the repository), so it is
    never 'updated' either -- downloading it would only add a flat copy that shadows
    the local run.
    """
    remote_files, err = fetch_repository_index(base_url)
    if remote_files is None:
        return [], err

    pth_names  = {os.path.splitext(f)[0] for f in remote_files if f.endswith('.pth')}
    json_names = {os.path.splitext(f)[0] for f in remote_files if f.endswith('.json')}
    model_names = sorted(pth_names & json_names)

    results = []
    for name in model_names:
        pth_remote  = remote_files.get(f"{name}.pth")
        json_remote = remote_files.get(f"{name}.json")
        found       = find_local_model(local_dir, name)
        local_pth   = found[0] if found else None
        local_size  = os.path.getsize(local_pth) if found else None

        if found is None:
            status = 'new'
        elif not _is_valid_pth(local_pth):
            status = 'corrupted'
        elif not _is_flat(local_dir, local_pth):
            status = 'filed'
        elif pth_remote is not None and local_size is not None and pth_remote != local_size:
            status = 'updated'
        else:
            status = 'current'

        results.append({
            'name':            name,
            'pth_remote_size': pth_remote,
            'json_remote_size': json_remote,
            'has_pth_local':   found is not None,
            'has_json_local':  found is not None,
            'local_pth':       local_pth,
            'local_pth_size':  local_size,
            'status':          status,
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
    FETCH_STATUSES = ('new', 'updated', 'corrupted')

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
        updated_count = sum(1 for r in results if r['status'] == 'updated')
        filed_count   = sum(1 for r in results if r['status'] == 'filed')

        for i, r in enumerate(results):
            remote_sz = _format_size(r['pth_remote_size'])
            if r['status'] == 'new':
                tag = '[NEW]'
            elif r['status'] == 'corrupted':
                tag = f"[CORRUPTED  local={_format_size(r['local_pth_size'])}]"
            elif r['status'] == 'updated':
                tag = f"[UPDATE  local={_format_size(r['local_pth_size'])}]"
            elif r['status'] == 'filed':
                where = os.path.relpath(os.path.dirname(r['local_pth']), self.local_dir)
                tag = f"[trained here: {where}]"
            else:
                tag = '[current]'
            label = f"{r['name']}   remote: {remote_sz}   {tag}"
            self.list_ctrl.Append(label)
            if r['status'] in self.FETCH_STATUSES:
                self.list_ctrl.Check(i, True)

        summary = f"{new_count} new, {updated_count} updated"
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
        total = len(to_download) * 2   # .pth + .json per model
        done  = 0

        for r in to_download:
            if self._stop:
                break
            for ext in ('pth', 'json'):
                if self._stop:
                    break
                fname = f"{r['name']}.{ext}"
                url   = self.base_url + fname
                dst   = os.path.join(self.local_dir, fname)
                self._ui(self.status_lbl.SetLabel, f"Downloading {fname}…")
                try:
                    urllib.request.urlretrieve(url, dst)
                except Exception as e:
                    self._ui(self.status_lbl.SetLabel, f"Failed {fname}: {e}")
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
