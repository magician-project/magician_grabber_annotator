#!/usr/bin/python3

"""
Author : "Ammar Qammaz"
Copyright : "2026 Foundation of Research and Technology, Computer Science Department Greece, See license.txt"
License : "FORTH"

Minimal HTTP front-end for wxAnnotator.py.

It does not reimplement any rendering: it builds a real (hidden) PhotoCtrl wx.App and
drives the very same code paths the GUI uses --

    app.onNewInputPath(dir)   -> load dataset
    app.gotoFrameUI(idx)      -> decode + classify + onView()
    app.imageCtrl             -> LEFT  panel (classifier visualization)
    app.secondaryImageCtrl    -> RIGHT panel (polarization view + ground-truth circles)

-- and re-encodes those two wx bitmaps as JPEG for HTTP transport.

wx is not thread safe, so every touch of the app is marshalled onto the wx main thread
(see wx_call) and serialized with a lock; the HTTP server runs in a background thread.

READ-ONLY: gotoFrameUI() normally calls onSave() and _rememberLastFrame(), which write
colorFrame_*.json / last.frame into the dataset. Browsing from a web page must not
mutate the user's datasets, so both are neutralized at startup (see _make_readonly).

Usage:  python3 webAnnotator.py --db /media/ammar/games2/Datasets/Magician
        (run it under the same venv as runAnnotatorAmmar.sh -- it needs torch + wx)
"""

import os
import sys
import io
import json
import time
import threading
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs, quote, unquote

import cv2
import numpy as np
import wx

import wxAnnotator as WA


# ---------------------------------------------------------------------------
# wx main-thread marshalling
# ---------------------------------------------------------------------------
def wx_call(fn, timeout=600):
    """Run fn() on the wx main thread, block until it finishes, return its result."""
    done = threading.Event()
    box  = {}

    def runner():
        try:
            box["value"] = fn()
        except Exception as e:
            box["error"] = e
        finally:
            done.set()

    wx.CallAfter(runner)
    if not done.wait(timeout):
        raise RuntimeError("wx call timed out after %ss" % timeout)
    if "error" in box:
        raise box["error"]
    return box.get("value")


def bmp_to_jpeg(bmp, quality=85):
    """wx.Bitmap -> JPEG bytes. Must be called on the wx main thread."""
    if bmp is None or not bmp.IsOk():
        return None
    img = bmp.ConvertToImage()
    w, h = img.GetWidth(), img.GetHeight()
    rgb = np.frombuffer(bytes(img.GetData()), dtype=np.uint8).reshape(h, w, 3)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                           [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return buf.tobytes() if ok else None


# ---------------------------------------------------------------------------
# Server state
# ---------------------------------------------------------------------------
class State:
    def __init__(self, base, model_dir, cooldown):
        self.base      = base
        self.model_dir = model_dir
        self.cooldown  = cooldown
        self.lock      = threading.Lock()      # serializes every wx-side operation
        self.datasets  = None                  # cached index, built lazily
        self.dataset   = None                  # currently loaded dataset name
        self.model     = None                  # currently loaded model name
        self.last_switch   = 0.0               # monotonic time of the last model change
        self.frame_cache   = None              # (dataset, idx) -> (left_jpeg, right_jpeg)
        self.frame_key     = None

STATE = None


# ---------------------------------------------------------------------------
# Dataset index
# ---------------------------------------------------------------------------
def scan_datasets(base):
    """One scandir pass per subdirectory: frame count, annotated count, info.json."""
    out = []
    try:
        entries = sorted(os.scandir(base), key=lambda e: e.name.lower())
    except OSError as e:
        print("Cannot scan dataset base %s: %s" % (base, e))
        return out

    for e in entries:
        if not e.is_dir():
            continue
        frames = annotated = 0
        try:
            with os.scandir(e.path) as it:
                for f in it:
                    n = f.name
                    if not n.startswith("colorFrame_0_"):
                        continue
                    if n.endswith(".png") or n.endswith(".pnm"):
                        frames += 1
                    elif n.endswith(".json"):
                        annotated += 1
        except OSError:
            continue
        if frames == 0:
            continue

        info = {}
        try:
            with open(os.path.join(e.path, "info.json")) as fh:
                info = json.load(fh)
        except Exception:
            pass

        out.append({
            "name":       e.name,
            "frames":     frames,
            "annotated":  annotated,
            "certified":  info.get("certified_by", ""),
            "when":       info.get("annotated_at", ""),
            "defects":    info.get("defect_counts", {}),
            "severities": info.get("severity_counts", {}),
            "total":      info.get("total_defects", 0),
        })
    return out


def get_datasets(rescan=False):
    if STATE.datasets is None or rescan:
        STATE.datasets = scan_datasets(STATE.base)
    return STATE.datasets


# ---------------------------------------------------------------------------
# Model index (same source as wxAnnotator: ClassifierPnm.model_scan)
# ---------------------------------------------------------------------------
def scan_models():
    names = WA.ClassifierPnm.model_scan(STATE.model_dir)
    out = []
    for n in names:
        cfg = {}
        try:
            with open(os.path.join(STATE.model_dir, "%s.json" % n)) as fh:
                cfg = json.load(fh)
        except Exception:
            pass
        out.append({
            "name":      n,
            "backbone":  cfg.get("model", "?"),
            "tile_size": cfg.get("hparams", {}).get("tile_size", "?"),
            "classes":   len(cfg.get("classes", []) or []),
            "loss":      cfg.get("loss", ""),
            "pfc":       cfg.get("penalize_false_clean", ""),
        })
    return out


def cooldown_left():
    return max(0.0, STATE.cooldown - (time.monotonic() - STATE.last_switch))


def change_model(name):
    """Global-cooldown-guarded model switch. Returns (ok, message)."""
    with STATE.lock:
        left = cooldown_left()
        if left > 0:
            return False, "Cooldown active: %.0f s remaining before another model change." % left

        def work():
            clf = getattr(WA.app, "ClassifierPnm", None)
            if clf is None:
                return False
            if not clf.reload_model(STATE.model_dir, name):
                return False
            # mirror the GUI: statistics belong to the model that produced them
            WA.app.stats.classifier_name = name.lower()
            WA.app.stats.reset()
            return True

        ok = wx_call(work)
        if ok:
            STATE.model       = name
            STATE.last_switch = time.monotonic()
            STATE.frame_key   = None          # cached JPEGs came from the old model
            return True, "Model changed to '%s'." % name
        return False, "Failed to load '%s' (corrupt or incomplete checkpoint?)." % name


# ---------------------------------------------------------------------------
# Frame rendering -- the wxAnnotator code path, verbatim
# ---------------------------------------------------------------------------
def render_frame(dataset, idx):
    """Return (left_jpeg, right_jpeg, meta) for dataset frame idx (UI index).

    Both panels come from one render: the HTML page asks for two images and the
    classifier pass is by far the expensive part, so the pair is cached."""
    with STATE.lock:
        key = (dataset, idx)
        if STATE.frame_key == key and STATE.frame_cache is not None:
            return STATE.frame_cache

        def work():
            path = os.path.join(STATE.base, dataset)
            if STATE.dataset != dataset:
                WA.app.photoTxt.SetValue(path)     # != "default" enables classification
                WA.app.onNewInputPath(path)
                STATE.dataset = dataset
            WA.app.gotoFrameUI(idx)
            meta = {
                "file":   os.path.basename(WA.app.filepath or ""),
                "ui_max": WA.app._ui_max(),
                "points": len(WA.app.points_of_interest),
                "view":   WA.app.ProcessorComboBox.GetValue(),
                "info":   WA.app.classifierInfo.GetLabel(),
            }
            return (bmp_to_jpeg(WA.app.imageCtrl.GetBitmap()),
                    bmp_to_jpeg(WA.app.secondaryImageCtrl.GetBitmap()),
                    meta)

        result = wx_call(work)
        STATE.frame_key   = key
        STATE.frame_cache = result
        return result


def frame_list(dataset):
    """Frame basenames for the dataset, in UI-index order."""
    with STATE.lock:
        def work():
            path = os.path.join(STATE.base, dataset)
            if STATE.dataset != dataset:
                WA.app.photoTxt.SetValue(path)
                WA.app.onNewInputPath(path)
                STATE.dataset = dataset
            names = []
            for ui in range(WA.app._ui_max() + 1):
                s = WA.app._stream_from_ui(ui)
                dl = WA.app.folderStreamer.directoryList
                names.append(os.path.basename(dl[s]) if 0 <= s < len(dl) else "?")
            return names
        return wx_call(work)


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------
CSS = """
body{font:13px/1.45 system-ui,sans-serif;margin:0;padding:16px;background:#161819;color:#ddd}
a{color:#6cf;text-decoration:none} a:hover{text-decoration:underline}
h1{font-size:17px;margin:0 0 12px} h2{font-size:14px;margin:18px 0 8px;color:#aaa}
table{border-collapse:collapse;width:100%;margin-bottom:16px}
th,td{padding:4px 8px;text-align:left;border-bottom:1px solid #2c2f31;white-space:nowrap}
th{color:#888;font-weight:600} tr:hover td{background:#1e2123}
.num{text-align:right;font-variant-numeric:tabular-nums}
.bar{background:#1e2123;padding:8px 12px;margin-bottom:14px;border:1px solid #2c2f31;
     display:flex;gap:12px;align-items:center;flex-wrap:wrap}
select,button,input{background:#26292b;color:#ddd;border:1px solid #3a3e40;padding:4px 8px;font:inherit}
button{cursor:pointer} button:disabled{opacity:.45;cursor:not-allowed}
.views{display:flex;gap:10px;flex-wrap:wrap} .views figure{margin:0;flex:1 1 480px}
.views img{width:100%;border:1px solid #2c2f31;background:#000}
figcaption{color:#888;padding:4px 0}
.msg{padding:8px 12px;margin-bottom:12px;border-left:3px solid #6cf;background:#1e2123}
.err{border-left-color:#e66}
.dim{color:#777}
"""


def page(title, body):
    return ("<!doctype html><html><head><meta charset='utf-8'>"
            "<title>%s</title><style>%s</style></head><body>%s</body></html>"
            % (esc(title), CSS, body)).encode("utf-8")


def esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
                  .replace(">", "&gt;").replace('"', "&quot;"))


def model_bar(msg=None, err=False):
    models = scan_models()
    left   = cooldown_left()
    cur    = STATE.model or "(none)"
    opts   = "".join("<option value='%s'%s>%s &mdash; %s, tile %s</option>"
                     % (esc(m["name"]), " selected" if m["name"] == STATE.model else "",
                        esc(m["name"]), esc(m["backbone"]), esc(m["tile_size"]))
                     for m in models)
    dis    = " disabled" if left > 0 else ""
    note   = ("<span class='dim'>cooldown %.0fs</span>" % left) if left > 0 else \
             ("<span class='dim'>ready</span>")
    out = ""
    if msg:
        out += "<div class='msg%s'>%s</div>" % (" err" if err else "", esc(msg))
    out += ("<div class='bar'><form method='post' action='/model' style='display:flex;gap:8px'>"
            "<b>Model</b> <select name='model'%s>%s</select>"
            "<button type='submit'%s>Load</button></form>%s"
            "<span class='dim'>active: %s</span></div>"
            % (dis, opts, dis, note, esc(cur)))
    return out


def page_index(msg=None, err=False):
    ds = get_datasets()
    rows = []
    for d in ds:
        defects = ", ".join("%s&times;%s" % (esc(v), esc(k)) for k, v in d["defects"].items())
        rows.append(
            "<tr><td><a href='/dataset?d=%s'>%s</a></td><td class='num'>%d</td>"
            "<td class='num'>%d</td><td>%s</td><td>%s</td><td>%s</td></tr>"
            % (quote(d["name"]), esc(d["name"]), d["frames"], d["annotated"],
               esc(d["certified"]), esc(d["when"]), defects or "<span class='dim'>-</span>"))
    return page("Magician datasets",
        "<h1>Magician datasets <span class='dim'>%s</span></h1>%s"
        "<div class='bar'><a href='/?rescan=1'>Rescan</a>"
        "<span class='dim'>%d datasets</span></div>"
        "<table><tr><th>Dataset</th><th class='num'>Frames</th><th class='num'>Annotated</th>"
        "<th>Certified by</th><th>Annotated at</th><th>Defects</th></tr>%s</table>"
        % (esc(STATE.base), model_bar(msg, err), len(ds), "".join(rows)))


def page_dataset(name):
    names = frame_list(name)
    cells = []
    for i, n in enumerate(names):
        cells.append("<a href='/view?d=%s&f=%d'>%d</a>" % (quote(name), i, i))
    return page(name,
        "<h1><a href='/'>&larr;</a> %s</h1>%s"
        "<h2>%d frames &mdash; click an index</h2><div style='line-height:2'>%s</div>"
        % (esc(name), model_bar(), len(names), " ".join(cells)))


def page_view(name, idx):
    left, right, meta = render_frame(name, idx)
    nav = []
    if idx > 0:
        nav.append("<a href='/view?d=%s&f=%d'>&larr; prev</a>" % (quote(name), idx - 1))
    if idx < meta["ui_max"]:
        nav.append("<a href='/view?d=%s&f=%d'>next &rarr;</a>" % (quote(name), idx + 1))
    stamp = int(time.time() * 1000)   # defeat browser caching between model switches
    return page("%s #%d" % (name, idx),
        "<h1><a href='/'>&larr;</a> <a href='/dataset?d=%s'>%s</a> "
        "<span class='dim'>frame %d/%d &mdash; %s</span></h1>%s"
        "<div class='bar'>%s<span class='dim'>%d ground-truth points &mdash; view: %s</span></div>"
        "<div class='bar dim'>%s</div>"
        "<div class='views'>"
        "<figure><img src='/img?d=%s&f=%d&side=left&t=%d'>"
        "<figcaption>LEFT &mdash; classifier visualization</figcaption></figure>"
        "<figure><img src='/img?d=%s&f=%d&side=right&t=%d'>"
        "<figcaption>RIGHT &mdash; %s + ground-truth circles</figcaption></figure></div>"
        % (quote(name), esc(name), idx, meta["ui_max"], esc(meta["file"]), model_bar(),
           " &nbsp; ".join(nav), meta["points"], esc(meta["view"]), esc(meta["info"]),
           quote(name), idx, stamp, quote(name), idx, stamp, esc(meta["view"])))


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    server_version = "webAnnotator/1.0"

    def log_message(self, fmt, *args):
        pass   # the wx/classifier console output is the interesting one

    def _send(self, body, ctype="text/html; charset=utf-8", code=200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _fail(self, e):
        self._send(page("error", "<div class='msg err'>%s</div><a href='/'>back</a>"
                                 % esc(e)), code=500)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            if u.path == "/":
                if "rescan" in q:
                    get_datasets(rescan=True)
                self._send(page_index())
            elif u.path == "/dataset":
                self._send(page_dataset(unquote(q["d"][0])))
            elif u.path == "/view":
                self._send(page_view(unquote(q["d"][0]), int(q.get("f", ["0"])[0])))
            elif u.path == "/img":
                left, right, _ = render_frame(unquote(q["d"][0]), int(q.get("f", ["0"])[0]))
                jpg = left if q.get("side", ["left"])[0] == "left" else right
                if jpg is None:
                    self._send(b"", "image/jpeg", code=404)
                else:
                    self._send(jpg, "image/jpeg")
            else:
                self._send(page("404", "<h1>404</h1><a href='/'>back</a>"), code=404)
        except Exception as e:
            import traceback; traceback.print_exc()
            self._fail(e)

    def do_POST(self):
        u = urlparse(self.path)
        if u.path != "/model":
            self._send(b"", code=404)
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            form = parse_qs(self.rfile.read(n).decode("utf-8"))
            name = form.get("model", [""])[0]
            ok, msg = change_model(name) if name else (False, "No model given.")
            self._send(page_index(msg, err=not ok))
        except Exception as e:
            import traceback; traceback.print_exc()
            self._fail(e)


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
def _make_readonly(app):
    """Browsing must not write into the datasets (see module docstring)."""
    app.onSave             = lambda event=None: None
    app._rememberLastFrame = lambda filepath: None


def main():
    global STATE

    ap = argparse.ArgumentParser(description="Minimal HTTP front-end for wxAnnotator.py")
    ap.add_argument("--db", default="/media/ammar/games2/Datasets/Magician",
                    help="dataset base directory")
    ap.add_argument("--models", default=WA.classifier_relative_directory,
                    help="directory holding <model>.pth/<model>.json pairs")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--cooldown", type=float, default=60.0,
                    help="seconds between model changes (global, process-wide)")
    ap.add_argument("--show", action="store_true", help="keep the wx window visible (debug)")
    args = ap.parse_args()

    STATE = State(os.path.abspath(args.db), args.models, args.cooldown)

    print("Building the annotator app (hidden)...")
    app = WA.PhotoCtrl()
    WA.app = app                 # wxAnnotator's methods reference the module-level 'app'
    app.local_base_path = STATE.base
    _make_readonly(app)
    app.classifierDisabledCheckbox.SetValue(False)   # left panel = classifier visualization
    app.photoTxt.SetValue("default")
    app.onNewInputPath("default")
    if not args.show:
        app.frame.Hide()

    clf = getattr(app, "ClassifierPnm", None)
    STATE.model = os.path.splitext(getattr(clf, "name", "") or "")[0] or None
    if STATE.model is None:
        models = WA.ClassifierPnm.model_scan(STATE.model_dir)
        STATE.model = models[0] if models else None

    httpd = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    print("\nwebAnnotator listening on http://localhost:%d" % args.port)
    print("  datasets : %s" % STATE.base)
    print("  models   : %s (active: %s)" % (STATE.model_dir, STATE.model))
    print("  cooldown : %.0f s between model changes\n" % STATE.cooldown)

    app.MainLoop()


if __name__ == "__main__":
    main()
