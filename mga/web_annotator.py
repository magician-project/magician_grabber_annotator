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

from mga import wx_annotator as WA
# Classifier glue comes from the hub, not the GUI module (Stage 3b of the
# wx_annotator refactor); WA stays for the PhotoCtrl app class/instance only.
from mga.core.classifier_hub import (ClassifierPnm, locate_model, web_model_scan,
                                     ensure_model_downloaded, GATE_DEFECT_MASS,
                                     GATE_MAX_PROB, GATE_OFF,
                                     classifier_relative_directory)


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
# Model index (same source as wxAnnotator: classifier_hub.web_model_scan, local + online)
# ---------------------------------------------------------------------------
def model_files_dir(name):
    """Directory that actually holds this model's .pth/.json and the report artifacts the
    training run dropped beside them: the flat models directory on a deployed box, or
    experiments/<campaign>/<run>/ on a training box (see classifier_hub.locate_model)."""
    found = locate_model(STATE.model_dir, name)
    return os.path.dirname(found[0]) if found else STATE.model_dir


def scan_models():
    local = set(ClassifierPnm.model_scan(STATE.model_dir))
    names = web_model_scan(STATE.model_dir)
    out = []
    for n in names:
        cfg = {}
        try:
            with open(os.path.join(model_files_dir(n), "%s.json" % n)) as fh:
                cfg = json.load(fh)
        except Exception:
            pass
        out.append({
            "name":      n,
            "local":     n in local,
            "backbone":  cfg.get("model", "?"),
            "tile_size": cfg.get("hparams", {}).get("tile_size", "?"),
            "classes":   len(cfg.get("classes", []) or []),
            "loss":      cfg.get("loss", ""),
            "pfc":       cfg.get("penalize_false_clean", ""),
        })
    return out


# Report artifacts the classifier's training run drops next to <model>.pth/.json --
# wherever that is, see model_files_dir(). 159 of the 172 loadable models carry
# confusion matrices, 141 also a threshold sweep.
MODEL_ASSETS = [
    ("conf_row",    "_confusion_row_normalized.png",        "Confusion &mdash; row normalized (per-class recall)"),
    ("conf_hybrid", "_confusion_hybrid_row_normalized.png", "Confusion &mdash; hybrid row normalized"),
    ("conf_total",  "_confusion_total_normalized.png",      "Confusion &mdash; total normalized"),
    ("conf_raw",    "_confusion_raw.png",                   "Confusion &mdash; raw counts"),
    ("thr_curve",   "_threshold_curve_curve.png",           "Threshold sweep &mdash; detection vs false alarm"),
]
ASSET_SUFFIX = {k: s for k, s, _ in MODEL_ASSETS}


_NA_CACHE = {}


def na_jpeg(label, w=640, h=480):
    """Placeholder served in place of a report PNG the training run never produced."""
    if label in _NA_CACHE:
        return _NA_CACHE[label]
    img = np.full((h, w, 3), 34, np.uint8)
    cv2.rectangle(img, (8, 8), (w - 9, h - 9), (60, 60, 60), 1)
    cv2.putText(img, "N/A", (w // 2 - 62, h // 2 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (90, 90, 90), 3, cv2.LINE_AA)
    cv2.putText(img, label[:46], (w // 2 - min(len(label), 46) * 5, h // 2 + 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    _NA_CACHE[label] = buf.tobytes() if ok else b""
    return _NA_CACHE[label]


def asset_path(name, kind):
    """Path of one report PNG, or None if it does not exist. `name` must be a known
    model, so the caller cannot walk out of the model directory."""
    suffix = ASSET_SUFFIX.get(kind)
    if suffix is None or name not in ClassifierPnm.model_scan(STATE.model_dir):
        return None
    p = os.path.join(model_files_dir(name), name + suffix)
    return p if os.path.isfile(p) else None


def model_detail(name):
    """Config + threshold-sweep operating points for this model."""
    d   = model_files_dir(name)
    cfg = {}
    try:
        with open(os.path.join(d, "%s.json" % name)) as fh:
            cfg = json.load(fh)
    except Exception:
        pass

    thr = {}
    try:
        with open(os.path.join(d, "%s_threshold_curve.json" % name)) as fh:
            thr = json.load(fh)
    except Exception:
        pass
    return cfg, thr


def cooldown_left():
    return max(0.0, STATE.cooldown - (time.monotonic() - STATE.last_switch))


def change_model(name):
    """Global-cooldown-guarded model switch. Downloads `name` from the online
    repository first if web_model_scan() found it there but not on disk.
    Returns (ok, message)."""
    with STATE.lock:
        left = cooldown_left()
        if left > 0:
            return False, "Cooldown active: %.0f s remaining before another model change." % left

        if not ensure_model_downloaded(STATE.model_dir, name):
            return False, "Could not find '%s' locally or on the online repository." % name

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
def get_settings():
    """Current inference knobs, read straight off the annotator's own widgets."""
    def work():
        return {
            "step":      WA.app.classifierTileSize.GetValue(),
            "threshold": WA.app.classifierThreshold.GetValue(),
            "gate":      WA.app.classifierGateMode.GetValue(),
            "best":      WA.app.classifierBestDefectClass.GetValue(),
        }
    return wx_call(work)


def apply_settings(step=None, threshold=None, gate=None, best=None):
    """Set those same widgets; wxAnnotator reads them on the next forward()
    (classifier.step and _applyGateSettings), so nothing else has to change."""
    with STATE.lock:
        def work():
            if step is not None:
                WA.app.classifierTileSize.SetValue(max(4, min(128, int(step))))
                WA.app.classifierTileSizeValue.SetLabel(str(WA.app.classifierTileSize.GetValue()))
            if threshold is not None:
                WA.app.classifierThreshold.SetValue(max(0, min(100, int(threshold))))
                WA.app.classifierThresholdValue.SetLabel(
                    "%.2f" % (WA.app.classifierThreshold.GetValue() / 100.0))
            if gate:
                WA.app.classifierGateMode.SetValue(gate)
            WA.app.classifierBestDefectClass.SetValue(bool(best))
        wx_call(work)
        STATE.frame_key = None          # cached JPEGs used the old settings


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


def model_option_label(m):
    if m["local"]:
        return "%s &mdash; %s, tile %s" % (esc(m["name"]), esc(m["backbone"]), esc(m["tile_size"]))
    return "%s &mdash; online, not downloaded" % esc(m["name"])


def model_bar(msg=None, err=False):
    models = scan_models()
    left   = cooldown_left()
    cur    = STATE.model or "(none)"
    opts   = "".join("<option value='%s'%s>%s</option>"
                     % (esc(m["name"]), " selected" if m["name"] == STATE.model else "",
                        model_option_label(m))
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
            "<span class='dim'>active: %s</span>"
            "<a href='/model?m=%s'>reports</a> <a href='/models'>all models</a></div>"
            % (dis, opts, dis, note, esc(cur), quote(cur)))
    return out


def op_row(tag, pt):
    """One operating point from the threshold sweep (detected / false alarm)."""
    if not isinstance(pt, dict):
        return ""
    return ("<tr><td>%s</td><td class='num'>%.3f</td><td class='num'>%.1f%%</td>"
            "<td class='num'>%.1f%%</td></tr>"
            % (tag, pt.get("threshold", 0.0),
               100.0 * pt.get("detected", 0.0), 100.0 * pt.get("false_alarm", 0.0)))


def page_models(msg=None, err=False):
    rows = []
    for m in scan_models():
        n = m["name"]
        rows.append("<tr><td><a href='/model?m=%s'>%s</a></td><td>%s</td>"
                    "<td class='num'>%s</td><td class='num'>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>"
                    % (quote(n), esc(n), esc(m["backbone"]), esc(m["tile_size"]),
                       esc(m["classes"]), esc(m["loss"]), esc(m["pfc"]),
                       "local" if m["local"] else "<span class='dim'>online</span>"))
    return page("Models",
        "<h1><a href='/'>&larr;</a> Models <span class='dim'>%s</span></h1>%s"
        "<table><tr><th>Model</th><th>Backbone</th><th class='num'>Tile</th>"
        "<th class='num'>Classes</th><th>Loss</th><th>pfc</th><th>Source</th></tr>%s</table>"
        % (esc(STATE.model_dir), model_bar(msg, err), "".join(rows)))


def page_model(name, msg=None, err=False):
    cfg, thr = model_detail(name)
    hp = cfg.get("hparams", {})

    facts = [("Backbone", cfg.get("model", "?")), ("Tile size", hp.get("tile_size", "?")),
             ("Classes", len(cfg.get("classes", []) or [])), ("Loss", cfg.get("loss", "")),
             ("Penalize false clean", cfg.get("penalize_false_clean", "")),
             ("Epochs", hp.get("training_epochs", "")),
             ("Training set", cfg.get("training_dataset", "")),
             ("Validation set", cfg.get("validation_dataset", "")),
             ("MD5", cfg.get("model_md5", ""))]
    facts_html = "".join("<tr><th>%s</th><td>%s</td></tr>" % (esc(k), esc(v)) for k, v in facts)

    if thr:
        ops = op_row("best balanced", thr.get("best_balanced")) + \
              op_row("best KPI",      thr.get("best_kpi"))
        sweeps = thr.get("sweeps") or {}
        gates = ("<p class='dim'>Per-gate sweeps in the report: %s</p>"
                 % esc(", ".join(sorted(sweeps)))) if sweeps else ""
        ops_html = ("<h2>Operating points <span class='dim'>%s</span></h2>"
                    "<table><tr><th>Point</th><th class='num'>Threshold</th>"
                    "<th class='num'>Detected</th><th class='num'>False alarm</th></tr>"
                    "%s</table>%s" % (esc(thr.get("title", "")), ops, gates))
    else:
        ops_html = "<h2>Operating points</h2><p class='dim'>No threshold sweep for this model.</p>"

    tiles = "".join(
        "<figure><a href='/asset?m=%s&k=%s'><img src='/asset?m=%s&k=%s'></a>"
        "<figcaption>%s</figcaption></figure>"
        % (quote(name), k, quote(name), k, cap) for k, _, cap in MODEL_ASSETS)

    load = ("<form method='post' action='/model'>"
            "<input type='hidden' name='model' value='%s'>"
            "<input type='hidden' name='back' value='%s'>"
            "<button type='submit'%s>Load this model</button></form>"
            % (esc(name), esc(name), " disabled" if cooldown_left() > 0 else ""))

    online_note = ("" if name in ClassifierPnm.model_scan(STATE.model_dir) else
                   "<p class='dim'>Not downloaded yet &mdash; \"Load this model\" fetches it "
                   "from the online repository first.</p>")

    return page(name,
        "<h1><a href='/models'>&larr;</a> %s%s</h1>%s%s"
        "<div class='bar'>%s</div>"
        "<table style='max-width:900px'>%s</table>%s"
        "<h2>Reports</h2><div class='views'>%s</div>"
        % (esc(name),
           " <span class='dim'>(active)</span>" if name == STATE.model else "",
           online_note, model_bar(msg, err), load, facts_html, ops_html, tiles))


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


def settings_bar(name, idx):
    """Step / threshold / gate, applied to the annotator's widgets and re-rendered."""
    s = get_settings()
    gates = "".join("<option%s>%s</option>"
                    % (" selected" if g == s["gate"] else "", esc(g))
                    for g in (GATE_DEFECT_MASS, GATE_MAX_PROB, GATE_OFF))
    return ("<div class='bar'><form method='post' action='/settings' "
            "style='display:flex;gap:10px;align-items:center;flex-wrap:wrap'>"
            "<input type='hidden' name='d' value='%s'><input type='hidden' name='f' value='%d'>"
            "<b>Step</b> <input type='number' name='step' min='4' max='128' value='%d' style='width:70px'>"
            "<b>Threshold</b> <input type='number' name='threshold' min='0' max='100' value='%d' "
            "style='width:70px'> <span class='dim'>= %.2f</span>"
            "<b>Gate</b> <select name='gate'>%s</select>"
            "<label><input type='checkbox' name='best'%s> best defect class</label>"
            "<button type='submit'>Apply &amp; re-render</button></form></div>"
            % (esc(name), idx, s["step"], s["threshold"], s["threshold"] / 100.0,
               gates, " checked" if s["best"] else ""))


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
        "%s"
        "<div class='bar dim'>%s</div>"
        "<div class='views'>"
        "<figure><img src='/img?d=%s&f=%d&side=left&t=%d'>"
        "<figcaption>LEFT &mdash; classifier visualization</figcaption></figure>"
        "<figure><img src='/img?d=%s&f=%d&side=right&t=%d'>"
        "<figcaption>RIGHT &mdash; %s + ground-truth circles</figcaption></figure></div>"
        % (quote(name), esc(name), idx, meta["ui_max"], esc(meta["file"]), model_bar(),
           " &nbsp; ".join(nav), meta["points"], esc(meta["view"]),
           settings_bar(name, idx), esc(meta["info"]),
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
            elif u.path == "/models":
                self._send(page_models())
            elif u.path == "/model":
                self._send(page_model(unquote(q["m"][0])))
            elif u.path == "/asset":
                name, kind = unquote(q["m"][0]), q.get("k", [""])[0]
                p = asset_path(name, kind)
                if p is None:
                    # no such report for this model -> N/A placeholder (still an image,
                    # so the reports grid keeps its shape)
                    self._send(na_jpeg(kind), "image/jpeg")
                else:
                    with open(p, "rb") as fh:
                        self._send(fh.read(), "image/png")
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
        if u.path not in ("/model", "/settings"):
            self._send(b"", code=404)
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            form = parse_qs(self.rfile.read(n).decode("utf-8"))

            if u.path == "/settings":
                apply_settings(step=form.get("step", [None])[0],
                               threshold=form.get("threshold", [None])[0],
                               gate=form.get("gate", [""])[0],
                               best="best" in form)
                self._send(page_view(unquote(form["d"][0]), int(form.get("f", ["0"])[0])))
                return

            name = form.get("model", [""])[0]
            back = form.get("back", [""])[0]
            ok, msg = change_model(name) if name else (False, "No model given.")
            # "Load this model" on a report page stays on that page
            self._send(page_model(back, msg, err=not ok) if back
                       else page_index(msg, err=not ok))
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
    ap.add_argument("--models", default=classifier_relative_directory,
                    help="directory holding <model>.pth/<model>.json pairs, "
                         "flat and/or under experiments/<campaign>/<run>/")
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
        models = ClassifierPnm.model_scan(STATE.model_dir)
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
