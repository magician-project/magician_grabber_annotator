#!/usr/bin/python3

"""
Author : "Ammar Qammaz"
Copyright : "2025 Foundation of Research and Technology, Computer Science Department Greece"
License : "FORTH"

Frame annotation state: the per-frame JSON schema and the parallel-list
bookkeeping it is built on.

Extracted from mga/wx_annotator.py (Stage 3a of its refactor): the schema had
drifted across ~12 handlers — a duplicated md5hash restore, three near-identical
empty-frame templates, four copy-pasted radius guards and eight open(...,"w")+
json.dump blocks. Every read or write of a frame annotation JSON now goes
through this module (schema keys documented in doc/annotator_guide.md).

The four point* lists (pointClicks/pointClasses/pointSeverities/pointSources)
are strictly parallel — same length, same order — and the helpers here enforce
that invariant in both directions. No wx / UI imports.
"""

import json


def empty_annotation(width, height, md5hash="", points=None, classes=None,
                     severities=None, sources=None):
    """The canonical empty per-frame annotation dict; callers with content pass
    the four parallel lists (e.g. Full Auto's per-frame writes)."""
    return {
        "width": width,
        "height": height,
        "md5hash": md5hash,
        "regionClicks": [],
        "pointClicks": list(points) if points is not None else [],
        "pointClasses": list(classes) if classes is not None else [],
        "pointSeverities": list(severities) if severities is not None else [],
        "pointSources": list(sources) if sources is not None else [],
    }


def align_sources(points, sources, trim=True):
    """pointSources aligned to len(points): legacy short lists are padded with
    'manual', overlong ones trimmed when trim is set (restoreFromJSON trims;
    onSave only pads)."""
    n = len(points)
    srcs = list(sources)
    if len(srcs) < n:
        srcs += ["manual"] * (n - len(srcs))
    elif trim and len(srcs) > n:
        srcs = srcs[:n]
    return srcs


def normalize_parallel(points, classes, severities, sources,
                       default_class, default_severity, default_source="manual"):
    """Trim/pad classes, severities and sources to len(points); missing entries
    get the defaults. The single definition of the parallel-array invariant
    (previously re-asserted ad hoc in onCopyPreviousPoints)."""
    n = len(points)
    classes = list(classes)
    severities = list(severities)
    sources = list(sources)
    if len(classes) < n:
        classes += [default_class] * (n - len(classes))
    if len(severities) < n:
        severities += [default_severity] * (n - len(severities))
    if len(sources) < n:
        sources += [default_source] * (n - len(sources))
    return list(points), classes[:n], severities[:n], sources[:n]


def annotation_to_dict(points, classes, severities, sources, regions,
                       width, height, md5hash,
                       tenengrad=0.0, light_direction=None, tracking=None):
    """Serialize frame state to the on-disk schema. tenengradFocusMeasure is
    written only when nonzero, lightDirection only when not 'Unknown' (the
    combo default), tracking only when truthy — exactly what onSave wrote."""
    data = {
        "width": width,
        "height": height,
        "md5hash": md5hash,
        "regionClicks": [(x, y) for x, y in regions],
        "pointClicks": [(x, y) for x, y in points],
        "pointClasses": list(classes),
        "pointSeverities": list(severities),
        "pointSources": align_sources(points, sources, trim=False),
    }
    if tenengrad != 0.0:
        data["tenengradFocusMeasure"] = tenengrad
    if light_direction != "Unknown":
        data["lightDirection"] = light_direction
    if tracking:
        data["tracking"] = tracking
    return data


def annotation_from_dict(data):
    """Read the frame schema out of a JSON dict, defaulting absent keys. (The
    GUI's restoreFromJSON keeps its own 'leave the field untouched when the key
    is absent' assignment, but the per-key read pattern is this.)"""
    points = list(data.get("pointClicks", []))
    return {
        "points": points,
        "classes": list(data.get("pointClasses", [])),
        "severities": list(data.get("pointSeverities", [])),
        "sources": align_sources(points, data.get("pointSources", []), trim=True),
        "regions": list(data.get("regionClicks", [])),
        "tracking": normalize_tracking(data),
        "lightDirection": data.get("lightDirection", "Unknown"),
        "tenengradFocusMeasure": data.get("tenengradFocusMeasure", 0.0),
        "width": data.get("width"),
        "height": data.get("height"),
        "md5hash": data.get("md5hash"),
    }


def normalize_tracking(data):
    """The frame's tracking records: a bare dict (early format) is wrapped as
    [dict]; missing -> None; otherwise the list (restoreFromJSON semantics).
    Batch passes that need a list use `normalize_tracking(data) or []`."""
    tr = data.get("tracking", None)
    return [tr] if isinstance(tr, dict) else tr


def add_point(points, classes, severities, sources, x, y,
              class_val, severity_val, source_val):
    """Append one point to the four parallel lists."""
    points.append((x, y))
    classes.append(class_val)
    severities.append(severity_val)
    sources.append(source_val)


def remove_point(points, classes, severities, sources, idx):
    """Delete index idx from the four parallel lists (sources is len-guarded
    for legacy short lists)."""
    del points[idx]
    del classes[idx]
    del severities[idx]
    if idx < len(sources):
        del sources[idx]


def nearest_point_sq(x, y, points):
    """(index, squared distance) of the point nearest to (x, y), else (-1, inf)."""
    best_i, best_d = -1, float("inf")
    for i, (px, py) in enumerate(points):
        d = (px - x) ** 2 + (py - y) ** 2
        if d < best_d:
            best_i, best_d = i, d
    return best_i, best_d


def is_near_any(x, y, points, radius):
    """True when (x, y) is within `radius` of any point (the min-dist guard)."""
    return any((x - px) ** 2 + (y - py) ** 2 < radius ** 2 for px, py in points)


def write_annotation_json(path, data, indent=None, tag=None):
    """json.dump(data, path, sort_keys=False); the single JSON-write definition
    (previously duplicated across ~8 handlers). Returns True on success; on
    failure prints '<tag> write failed' (tag prefixes the call site's usual
    message) and returns False."""
    try:
        with open(path, "w") as f:
            json.dump(data, f, sort_keys=False, indent=indent)
        return True
    except Exception as e:
        print(("%s write failed" % tag) if tag else "write failed", path, ":", e)
        return False
