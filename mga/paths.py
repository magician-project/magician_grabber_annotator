"""Repo-root resolution.

Every module that needs a root-anchored path (server.json,
annotationStatusCache.json, the default/ sample dataset, doc/) goes through
repo_root() instead of assuming the current working directory -- so a future
layout move cannot break a dozen files at once. classifier_root() is the one
place that knows where the sibling magician_vision_classifier repo lives.
"""

import os


def repo_root():
    """Absolute path of the magician_grabber_annotator root (two levels above this file)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def classifier_root():
    """Absolute path of the sibling magician_vision_classifier repo (checked out next to this one)."""
    return os.path.abspath(os.path.join(repo_root(), os.pardir, "magician_vision_classifier"))
