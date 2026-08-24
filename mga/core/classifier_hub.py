#!/usr/bin/python3

"""
Author : "Ammar Qammaz"
Copyright : "2025 Foundation of Research and Technology, Computer Science Department Greece"
License : "FORTH"

The classifier hub: this module owns the ENTIRE cross-repo surface into
magician_vision_classifier — the guarded mvc.inference imports, the gate-name
mirrors used when the classifier is disabled, and the model helpers built on
them (locate_model / web_model_scan / ensure_model_downloaded).

Extracted from mga/wx_annotator.py (Stage 3b of its refactor) so web_annotator
and classifier_tab no longer read classifier glue off the GUI module.
wx_annotator re-exports every name below for ClassifierTabMixin, which resolves
them through sys.modules — keep those re-exports when this list changes.

The checklist to re-check after a refactor over in magician_vision_classifier
(verified against the 2026-08-20 mvc/ package refactor):

  mvc.inference.classifier_pnm   ClassifierPnm, GATE_DEFECT_MASS,
                                 GATE_MAX_PROB, GATE_OFF
                                 load_recommended_configuration,
                                 recommended_configuration_available
  mvc.inference.ensemble_classifier
                                 EnsembleClassifierPnm
  mvc.inference.model_download   remote_model_names     (web_model_scan below)
                                 download_model
                                 ensure_model           (ensure_model_downloaded below)
  mga/stream_dataset.py          mvc.core.shared_memory  SharedMemoryManager (not here)

The classifier core lives in mvc/inference/classifier_pnm.py (ensemble in
mvc/inference/ensemble_classifier.py). The ClassifierPnm/EnsembleClassifierPnm
members used here are forward(), reload_model(), model_scan(), apply_min_hz(),
.step/.tile_size/.hz/.classifiers/._all_classifiers/.model_perf and the three
gate knobs set by _applyGateSettings().

NOTE: `useClassifier` below is the module-level default. `python3 -m
mga.wx_annotator --classifier` rebinds the re-export in wx_annotator's globals
— readers on the GUI side see that binding; never read useClassifier from this
module expecting that mutation.
"""

import sys
import os

from mga.paths import classifier_root

useClassifier   = True #<- Master switch classifier off if you have hw/sw limitations
benchmark       = False #<- Set to True to run a forward-pass timing test on each model at startup
classifier_online_repository = "http://ammar.gr/magician/ckpts2/"
classifier_relative_directory = classifier_root()

#-------------------------------------------------------------------------------
# Make Classifier completely seperatable from the rest of the codebase
#-------------------------------------------------------------------------------
if useClassifier:
  parent_path = classifier_relative_directory
  sys.path.append(parent_path)
  try:
    from mvc.inference.classifier_pnm import ClassifierPnm, GATE_DEFECT_MASS, GATE_MAX_PROB, GATE_OFF
    from mvc.inference.ensemble_classifier import EnsembleClassifierPnm
    # Deployment presets shared with the ROS node (recommended_configuration.json in
    # the classifier repo). Optional: older classifier checkouts will not have it.
    try:
        from mvc.inference.classifier_pnm import (load_recommended_configuration,
                                                  recommended_configuration_available)
    except Exception:
        load_recommended_configuration = None
        def recommended_configuration_available(*args, **kwargs):
            """No presets file support in this classifier checkout."""
            return False
  except Exception as e:
    print("Can't seem to be able to access the magician_vision_classifier, consider setting useClassifier=False in mga/core/classifier_hub.py")
    print("Classifier Path : ",parent_path)
    print("If you want the classifier but don't have it get it @ https://github.com/magician-project/magician_vision_classifier")
    print(f"Exact error was : {e}")
    sys.exit(1)
else:
  # Mirror the classifier's gate names so the GUI builds without the classifier.
  GATE_DEFECT_MASS = "defect_mass"
  GATE_MAX_PROB    = "max_prob"
  GATE_OFF         = "off"
  load_recommended_configuration = None
  def recommended_configuration_available(*args, **kwargs):
      """Presets are a classifier feature; the classifier is disabled here."""
      return False
  class ClassifierPnm:
    def __init__(self, model_path='foo', cfg_path='foo', tile_classes=['foo'],tile_size=64, step=16):
        print("Classifier PNM is disabled, please start with --classifier or change the useClassifier variable in mga/core/classifier_hub.py to use it!")
        pass
    def load_model(self):
        return None
    def model_scan(directoryPath):
        return ['Disabled']
    def reload_model(self, directoryPath, name):
        return False
    def forward(self, image, majorityVote=True):
        return None
#-------------------------------------------------------------------------------

# The classifier repo keeps models in two layouts and we have to read both: a DEPLOYED
# box gets them unpacked flat into magician_vision_classifier/ by mvc.inference.model_download, while a
# TRAINING box files each finished run under experiments/<campaign>/<run>/ next to its
# config and report artifacts. ClassifierPnm.model_scan()/model_locate() already cover
# both; everything here that builds "<dir>/<name>.pth" by hand must go through this so
# the two stay in step. Flat wins on a tie, same as model_scan.
def locate_model(model_dir, name):
    """(pth_path, cfg_path) for model `name`, or None if either file is missing."""
    locate = getattr(ClassifierPnm, "model_locate", None)
    if locate is not None:            # older classifier checkouts are flat-only
        return locate(model_dir, name)
    pth = os.path.join(model_dir, "%s.pth"  % name)
    cfg = os.path.join(model_dir, "%s.json" % name)
    return (pth, cfg) if (os.path.isfile(pth) and os.path.isfile(cfg)) else None


def remote_model_names(timeout=5):
    """Names advertised by the online model repository (mvc.inference.
    model_download), or [] when the repository is unreachable."""
    try:
        from mvc.inference.model_download import remote_model_names as _remote_model_names
        return _remote_model_names(timeout=timeout)
    except Exception as e:
        print(f"[Models] Online repository unavailable: {e}")
        return []


def download_model(name, model_dir, include_plots=True):
    """Download `name` from the online repository into `model_dir`
    (mvc.inference.model_download.download_model)."""
    from mvc.inference.model_download import download_model as _download_model
    return _download_model(name, model_dir, include_plots=include_plots)


def web_model_scan(model_dir):
    """All models available to the web annotator: model_scan()'s local pairs plus
    whatever the online repository (mvc.inference.model_download) advertises, so a
    model that only lives on the server still shows up in the list (see
    web_annotator.change_model, which downloads it on demand when selected)."""
    local = ClassifierPnm.model_scan(model_dir)
    remote = remote_model_names()
    return sorted(set(local) | set(remote))


def ensure_model_downloaded(model_dir, name):
    """Fetch `name` from the online repository if model_scan() doesn't see it in
    `model_dir` yet. Returns True once the model is available locally (already
    there, or freshly downloaded), False if it's on neither disk nor the server."""
    try:
        from mvc.inference.model_download import ensure_model
        return ensure_model(name, model_dir)
    except Exception as e:
        print(f"[Models] Download of '{name}' failed: {e}")
        return False
