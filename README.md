# Magician Grabber Annotator

> **Version 0.44** — Interactive annotation and inspection tool for polarization-camera industrial datasets

The **Magician Grabber Annotator** is a desktop GUI application for annotating, inspecting, and managing industrial defect-detection datasets captured with polarization cameras.
It is the primary human-in-the-loop component of the [MAGICIAN](https://github.com/magician-project) project software stack, bridging raw sensor data and trained neural networks.

Datasets are captured by the [Magician Grabber](https://github.com/magician-project/magician_grabber) and the annotations produced here are used to train real-time defect-detection models with the [Magician Vision Classifier](https://github.com/magician-project/magician_vision_classifier).

---

<img src="https://github.com/magician-project/magician_grabber_annotator/blob/main/doc/illustration.jpg?raw=true" />

---

## Table of Contents

1. [Features](#features)
2. [Software Stack](#software-stack)
3. [Installation](#installation)
4. [Usage](#usage)
5. [GUI Overview](#gui-overview)
6. [Polarization Data Format](#polarization-data-format)
7. [Annotation Output](#annotation-output)
8. [Neural Network Classifier](#neural-network-classifier)
9. [Dataset Creation](#dataset-creation)
10. [Uploading Annotations](#uploading-annotations)
11. [License](#license)

---

## Features

- Browse local folders or stream datasets over the network
- Interactive defect-point and region-of-interest annotation
- Per-point defect class and severity labeling
- Dual-panel view: raw polarization image alongside live neural-network heatmap
- Real-time sliding-window tile classifier with single-model and ensemble modes
- SAM (Segment Anything Model) integration for foreground extraction
- Magnifier tool with optional crosshair overlay for precise inspection
- Brightness, contrast, and polarization-channel visualization controls
- Tenengrad focus-quality metric per frame
- Automatic lighting-direction estimation
- Reinforcement-learning-assisted annotation
- Per-frame JSON metadata with MD5 integrity hash
- Dataset creation pipeline: tiling, class balancing, and RGBA PNG export
- Annotation upload to a central project server
- Model update manager (download new checkpoints from the project repository)

---

## Software Stack

The MAGICIAN toolchain consists of three repositories that share a single Python virtual environment:

| Repository | Purpose |
|---|---|
| **magician_grabber_annotator** ← *this repo* | Dataset inspection, annotation, dataset creation |
| [magician_grabber](https://github.com/magician-project/magician_grabber) | Live data acquisition from the polarization camera |
| [magician_vision_classifier](https://github.com/magician-project/magician_vision_classifier) | CNN training, evaluation, and live inference |

An [all-in-one setup script](https://github.com/magician-project/magician_grabber_annotator/blob/main/scripts/setup.sh) is available to clone and configure all three repositories at once:

```bash
bash <(curl -s https://raw.githubusercontent.com/magician-project/magician_grabber_annotator/main/scripts/setup.sh)
```

---

## Installation

Tested on **Ubuntu 22.04.5** with **Python 3.10**. The dependency set is intentionally small and should be compatible with newer Ubuntu/Python releases.

**1. Clone the repository**

```bash
git clone https://github.com/magician-project/magician_grabber_annotator
cd magician_grabber_annotator
```

**2. Create and activate a virtual environment**

```bash
python3 -m venv venv
source venv/bin/activate
```

**3. Install system dependencies**

```bash
sudo apt install python3-venv python3-pip zip wget curl
```

**4. Install Python dependencies**

```bash
pip install -r requirements.txt
```

Or for a minimal installation without the neural-network stack:

```bash
pip install wxPython opencv-python numpy
```

> The neural-network classifier (PyTorch, torchvision, pytorch-lightning) is only required if you intend to run live inference inside the annotator. The tool is fully functional for annotation without it.

---

## Usage

**Quick start with the provided shell script** (creates the virtual environment if it does not exist):

```bash
./runAnnotator.sh --from /path/to/dataset/
```

**Direct invocation:**

```bash
source venv/bin/activate
python3 wxAnnotator.py --from /path/to/dataset/
```

**Command-line options:**

| Flag | Description |
|---|---|
| `--from <path>` | Open a dataset directory or a single image file on startup |
| `--db <path>` | Set the base path used for dataset storage |
| `--classifier` | Enable the neural-network classifier panel on startup |
| `--debug` | Open the wxPython inspector for UI debugging |

### Supported Inputs

- **Single image file** — `.jpg`, `.jpeg`, `.png`, `.pnm`
- **Local dataset directory** — any folder containing image files
- **Remote dataset** — via the integrated network selector (`File → Open Network`)

For a full walkthrough see the [annotator guide](https://github.com/magician-project/magician_grabber_annotator/blob/main/doc/annotator_guide.md).

---

## GUI Overview

The interface is divided into a main image area (dual-panel) and a tabbed side panel on the right.

### Mouse and Keyboard Controls

| Action | Effect |
|---|---|
| **Left-click** on image | Add a defect annotation point |
| **Right-click** on image | Add a region of interest |
| **Middle-click** | Navigate to the next frame |
| **Mouse wheel** | Step through the dataset |

### Side Panel Tabs

- **Annotator** — defect class, severity, lighting direction, and annotation list
- **Classifier** — neural-network model selection, confidence threshold, tile step size, erosion filter, two-stage ensemble toggle, and the *Disabled Model* switch
- **Sensors** — tactile sensor data plots synchronized with the current frame

### Visualization Controls

- **Processor combo box** — select the polarization-channel rendering mode (`PolarizationRGB1–3`, `AoLP`, `DoLP`, `Intensity`, `Sobel`, individual angles, etc.)
- **Brightness / Contrast sliders** — fine-tune image rendering
- **Magnifier** — zoom into either image panel; toggle the crosshair overlay from the tool menu

---

## Polarization Data Format

Datasets are acquired with the **SONY XCG-CP510 Polarsense** polarization camera. Raw frames are stored as lossless **PNM** (Netpbm) files or as **RGBA PNG** files, both decoded identically by `readPolarPNMToRGBALive`.

After de-bayering, each pixel is split into four polarization angles assigned to separate image channels:

| Channel | Polarization angle |
|---|---|
| R | 0° |
| G | 45° |
| B | 90° |
| A | 135° |

This RGBA representation is the direct input to the tile classifier.

Example files: [`doc/example.pnm`](https://github.com/magician-project/magician_grabber_annotator/blob/main/doc/example.pnm?raw=true) · [`doc/example.png`](https://github.com/magician-project/magician_grabber_annotator/blob/main/doc/example.png?raw=true)

The [`comparePNMPNG.py`](comparePNMPNG.py) utility verifies that both encodings decode to identical polarization data:

```bash
python3 comparePNMPNG.py doc/example.pnm doc/example.png
# Expected output: OK
```

---

## Annotation Output

Saving a frame (`File → Save` or automatic save on navigation) writes a `.json` file alongside the image with the following fields:

```json
{
  "width": 1224,
  "height": 1024,
  "md5hash": "...",
  "tenengradFocusMeasure": 142.5,
  "pointClicks": [[x, y], ...],
  "pointClasses": ["Negative Dent", ...],
  "pointSeverities": ["Class A", ...],
  "regionClicks": [[x, y], ...],
  "lightDirection": "Top"
}
```

If a foreground region is selected, a corresponding `*_foreground.png` mask is also written.

Dataset annotations can be packaged and exported via `File → Export Annotations (zip)`.

---

## Neural Network Classifier

When the `magician_vision_classifier` repository is present alongside this one, the annotator can run live inference on each frame and display a color-coded tile heatmap in the left panel.

<img src="https://github.com/magician-project/magician_grabber_annotator/blob/main/doc/allclass_smallcnn.jpg?raw=true" height="350"/>

### Classifier Tab Controls

| Control | Description |
|---|---|
| **Disabled Model** | Master on/off switch (active by default — inference is not run until unchecked) |
| **Model** | Select from available checkpoints in the classifier directory |
| **Threshold** | Minimum confidence required to register a tile as a defect |
| **Step size** | Pixel stride between tiles (smaller = denser, slower) |
| **Use majority voting** | Smooth predictions with a 3×3 spatial majority-vote filter |
| **Erode Kernel / Min. Threshold** | Post-processing erosion to suppress isolated false positives |
| **Enable two-stage classification** | Use the full model ensemble for higher accuracy |
| **Two-stage parallelism** | Run ensemble models concurrently (higher VRAM usage) |
| **Ensemble min Hz filter** | Drop ensemble members slower than the specified frame rate |
| **Check for Model Updates** | Download new or updated checkpoints from the project repository |
| **Check Model Statistics** | Display per-class accuracy and per-model throughput statistics |

### Supported Architectures

ResNet18 · ResNeXt50 · ConvNeXt Tiny · EfficientNet V2 · Swin V2 · RegNet · custom small CNNs

<img src="https://github.com/magician-project/magician_grabber_annotator/blob/main/doc/performance.jpg?raw=true"/>

> **Important:** the tile size used during inference must match the tile size used when the model was trained.

---

## Dataset Creation

Annotated frames can be exported as a training-ready tile dataset via `Tools → Create Dataset`.

<img src="https://github.com/magician-project/magician_grabber_annotator/blob/main/doc/dataset_creator.jpg?raw=true"/>

Configure the output directory, tile size, and class thresholds, then click **Start Dump**. The tool generates a `keras_dataset/` directory with one sub-folder per class:

```
keras_dataset/
├── class_clean/
├── class_NegativeDentClassA/
├── class_NegativeDentClassB/
├── class_PositiveDentClassA/
└── class_WeldingClassA/
```

Each tile is an **RGBA PNG** file whose metadata comment encodes the source image path and pixel coordinate, enabling lossless traceability from a high-loss training sample back to the original annotation.

---

## Uploading Annotations

Annotations can be uploaded to the project server directly from the GUI via `File → Upload Annotations`.

For data access requests, please consult the [data access guide](https://github.com/magician-project/magician_grabber_annotator/blob/main/doc/data_access_guide.md).

---

## License

© 2025 Foundation of Research and Technology – Hellas (FORTH), Computer Science Department, Greece  
Author: Ammar Qammaz · [ammar.gr](http://ammar.gr) · ammarkov@ics.forth.gr

See [`license.txt`](license.txt) for the full license terms.
