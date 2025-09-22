# Magician Grabber Annotator Tool

The **Magician Annotator Tool** is a graphical application for interactive annotation of industrial datasets.  
It allows manual marking of defects, severity classification, lighting estimation, and region selection.  
The tool supports batch processing, magnification, JSON-based metadata storage, and optional server upload.

The datasets this tool can view, edit and annotate come from the [Magician Grabber](https://github.com/magician-project/magician_grabber)


---

<img src="https://github.com/magician-project/magician_grabber_annotator/blob/main/doc/illustration.jpg?raw=true" />

---

## ✨ Features

- Image browsing from local folders or network datasets  
- Interactive defect point & region annotation  
- Defect classification and severity labeling  
- Auto-generated JSON metadata per image  
- Magnifier tool for precise inspection  
- Brightness & contrast adjustment controls  
- Batch dataset processing and export  
- Upload of annotations to a central server  

---

## 🛠 Installation

Tested on **Ubuntu 22.04.5** with **Python 3.10**.

1. Clone this repository:
   ```bash
   git clone https://github.com/YourOrg/MagicianAnnotator.git
   cd MagicianAnnotator
   ```

2. Create and activate a Python virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   python3 -m pip install wxPython opencv-python numpy
   ```

---

## ▶️ Usage

Run the annotator on a dataset folder:
```bash
python3 wxAnnotator.py --from /path/to/dataset/
```

### Supported inputs
- **Single image file** (`.jpg`, `.jpeg`, `.png`, `.pnm`)  
- **Dataset directory** containing images  
- **Remote datasets** (via the integrated network selector)

---

## 📂 Output

Each annotated image generates a corresponding `.json` file with metadata including:
- Image dimensions  
- MD5 hash  
- Focus measure (Tenengrad)  
- Annotated points (with class & severity)  
- Annotated regions  
- Lighting direction  

Foreground masks can also be saved if regions are selected.

---

## 🔧 GUI Overview

- **Left-click**: Add defect point  
- **Right-click**: Add region of interest  
- **Middle-click / Mouse Wheel**: Navigate through dataset  
- **Checkboxes**: Toggle classifier, auto-advance, or lighting guess  
- **Magnifier Tool**: Zoom and inspect with optional crosshair overlay  
- **Brightness / Contrast controls**: Fine-tune visualization  

---

## ☁️ Uploading Annotations

Annotations can be uploaded to the project server directly from the GUI.  
When selecting **File → Upload Annotations**, provide your credentials.  

For access requests, contact:  
📧 ammarkov@ics.forth.gr

---

## 📜 License

© 2025 Foundation of Research and Technology – Computer Science Department, Greece  
Author: Ammar Qammaz  
