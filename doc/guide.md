# MAGICIAN – Data Pre-Processing & Annotation Guide  

This document describes the data pre-processing pipeline, annotation tools, dataset formats, and training workflow used in the **MAGICIAN project**.  
It is intended for collaborators working on dataset preparation, model training, and optimization.


---

## 1. Repository Overview

The FORTH software stack can be fetched and installed using an [all-in-one setup script](https://github.com/magician-project/magician_grabber_annotator/blob/main/scripts/setup.sh):

```bash
https://github.com/magician-project/magician_grabber_annotator/blob/main/scripts/setup.sh
```

The script is interactive and allows you to install only the components you need.

### Main Repositories

#### A) Annotator Tool (a.k.a. the repository you are in :) )
**Purpose:** Data inspection, annotation, dataset creation  

Repository:  
https://github.com/magician-project/magician_grabber_annotator

Use this to:
- Stream and inspect datasets
- Annotate defects
- Export training-ready datasets

---

#### B) Vision Classifier (a.k.a. the repository that uses the annotated data)
**Purpose:** Defect classification training & inference  

Repository:  
https://github.com/magician-project/magician_vision_classifier

Use this to:
- Train classification networks
- Test ensemble strategies
- Apply model optimization techniques

> Uses **PyTorch**  
> Shares the same Python virtual environment as the Annotator

---
 

## 2. Polarization Data Format & PNM Files 

The datasets use **PNM (Netpbm)** files:

- Lossless
- Uncompressed
- Binary format

Reference:  
https://en.wikipedia.org/wiki/Netpbm

### Polarization Encoding

Raw `.pnm` frames are:
- Read/Dumped directly from the SONY XCG-CP510 Polarsense camera
- De-bayered
- Converted to **RGBA OpenCV images**

Each channel represents a polarization angle:

| Channel | Polarization |
|-------|--------------|
| R     | 0°           |
| G     | 45°          |
| B     | 90°          |
| A     | 135°         |

An example can be found in this [example.pnm](https://github.com/magician-project/magician_grabber_annotator/blob/main/doc/example.pnm?raw=true) file


Relevant code to use as template to unpack polarized data:
```python
readPolarPNMToRGBA
```

https://github.com/magician-project/magician_grabber_annotator/blob/main/readData.py#L292

---

## 3. Using the Annotation Tool

After activating the classifier virtual environment:

```bash
python3 wxAnnotator.py --db /path/to/dataset/storage
```

<img src="https://github.com/magician-project/magician_grabber_annotator/blob/main/doc/allclass_convnext_tiny.jpg?raw=true" height=200/> 


<img src="https://github.com/magician-project/magician_grabber_annotator/blob/main/doc/allclass_convnext_tiny2.jpg?raw=true" height=200/> 

### Workflow

1. **File → Open Network**
2. **Connect To Server**
3. Select a dataset (e.g. `NDA_1_A_T1`)
4. Browse streamed frames

For lower latency:
- **File → Download All Frames**

---

### Dataset Creation

Use:

```
Tools → Create Dataset
```

<img src="https://github.com/magician-project/magician_grabber_annotator/blob/main/doc/dataset_creator.jpg?raw=true" /> 


- Select output directory
- Configure tiling & thresholds
- Click **Start Dump**

This generates a `keras_dataset/` directory containing:
- Class-specific folders
- Training-ready RGBA PNG tiles

---

## 4. Dataset Structure & Classes

Generated datasets follow this structure:


<img src="https://github.com/magician-project/magician_grabber_annotator/blob/main/doc/datasets.jpg?raw=true" /> 


```
keras_dataset/
 ├── class_NegativeDentClassA/
 ├── class_NegativeDentClassB/
 ├── class_PositiveDentClassA/
 ├── class_WeldingClassA/
 ├── class_clean/
```

### Notes

- Class names are **simple strings**
- Easy to extend with new defect types
- Extra classes like `Unknown` or `Suspicious` can be removed safely
- Deleting folders removes those samples from training

---

## 5. Training the Classifier

Training script:

```bash
python3 trainClassifierTorch.py configs/bigmodel.json
```

Training code reference:  
https://github.com/magician-project/magician_vision_classifier/blob/main/trainClassifierTorch.py#L605

### Supported Models

- ResNet18
- ResNeXt50
- ConvNeXt Tiny
- EfficientNet V2
- Swin V2
- RegNet
- Custom CNNs

> ⚠️ Tile size used for training **must match** tile size used during dataset generation.

---

### Model Strategy


<img src="https://github.com/magician-project/magician_grabber_annotator/blob/main/doc/confusion.png?raw=true" /> 

There is **no single “correct” model**.

Current direction:
- Heavy compute
- Multiple architectures
- **EnsembleClassifier** approach for best accuracy

---

## 6. Train / Validation Splits

Two supported strategies:

1. Separate directories  
   - `keras_dataset/`
   - `val_dataset/`

2. Standard train/validation split within one dataset

---

## 7. Defects to Focus On

Based on current results, prioritize:

- **Negative Dents**
- **Positive Dents**
- **Welding Spots**
- **Clean tiles**

---

## 8. Tactile Sensor Data

Some datasets include tactile sensing:

```
NDA_*
NDB_*
NDC_*
PDA_*
PDB_*
PDC_*
```

These contain:
- Dent classes A / B / C
- Positive & negative dents
- Associated **tactile CSV data**

---

## 9. Annotation JSON Files

Each frame has an accompanying `.json` file containing:

- Image hash
- Defect types
- Severity
- Lists of `(X, Y)` coordinates

Parsed via:
```python
restoreFromJSON
```

https://github.com/magician-project/magician_grabber_annotator/blob/main/wxAnnotator.py#L1415

---

## 10. PNG Files & Metadata


<img src="https://github.com/magician-project/magician_grabber_annotator/blob/main/doc/classwelding.jpg?raw=true" /> 

PNG tiles are **training-ready**.

They include metadata headers that allow tracing back to the original dataset:

```bash
identify -verbose class_NegativeDentClassA/image.png | grep -A1 "Comment"
```

Example:
```
Comment:
/media/.../colorFrame_0_00000.pnm.json(528,576)
```

This allows:
- Debugging high-loss samples
- Fixing annotation errors at the source

---

## 11. Final Notes

- The same toolchain can support **any local defect detection task**
- As long as:
  - Defects are local
  - Tiling logic applies
- Annotation errors can be traced and corrected efficiently

For further questions, feel free to reach out.

---

**Author**  
Ammar Qammaz  
http://ammar.gr
ammarkov@ics.forth.gr
