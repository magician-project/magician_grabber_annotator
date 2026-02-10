# Dataset Access Guidelines – MAGICIAN Project

## Overview

MAGICIAN project datasets are provided **via the Magician Annotator tool**.  
Direct dataset downloads are **not publicly available**. Access is granted **only after approval by the MAGICIAN Data Access Committee (DAC)** and the signing of a **Data Transfer Agreement (DTA)**.

This document describes **how to access the tool**, **how to request dataset access**, and **how approved users receive the data**.

---

## 1. Tool Availability

The dataset is accessed through the **Magician Annotator**, a Python-based graphical tool that allows users to:

- Download approved datasets  
- View and inspect images  
- Annotate data for research and development purposes  

The tool itself is publicly available and can be installed without requiring a GitHub account.

Repository:
```
https://github.com/magician-project/magician_grabber_annotator
```

---

## 2. Tool Installation

### Step 1 – Download the Annotator

Download the latest version of the tool:
```
https://github.com/magician-project/magician_grabber_annotator/archive/refs/heads/main.zip
```

Extract the archive locally.

---

### Step 2 – Install Dependencies

From the extracted directory, run:
```bash
python3 -m pip install -r requirements.txt
```

This installs all required Python dependencies.

---

### Step 3 – Run the Annotator

Launch the tool using:
```bash
python3 wxAnnotator.py
```

At this stage, the user has full access to the **annotation tool itself**, but **no access to MAGICIAN consortium datasets**.
You can use it to annotate your data and to facilitate your own defect detection pipeline.

For a more [thorough guide on the Annotator Tool click here](https://github.com/magician-project/magician_grabber_annotator/blob/main/doc/annotator_guide.md) 

---

## 3. Dataset Access Policy

Due to the various stakeholders, MAGICIAN datasets are **not directly publicly accessible** and are provided only after:

1. Submission of a **formal data access request**
2. Approval by the **MAGICIAN Data Access Committee (DAC)**
3. Signature of a **Data Transfer Agreement (DTA)** by all parties

The data is intended **strictly for R&D activities within the scope of the MAGICIAN project**.

No commercial use or redistribution is permitted.

---

## 4. Requesting Dataset Access

Interested partners must submit a dataset access request including:

- Purpose of the request (e.g. R&D, benchmarking, validation)
- Description of the requested dataset
- Technical and ethical rationale
- Intended usage limitations and safeguards
- Explicit confirmation on whether the data involves humans

Once reviewed and approved, the requester will be contacted by the data owner.

---

## 5. Credential-Based Access

After approval:

- The data owner creates a **unique username and password**
- Credentials are provided securely to the approved partner
- These credentials are entered directly into the **Magician Annotator GUI**

Once authenticated, the user can:

- Download approved datasets
- View and inspect images
- Perform annotations within the tool

---

## 6. Data Handling and Safeguards

Approved users must comply with the following:

- Access restricted to authorized project members
- Secure, access-controlled storage
- No redistribution to third parties
- Use limited strictly to MAGICIAN project R&D
- Deletion or return of data upon request or project completion

---

## 7. Notes

- Dataset access is **not immediate** and depends on DAC approval.
- The tool may display an informational prompt when a user without credentials attempts to download data.
- All access is logged and traceable.

---

## Contact

For dataset access requests or technical issues, please contact the MAGICIAN Data Access Committee via the project coordination channels @ https://www.magician-project.eu

