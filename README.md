# Trajectory Prediction with DESIRE & HiVT

This repository contains the complete structure for a trajectory prediction pipeline using **DESIRE** and **HiVT**.  
It covers **data preprocessing**, **frame extraction**, **BEV/raster generation**, **training**, and **evaluation**.

⚠️ **Important Notice**  
The raw dataset **is not included** in this repository due to its large size.  
Only cleaned CSV trajectory files and quality reports are included in:

```
data/processed/
```

To run the full pipeline, you must **manually download the dataset** and insert it into the correct folder (see below).

---

## Repository Structure

```
data/
├── processed/
│   ├── frames/                  # exported frames (from preprocessing)
│   ├── quality_report.json      # data-quality inspection report
│   └── trajectories.csv         # cleaned, standardized trajectories (included)
│
└── raw/
    ├── annotated_frames/        # (empty until dataset is added)
    ├── Scripts/                 # (empty)
    ├── Trajectories/            # (empty)
    └── Videos/                  # (empty)
```

Other project folders:

```
configs/                         # YAML configs for DESIRE & HiVT
experiments/                     # experiment setups & logs
logs/                            # logging outputs and W&B links
models/                          # trained model checkpoints (optional)
notebooks/                       # exploratory data analysis
scripts/
│   ├── prepare_data.py          # raw data -> CSV + frame extraction
│   ├── generate_bev.py          # BEV / raster generation
│   └── evaluate.py              # ADE / FDE evaluation
src/
├── desire/                      # DESIRE implementation
├── hivt/                        # HiVT implementation
└── utils/                       # helper functions
```

---

## Adding the Raw Dataset

Because the raw dataset is too large, it must be downloaded separately from **Google Drive**.

### **1. Download the dataset folder named:**

```
DATASET/
```

This folder usually contains:

- Videos/  
- Trajectories/  
- Scripts/  
- annotated_frames/  

### **2. Copy the contents of `DATASET/` into:**

```
data/raw/
```

The final structure should look like:

```
data/raw/
├── annotated_frames/
├── Scripts/
├── Trajectories/
└── Videos/
```

Once these folders are added, the project is ready to run.

---

## Running the Project

### **1. Prepare the data**

```bash
python scripts/prepare_data.py
```

This script:

- loads raw data from `data/raw/`
- extracts and synchronizes frames
- generates cleaned CSV trajectory files
- writes outputs into `data/processed/`

### **2. Generate BEV / Raster Maps**

```bash
python scripts/generate_bev.py
```

Used by DESIRE and HiVT to create map-based model inputs.

### **3. Evaluate Model Performance**

```bash
python scripts/evaluate.py
```

Computes standard metrics:

- ADE (Average Displacement Error)
- FDE (Final Displacement Error)

---

## Current State

- The preprocessing pipeline has already been executed before, and models were trained successfully.
- This repository only contains:
  - **cleaned trajectories (`trajectories.csv`)**
  - **quality reports**
  - **project code and structure**
- To fully reproduce training, the raw dataset must be inserted into `data/raw/`.

---

## Notes

- Scripts are intended to be executed on a **local high-performance machine**.
- The structure is designed to fully separate raw data from processed outputs.
- After inserting the data, the entire workflow (preprocessing → BEV generation → training → evaluation) can be reproduced.

---
