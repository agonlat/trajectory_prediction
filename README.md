# Trajectory Prediction Pipeline with Trajectron++ & Social-LSTM

This repository contains the complete structure for a trajectory prediction pipeline using **Trajectron++** and **Social-LSTM**.  
It covers **data preprocessing**, **frame extraction**, **model-specific dataset building**, **training**, and **evaluation**.

⚠️ **Important Notice**  
The raw dataset **is not included** in this repository due to its large size.  
Only cleaned CSV trajectory files and quality reports are included in:

```
data/processed/
```

To run the full pipeline, you must **manually download the dataset** and insert it into the correct folder.

---

## Repository Structure

```
data/
├── processed/
│   ├── frames/                  # exported frames (from preprocessing)
│   ├── quality_report.json      # data-quality inspection report
│   └── trajectories.csv         # cleaned, standardized trajectories
│
└── raw/
    ├── annotated_frames/        # (empty until dataset is added)
    ├── Scripts/                 # (empty)
    ├── Trajectories/            # (empty)
    └── Videos/                  # (empty)
```

Other project folders:

```
configs/                         # YAML configs for models
experiments/                     # experiment setups & logs
logs/                            # logging outputs
models/                          # trained model checkpoints (optional)
notebooks/                       # exploratory data analysis
scripts/
│   ├── prepare_data.py          # raw data -> CSV + frame extraction
│   ├── generate_bev.py          # BEV / raster generation (optional)
│   ├── build_trajectronpp.py    # build Trajectron++ dataset
│   ├── build_social_lstm.py     # build Social-LSTM dataset
│   └── evaluate.py              # ADE / FDE evaluation
src/
├── trajectronpp/                # Trajectron++ implementation
├── social_lstm/                 # Social-LSTM implementation
└── utils/                        # helper functions
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
python scripts/prepare_data.py --input data/raw --out data/processed
```

This script:
- loads raw data from `data/raw/`
- extracts and synchronizes frames
- generates cleaned CSV trajectory files
- writes outputs into `data/processed/`


(Optional, used for map-based models if needed.)

### **2. Build dataset for Trajectron++**

```bash
python scripts/build_trajectronpp.py --input data/processed/trajectories.csv --out data/trajectronpp
```

This script:
- normalizes coordinates
- builds interaction graphs
- saves `.pkl` files per scene

### **3. Build dataset for Social-LSTM**

```bash
python scripts/build_social_lstm.py --input data/processed/trajectories.csv --out data/social_lstm
```

This script:
- creates observation/prediction sequences
- pads missing agents
- saves `.npy` arrays per scene

### **4. Evaluate Model Performance**

```bash
python scripts/evaluate.py
```

Computes standard metrics:
- ADE (Average Displacement Error)
- FDE (Final Displacement Error)

---

## Current State

- Preprocessing is ready and CSV files exist.
- Model-specific dataset scripts must be executed to prepare inputs for training.
- Raw dataset must be inserted into `data/raw/` to fully reproduce training.

---

## Notes

- Scripts are intended to be executed on a **local high-performance machine**.
- The structure separates raw data from processed outputs.
- After inserting the data, the entire workflow (preprocessing → dataset build → training → evaluation) can be reproduced.