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
│   ├── frames/               # Exported frames (from preprocessing)
│   ├── quality_report.json   # Data-quality inspection report
│   └── trajectories.csv      # Cleaned, standardized trajectories
└── raw/
    ├── annotated_frames/     # (empty until dataset is added)
    ├── Scripts/              # (empty)
    ├── Trajectories/         # (empty)
    └── Videos/               # (empty)
```

Other project folders:

```
configs/          # YAML configs for models
experiments/      # Experiment setups & logs
logs/             # Logging outputs
models/           # Trained model checkpoints (optional)
notebooks/        # Exploratory data analysis
scripts/
│   ├── prepare_data.py       # Raw data -> CSV + frame extraction
│   ├── generate_bev.py       # BEV / raster generation (optional)
│   ├── build_trajectronpp.py # Build Trajectron++ dataset
│   ├── build_social_lstm.py  # Build Social-LSTM dataset
│   ├── train_trajectronpp.sh # Automated training pipeline
│   └── evaluate.py           # ADE / FDE evaluation
src/
├── trajectronpp/  # Trajectron++ implementation
├── social_lstm/   # Social-LSTM implementation
└── utils/         # Helper functions
```

---

## Adding the Raw Dataset

Because the raw dataset is too large, it must be downloaded separately.

### 1. Download the dataset folder named:

```
DATASET/
```

Contains:

- Videos/
- Trajectories/
- Scripts/
- annotated_frames/

### 2. Copy the contents into:

```
data/raw/
```

Final structure:

```
data/raw/
├── annotated_frames/
├── Scripts/
├── Trajectories/
└── Videos/
```

---

## Running the Project

### 1. Prepare the data

```bash
python scripts/prepare_data.py --input data/raw --out data/processed
```

This script:

- Loads raw data
- Extracts & synchronizes frames
- Produces `trajectories.csv`
- Writes outputs to `data/processed/`

---

### 2. Build dataset for Trajectron++

```bash
python scripts/build_trajectronpp.py --input data/processed/trajectories.csv --out data/trajectronpp
```

This script:

- Normalizes coordinates
- Builds per-frame interaction graphs
- Produces `.pkl` files used for Trajectron++ training

---

### 3. Build dataset for Social-LSTM

```bash
python scripts/build_social_lstm.py --input data/processed/trajectories.csv --out data/social_lstm
```

This script:

- Creates observation/prediction sequences
- Pads missing agents
- Saves arrays for Social-LSTM training

---

## 🚀 Automated Training Script (`train_trajectronpp.sh`)

To simplify the workflow, this repository includes a fully automated training script:

```bash
scripts/train_trajectronpp.sh
```

This script performs all steps automatically:

- Initializes the Trajectron++ submodule
- Creates/activates the Conda environment
- Installs dependencies
- Converts the CSV into `.pkl` files
- Starts Trajectron++ training
- Stores logs & checkpoints in `experiments/models/trajectronpp/`

---

### 🟦 Running on Windows

Windows does not execute `.sh` files natively. Use Git Bash:

1. Open Git Bash inside the repository folder
2. Make the script executable:

```bash
chmod +x scripts/train_trajectronpp.sh
```

3. Run it:

```bash
./scripts/train_trajectronpp.sh
```

---

### 🟩 Optional: Using VSCode Terminal

- Open Terminal → New Terminal
- Select **Git Bash** as the shell
- Run:

```bash
./scripts/train_trajectronpp.sh
```

---

### 🟨 What the Script Does (Step-by-Step)

- Updates/initializes the Trajectron++ Git submodule
- Creates and activates a Conda environment
- Installs Python dependencies
- Builds the Trajectron++ dataset (`trajectron_train.pkl`, etc.)
- Starts a default Trajectron++ training run
- Writes logs and model checkpoints into:

```bash
experiments/models/trajectronpp/
```

---

## Evaluation

After training, evaluate performance:

```bash
python scripts/evaluate.py
```

Computes:

- **ADE** (Average Displacement Error)
- **FDE** (Final Displacement Error)

