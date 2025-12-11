# 🤖 Trajectory Prediction Pipeline with Trajectron++ & Social-LSTM

This repository contains a complete trajectory prediction pipeline using **Trajectron++** and **Social-LSTM**.

It covers:

- Data preprocessing
- Frame extraction
- Model-specific dataset creation
- Training
- Evaluation

---

## ⚠️ Important Notices

- **Raw Dataset:** Raw dataset files are not included due to their size. Only cleaned CSV trajectory files and quality reports are included (`data/processed/`). To run the full pipeline, the original dataset must be downloaded manually.
- **Social-LSTM Compatibility:** Social-LSTM runs only on Linux/WSL. All provided shell scripts (`.sh`) must be executed in Linux or WSL, not Windows PowerShell.
- **Social-LSTM Data Location:** The script `prepare_sociallstm_data.sh` automatically generates `.txt` files directly inside `social-lstm/data/`. The `train.py` in Social-LSTM requires data in this exact location.
- **Social-LSTM Submodule:** Do not commit or push any changes made inside the `social-lstm` submodule — Git will not allow pushes. All modifications inside Social-LSTM should remain local. Only files in the root repository (e.g., `scripts/`, `experiments/`) can be safely committed and pushed.

---

## 📂 Repository Structure

The repository is structured to separate configuration, data, scripts, and model-specific submodules.

| Directory         | Purpose                                  |
|------------------|------------------------------------------|
| `configs/`        | YAML configurations for models           |
| `experiments/`    | Experiment setups & logs                 |
| `logs/`           | Logging outputs                          |
| `models/`         | Trained model checkpoints (optional)     |
| `notebooks/`      | Exploratory Data Analysis                |
| `scripts/`        | All scripts for preprocessing & training (`.sh` and `.py`) |
| `utils/`          | Utility functions (e.g., `find_classes.py`) |
| `social-lstm/`    | Submodule for Social-LSTM                |
| `trajectronpp/`   | Submodule for Trajectron++               |

### Data Structure (`data/`)

```
data/
├── processed/
│   ├── frames/               # Exported frames
│   ├── quality_report.json   # Data quality report
│   └── trajectories.csv      # Cleaned trajectories
├── processed_trajectronpp/   # Trajectron++ specific data output
└── raw/
    ├── annotated_frames/     # Original raw data (after download)
    ├── Scripts/              
    ├── Trajectories/         
    └── Videos/              
```

---

## 🛠️ Installation & Setup (Linux/WSL)

### 1. Clone the Repository

Clone the repository, making sure to include the Social-LSTM submodule:

```bash
git clone --recurse-submodules https://github.com/yourusername/trajectory_prediction.git
cd trajectory_prediction
```

### 2. Create and Activate Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🏃 Data Preparation & Training Steps

The order of operations is critical for a smooth run.

### 1. Detect Classes

Automatically detect the classes (e.g., pedestrian, car) present in your raw dataset:

```bash
python utils/find_classes.py --input data/raw/Trajectories/ --output data/processed/detected_classes.txt
```

### 2. Generate Configuration File for Trajectron++

Use the detected classes to generate the Trajectron++ configuration file:

```bash
python scripts/generate_config.py --classes data/processed/detected_classes.txt --output experiments/custom_config.json
```

### 3. Process All Videos for Trajectron++

Process the raw video data to create the dataset format required by Trajectron++:

```bash
bash scripts/process_all_videos.sh --input data/raw/Trajectories/ --output data/processed_trajectronpp/
```

### 4. Prepare Social-LSTM Data

Convert the cleaned trajectories into the `.txt` format required by Social-LSTM.

⚠️ **Important:** This script will generate `.txt` files directly in `social-lstm/data/`. Do not modify the submodule manually or commit its contents.

```bash
bash scripts/prepare_sociallstm_data.sh --input data/processed/trajectories.csv
```

### 5. Train Trajectron++

Start training the Trajectron++ model. Use `--device cuda` for GPU training:

```bash
bash scripts/train_trajectronpp.sh --device cuda
```

### 6. Train Social-LSTM

Start training the Social-LSTM model.

Ensure the `.txt` files generated in the previous step exist in `social-lstm/data/`.

```bash
bash scripts/train_sociallstm.sh
```

---

## 📊 Evaluation

After training, evaluate the model performance. This script will compute **ADE** (Average Displacement Error) and **FDE** (Final Displacement Error).

```bash
python scripts/evaluate.py --predictions experiments/models/trajectronpp/ --ground_truth data/processed_trajectronpp/
```

---

## 💡 Notes & Recommendations

- **Social-LSTM is Linux-only** — the `.sh` scripts must be run in WSL or Linux.
- **Commit Safety:** Only files in the root `trajectory_prediction` repo should be committed. Changes inside `social-lstm/` are local and cannot be pushed.
- **GPU Setup:** For GPU training, ensure your CUDA versio