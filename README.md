#  Trajectory Prediction Pipeline with Trajectron++ & Social-LSTM

This repository provides a **complete, practical trajectory prediction pipeline** using **Trajectron++** and **Social-LSTM**.

The pipeline emphasizes **data sanity, class consistency, and correct preprocessing**, which are critical for stable training and meaningful evaluation.

---

##  Overview

This pipeline covers:

- Dataset inspection and class analysis  
- Cleaning and normalizing trajectory labels  
- Model-specific preprocessing  
- Training Trajectron++ and Social-LSTM  
- Evaluation of trained models  

---

## 🔍 Dataset Inspection & Class Analysis

Before any training, inspect the dataset to understand which object classes are present and whether labels are consistent.

```python
import dill
from collections import Counter

env = dill.load(open("data/processed_trajectronpp/custom_test.pkl", "rb"))
print(Counter(n.type for s in env.scenes for n in s.nodes))
```

Example output:
```
Counter({car: 32, person: 9, truck: 3, bus: 2, motorcycle: 2})
```

###  Important Interpretation

- **`car`** is the **most reliable and recommended class** for evaluation.
- **`person`** *may* work, but performance is often unstable due to sparse trajectories and inconsistent motion patterns.
- Classes such as `truck`, `bus`, and `motorcycle` typically **do not have enough samples** and are **not recommended for evaluation** unless the dataset is much larger.

➡️ For meaningful metrics (ADE / FDE), **evaluate primarily on `car`**.

---

##  Step 1: Detect All Classes (Raw CSV)

Before cleaning, detect all class names present in the raw trajectory CSV files.

```bash
python utils/find_classes.py   --input data/raw/Trajectories/   --output data/processed/detected_classes_raw.txt
```

Review the output carefully.

 **It is common to see inconsistent or unexpected names**  
(e.g. `Car`, `car_1`, `vehicle`, `ped`, etc.).

---

##  Step 2: Clean Trajectory CSV Files

Use the trajectory cleaning script to normalize class names and remove invalid entries.

```bash
python scripts/clean_trajectory_csv.py   --input data/processed/trajectories.csv   --output data/processed/trajectories_cleaned.csv
```

###  Custom Class Mapping

If your dataset uses different names, **edit the class mapping inside the script**:

```python
CLASS_MAPPING = {
    "Car": "car",
    "vehicle": "car",
    "ped": "person",
    "Pedestrian": "person"
}
```

Make sure all labels map to a **small, consistent set**.

---

##  Step 3: Re-run Class Detection (Validation)

After cleaning, verify that the labels are now consistent.

```bash
python utils/find_classes.py   --input data/processed/trajectories_cleaned.csv   --output data/processed/detected_classes_cleaned.txt
```

 At this stage, class names **should be clean and intentional**  
(e.g. only `car`, optionally `person`).

---

##  Step 4: Preprocess CSV → Trajectron++ Format

Trajectron++ **does not train directly on CSV files**.  
The data must be converted into a **specific pickled scene format (`.pkl`)**.

Run the preprocessing pipeline to generate Trajectron++-compatible files:

```bash
bash scripts/process_all_videos.sh   --input data/processed/trajectories_cleaned.csv   --output data/processed_trajectronpp/
```

This generates:
- `custom_train.pkl`
- `custom_val.pkl`
- `custom_test.pkl`

---

##  Step 5: Train Trajectron++ (Custom Command)

Use the following command to train Trajectron++:

```bash
python train.py   --train_data_dict C:\Users\agon_\trajectory_prediction\data\processed_trajectronpp\custom_train.pkl   --eval_data_dict  C:\Users\agon_\trajectory_prediction\data\processed_trajectronpp\custom_val.pkl   --log_dir         C:\Users\agon_\trajectory_prediction\experiments\custom_int_ee\models   --log_tag         custom_int_ee   --train_epochs    40   --augment   --conf            C:\Users\agon_\trajectory_prediction\trajectronpp\experiments\nuScenes\models\int_ee\config.json
```

 Trajectron++ **expects a very specific format**.  
Training will fail if preprocessing is skipped or labels are inconsistent.

---

##  Step 6: Evaluate Trajectron++

After training, evaluate the model using ADE and FDE metrics.

```bash
python scripts/evaluate.py   --predictions experiments/custom_int_ee/models/   --ground_truth data/processed_trajectronpp/
```

 **Evaluate primarily on `car`** for reliable results.

---

##  Social-LSTM Pipeline

### Step 1: Prepare Data

Convert cleaned trajectories into the Social-LSTM `.txt` format.

```bash
bash scripts/prepare_sociallstm_data.sh   --input data/processed/trajectories_cleaned.csv
```

 This script writes directly into:
```
social-lstm/data/
```

Do **not** commit changes inside the submodule.

---

### Step 2: Train Social-LSTM

```bash
bash scripts/train_sociallstm.sh
```

---

### Step 3: Evaluate Social-LSTM

After training completes, run the evaluation script provided by Social-LSTM:

```bash
$env:PYTHONPATH = "$PWD\..\..\trajectron"
python evaluate.py --model ../../../experiments/custom_int_ee/models/models_30_Jan_2026_16_00_49custom_int_ee --checkpoint 40 --data ../../../data/processed_trajectronpp/custom_test.pkl --output_path ../../../experiments/custom_int_ee/results --output_tag car_h3 --node_type car --prediction_horizon 3
```

---

## 🧠 Key Recommendations

-  **Clean labels first — always**
-  **Use `car` as the primary evaluation class**
-  `person` may work but is **not recommended** unless well-sampled
-  Re-run class detection after every cleaning step
-  Social-LSTM must be run on **Linux / WSL**
-  Do not commit changes inside submodules

---

This pipeline prioritizes **correctness over convenience**, ensuring reproducible and interpretable trajectory prediction results.
