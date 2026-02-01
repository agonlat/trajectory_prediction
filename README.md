#  Trajectory Prediction Pipeline with Trajectron++ & Social-LSTM

This repository provides a **complete, practical trajectory prediction pipeline** using **Trajectron++** and **Social-LSTM**.

---

##  Overview

This pipeline covers:

- Dataset inspection and class analysis  
- Cleaning and normalizing trajectory labels  
- Model-specific preprocessing  
- Training Trajectron++ and Social-LSTM  
- Evaluation of trained models  

---

## Dataset Inspection & Class Analysis

Before any training, inspect the dataset to understand which object classes are present and whether labels are consistent.


##  Step 1: Detect All Classes (Raw CSV)

Before cleaning, detect all class names present in the raw trajectory CSV files.

```bash
python utils/find_classes.py
```

Review the output carefully.

 **It is common to see inconsistent or unexpected names**  
(e.g. `Car`, `car_1`, `vehicle`, `ped`, etc.).

---

##  Step 2: Clean Trajectory CSV Files

Use the trajectory cleaning script to normalize class names and remove invalid entries. Be careful: The cleaning script is based on the detected classes run by find_classes.py, therefore the mapping in the clean_trajectory_classes.py used the specific wrong agent names that were detected. eg.: ped -> pedestrian

```bash
python scripts/clean_trajectory_classes.py   
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
python utils/find_classes.py
```

 At this stage, class names **should be clean and intentional**  
(e.g. only `car`, optionally `person`).

---

##  Step 4: Preprocess CSV → Trajectron++ Format

Trajectron++ **does not train directly on CSV files**.  
The data must be converted into a **specific pickled scene format (`.pkl`)**.
The specific structure can be analyzed in the official paper.

Run the preprocessing pipeline to generate Trajectron++-compatible files:

```bash
python scripts/prepare_trajectronpp_data.py
```

This generates at data/:
- `custom_train.pkl`
- `custom_val.pkl`
- `custom_test.pkl`

---

##  Step 5: Train Trajectron++ (Custom Command)

Use the following command inside the submodule trajectronpp/trajectron to train Trajectron++:
Note that the dicts have to be changed and adjusted to one's specific computer.

```bash
python train.py   --train_data_dict C:\Users\agon_\trajectory_prediction\data\processed_trajectronpp\custom_train.pkl   --eval_data_dict  C:\Users\agon_\trajectory_prediction\data\processed_trajectronpp\custom_val.pkl   --log_dir         C:\Users\agon_\trajectory_prediction\experiments\custom_int_ee\models   --log_tag         custom_int_ee   --train_epochs    40   --augment   --conf            C:\Users\agon_\trajectory_prediction\trajectronpp\config\config.json
```

 Trajectron++ **expects a very specific format**.  
Training will fail if preprocessing is skipped or labels are inconsistent.

---

##  Step 6: Evaluate Trajectron++

After training, evaluate the model using ADE and FDE metrics. Note that the paths can be adjusted

```bash
python .\experiments\pedestrians\evaluate.py `
  --model ..\experiments\custom_int_ee\models\models_31_Jan_2026_10_38_26custom_int_ee `
  --checkpoint 40 `
  --data ..\data\processed_trajectronpp\custom_train.pkl `
  --output_path .\experiments\pedestrians\results `
  --output_tag custom_car `
  --node_type person

```

 **Evaluate primarily on `car`** for reliable results.

---

##  Social-LSTM Pipeline

### Step 1: Prepare Data

Convert cleaned trajectories into the Social-LSTM `.txt` format.

```bash
python scripts/prepare_sociallstm_data.py
```

This script writes directly into:
```
social-lstm/data/
```



---

### Step 2: Train Social-LSTM inside social-lstm submodule

```bash
python social-lstm/train.py
```

---

### Step 3: Evaluate Social-LSTM

After training completes, run the evaluation script provided by Social-LSTM:

```bash
$env:PYTHONPATH = "$PWD\..\..\trajectron"
python evaluate.py --model ../../../experiments/custom_int_ee/models/models_30_Jan_2026_16_00_49custom_int_ee --checkpoint 40 --data ../../../data/processed_trajectronpp/custom_test.pkl --output_path ../../../experiments/custom_int_ee/results --output_tag car_h3 --node_type car --prediction_horizon 3
```

---
