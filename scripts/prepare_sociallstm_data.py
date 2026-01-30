import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------
# CONFIG — Change only these paths if needed
# ---------------------------------------------------
CSV_DIR = "data/raw/Trajectories"   #original trajectory
SOCIAL_LSTM_DATA_DIR = "social-lstm/data"  # Submodule data directory
TRAIN_DIR = os.path.join(SOCIAL_LSTM_DATA_DIR, "train")
VAL_DIR = os.path.join(SOCIAL_LSTM_DATA_DIR, "validation")
TEST_DIR = os.path.join(SOCIAL_LSTM_DATA_DIR, "test")

# ---------------------------------------------------
# Helper: Remove all files inside a folder
# ---------------------------------------------------
def clean_folder(path):
    if os.path.exists(path):
        for f in os.listdir(path):
            fp = os.path.join(path, f)
            if os.path.isfile(fp):
                os.remove(fp)
            else:
                shutil.rmtree(fp)
    else:
        os.makedirs(path)
    print(f"Cleaned: {path}")

# ---------------------------------------------------
# Convert CSV → Social-LSTM TXT
# ---------------------------------------------------
def convert_csv_to_txt(csv_path, out_path):
    df = pd.read_csv(csv_path)

    # Required Social-LSTM columns
    df_out = df[["frame", "id", "new_x_center", "new_y_center"]]
    df_out.columns = ["frame", "ped_id", "x", "y"]

    # Sort by frame then id
    df_out = df_out.sort_values(["frame", "ped_id"])

    df_out.to_csv(out_path, sep=" ", header=False, index=False)
    print(f"✔ Converted: {csv_path} → {out_path}")

# ---------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------
def main():

    # 1) Load all CSVs
    csv_files = [f for f in os.listdir(CSV_DIR) if f.endswith(".csv")]
    csv_paths = [os.path.join(CSV_DIR, f) for f in csv_files]
    print(f"Found {len(csv_paths)} CSV scenes.")

    if len(csv_paths) == 0:
        print("No CSVs found — aborting.")
        return

    # 2) Train / Validation / Test split
    train_paths, temp_paths = train_test_split(csv_paths, test_size=0.30, random_state=42)
    val_paths, test_paths = train_test_split(temp_paths, test_size=0.50, random_state=42)

    splits = {
        "train": train_paths,
        "validation": val_paths,
        "test": test_paths,
    }

    # 3) CLEAN OLD DATA
    clean_folder(TRAIN_DIR)
    clean_folder(VAL_DIR)
    clean_folder(TEST_DIR)

    # 4) CONVERT AND SAVE NEW DATA
    for split_name, file_list in splits.items():
        out_dir = os.path.join(SOCIAL_LSTM_DATA_DIR, split_name)
        os.makedirs(out_dir, exist_ok=True)

        for csv in file_list:
            scene_name = os.path.basename(csv).replace(".csv", ".txt")
            out_path = os.path.join(out_dir, scene_name)
            convert_csv_to_txt(csv, out_path)

    print("\nAll data prepared successfully!")
    print("Training scenes →", TRAIN_DIR)
    print("Validation scenes →", VAL_DIR)
    print("Test scenes →", TEST_DIR)

if __name__ == "__main__":
    main()
