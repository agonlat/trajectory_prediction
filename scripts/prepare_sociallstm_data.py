import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split


# CONFIG 

CSV_DIR = "data/raw/Trajectories"        # Folder with CSV trajectory files
SOCIAL_LSTM_DATA_DIR = "social-lstm/data"

TRAIN_DIR = os.path.join(SOCIAL_LSTM_DATA_DIR, "train")
VAL_DIR = os.path.join(SOCIAL_LSTM_DATA_DIR, "validation")
TEST_DIR = os.path.join(SOCIAL_LSTM_DATA_DIR, "test")


# Helper: Remove all files inside a folder

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


# Convert CSV → Social-LSTM TXT

def convert_csv_to_txt(csv_path, out_path):
    df = pd.read_csv(
    csv_path,
    engine="python",      # more tolerant parser
    on_bad_lines="skip"   # skip malformed rows
)


    # Keep only pedestrians (IMPORTANT for Social-LSTM)
    if "class" in df.columns:
        df = df[df["class"] == "person"]

    # Sanity check
    required_cols = {"frame", "id", "x_center", "y_center"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {missing}")

    # Select required columns
    df_out = df[["frame", "id", "x_center", "y_center"]].copy()
    df_out.columns = ["frame", "ped_id", "x", "y"]

    # Sort by frame then pedestrian id
    df_out = df_out.sort_values(["frame", "ped_id"])

    # Save in Social-LSTM format
    df_out.to_csv(out_path, sep=" ", header=False, index=False)
    print(f"✔ Converted: {os.path.basename(csv_path)}")


# MAIN PIPELINE

def main():

    # 1) Load all CSV files
    csv_files = [f for f in os.listdir(CSV_DIR) if f.endswith(".csv")]
    csv_paths = [os.path.join(CSV_DIR, f) for f in csv_files]

    print(f"Found {len(csv_paths)} CSV scenes.")

    if len(csv_paths) == 0:
        print("No CSV files found. Aborting.")
        return

    # 2) Train / Validation / Test split (by scene)
    train_paths, temp_paths = train_test_split(
        csv_paths, test_size=0.30, random_state=42
    )
    val_paths, test_paths = train_test_split(
        temp_paths, test_size=0.50, random_state=42
    )

    splits = {
        "train": train_paths,
        "validation": val_paths,
        "test": test_paths,
    }

    # 3) Clean old Social-LSTM data
    clean_folder(TRAIN_DIR)
    clean_folder(VAL_DIR)
    clean_folder(TEST_DIR)

    # 4) Convert CSVs to TXT
    for split_name, file_list in splits.items():
        out_dir = os.path.join(SOCIAL_LSTM_DATA_DIR, split_name)
        os.makedirs(out_dir, exist_ok=True)

        for csv_path in file_list:
            scene_name = os.path.basename(csv_path).replace(".csv", ".txt")
            out_path = os.path.join(out_dir, scene_name)
            convert_csv_to_txt(csv_path, out_path)

    print("\nAll data prepared successfully!")
    print("Training data   →", TRAIN_DIR)
    print("Validation data →", VAL_DIR)
    print("Test data       →", TEST_DIR)

# Entry point

if __name__ == "__main__":
    main()
