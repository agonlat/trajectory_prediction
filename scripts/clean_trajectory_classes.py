import os
import glob
import pandas as pd
from collections import Counter

# Directories
RAW_DATA_DIR = os.path.join("data", "raw", "Trajectories")
CLEAN_DATA_DIR = os.path.join("data", "raw", "Trajectories_cleaned")
os.makedirs(CLEAN_DATA_DIR, exist_ok=True)

# Class definitions
CANONICAL_CLASSES = {
    "car",
    "truck",
    "bus",
    "bicycle",
    "motorcycle",
    "person",
}

CLASS_MAP = {
    "car": "car",
    "caar": "car",

    "person": "person",
    "person,": "person",
    "perso": "person",
    "prson": "person",
    "person 222": "person",

    "motorcycle": "motorcycle",
    "motor": "motorcycle",
    "motocycle": "motorcycle",
    "motorcyclr": "motorcycle",

    "bicycle": "bicycle",
    "truck": "truck",
    "bus": "bus",
}

INVALID_CLASSES = {
    "12", "13762", "2", "51", "8", "v", "r", "ar"
}

# Statistics
global_before = Counter()
global_after = Counter()

# Load CSV files
files = sorted(glob.glob(os.path.join(RAW_DATA_DIR, "*.csv")))
print(f"Found {len(files)} CSV files")

for csv_path in files:
    try:
        df = pd.read_csv(csv_path, engine="python", on_bad_lines="skip")
    except Exception as e:
        print(f"Skipping {csv_path}: {e}")
        continue

    if "class" not in df.columns or "id" not in df.columns:
        continue

    df["class"] = (
        df["class"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    cleaned_tracks = []

    for tid, track in df.groupby("id"):
        cls = track["class"].iloc[0]
        global_before[cls] += 1

        if cls in INVALID_CLASSES:
            continue

        if cls not in CLASS_MAP:
            continue

        norm_cls = CLASS_MAP[cls]

        if norm_cls not in CANONICAL_CLASSES:
            continue

        track = track.copy()
        track["class"] = norm_cls
        cleaned_tracks.append(track)
        global_after[norm_cls] += 1

    if cleaned_tracks:
        out_df = pd.concat(cleaned_tracks, ignore_index=True)
        out_path = os.path.join(
            CLEAN_DATA_DIR,
            os.path.basename(csv_path)
        )
        out_df.to_csv(out_path, index=False)

# Output statistics
print("\n=== CLASS DISTRIBUTION AFTER CLEANING ===")
for k, v in global_after.items():
    print(f"{k:12s} : {v}")
