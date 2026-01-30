import os
import glob
import pandas as pd
from collections import Counter

# Paths

RAW_DATA_DIR = os.path.join("data", "raw", "Trajectories")
CLEAN_DATA_DIR = os.path.join("data", "raw", "Trajectories_cleaned")

os.makedirs(CLEAN_DATA_DIR, exist_ok=True)


# Canonical class set

CANONICAL_CLASSES = {
    "car",
    "truck",
    "bus",
    "bicycle",
    "motorcycle",
    "person"
}


# Class normalization map (typos → canonical)

CLASS_MAP = {
    # cars
    "car": "car",
    "caar": "car",

    # persons
    "person": "person",
    "person,": "person",
    "person 222": "person",
    "perso": "person",
    "prson": "person",

    # motorcycles
    "motorcycle": "motorcycle",
    "motocycle": "motorcycle",
    "motorcyce": "motorcycle",
    "motorcyclr": "motorcycle",
    "motor": "motorcycle",

    # bicycles
    "bicycle": "bicycle",

    # trucks / buses
    "truck": "truck",
    "bus": "bus"
}


# Junk / invalid labels (drop entire tracks)
INVALID_CLASSES = {
    "12", "13762", "2", "51", "8", "v", "r", "ar"
}


# Stats

global_before = Counter()
global_after = Counter()
dropped_tracks = 0
kept_tracks = 0
skipped_files = 0

# Cleaning loop

files = sorted(glob.glob(os.path.join(RAW_DATA_DIR, "*.csv")))
print(f"Found {len(files)} CSV files")

for csv_path in files:
    try:
        df = pd.read_csv(
            csv_path,
            engine="python",      # tolerant parser
            on_bad_lines="skip"   # skip malformed rows
        )
    except Exception as e:
        print(f"Skipping {os.path.basename(csv_path)}: {e}")
        skipped_files += 1
        continue

    # Required columns
    if "class" not in df.columns or "id" not in df.columns:
        print(f"Skipping {os.path.basename(csv_path)} (missing 'class' or 'id')")
        skipped_files += 1
        continue

    df["class"] = df["class"].astype(str).str.strip().str.lower()

    cleaned_tracks = []

    for track_id, track in df.groupby("id"):
        cls = track["class"].iloc[0]
        global_before[cls] += 1

        # Drop junk labels
        if cls in INVALID_CLASSES:
            dropped_tracks += 1
            continue

        # Normalize typos
        if cls in CLASS_MAP:
            norm_cls = CLASS_MAP[cls]
        else:
            dropped_tracks += 1
            continue

        # Safety check
        if norm_cls not in CANONICAL_CLASSES:
            dropped_tracks += 1
            continue

        track = track.copy()
        track["class"] = norm_cls
        cleaned_tracks.append(track)

        global_after[norm_cls] += 1
        kept_tracks += 1

    # Save cleaned CSV
    if cleaned_tracks:
        clean_df = pd.concat(cleaned_tracks, ignore_index=True)
        out_path = os.path.join(CLEAN_DATA_DIR, os.path.basename(csv_path))
        clean_df.to_csv(out_path, index=False)

# =================================================
# Report
# =================================================
print("\n================ CLEANING REPORT ================\n")
print(f"CSV files processed : {len(files) - skipped_files}")
print(f"CSV files skipped   : {skipped_files}\n")

print(f"Tracks kept         : {kept_tracks}")
print(f"Tracks dropped      : {dropped_tracks}\n")

print("Class distribution BEFORE:")
for k, v in global_before.most_common():
    print(f"  {k:15s} : {v}")

print("\nClass distribution AFTER:")
for k, v in global_after.most_common():
    print(f"  {k:15s} : {v}")

print("\nCleaned files written to:")
print(f"  {CLEAN_DATA_DIR}")
print("\n=================================================")
