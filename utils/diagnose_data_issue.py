import os
import pandas as pd
import numpy as np
import glob

RAW_DATA_DIR = "./data/raw/Trajectories"

print("\n==============================")
print("CHECKING FOR MALFORMED DATA")
print("==============================\n")

csv_files = glob.glob(os.path.join(RAW_DATA_DIR, "*.csv"))

def safe_load(path):
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"FAILED TO READ CSV: {path}")
        print("   Error:", e)
        return None

for file in csv_files:
    print("\n--------------------------------------")
    print("Checking:", os.path.basename(file))

    df = safe_load(file)
    if df is None:
        continue
    
    # 1. Check required columns
    required = ['frame','class','id','x_center','y_center']
    missing = [c for c in required if c not in df.columns]
    
    if missing:
        print("Missing required columns:", missing)
        continue
    else:
        print("Columns OK")

    # 2. Check for non-numeric positions
    for c in ['x_center','y_center']:
        if not pd.api.types.is_numeric_dtype(df[c]):
            print(f"NON-NUMERIC VALUES IN {c}:")
            print(df[c].head())
            continue

    # 3. Check invalid frame values
    if df['frame'].isnull().any():
        print("NULL FRAMES DETECTED")
    if (df['frame'] <= 0).any():
        print("INVALID FRAME NUMBER (<=0) detected")

    # 4. Check ID consistency
    bad_ids = df[df['id'].isnull()]
    if not bad_ids.empty:
        print("NULL IDs:")
        print(bad_ids.head())

    # 5. Check if a track changes class mid-sequence
    for track_id, tdf in df.groupby('id'):
        unique_classes = tdf['class'].dropna().unique()
        if len(unique_classes) > 1:
            print(f"⚠ Track {track_id} has MULTIPLE class values: {unique_classes}")

    # 6. Check for malformed strings in "class"
    for value in df['class'].dropna().unique():
        if not isinstance(value, str):
            print("CLASS IS NOT STRING:", value, type(value))
            continue
        if len(value.strip()) == 0:
            print("EMPTY CLASS STRING")
        if "," in value or "[" in value or "]" in value:
            print("Suspicious class:", value)

    # 7. Check for NaNs in positions
    if df[['x_center','y_center']].isnull().any().any():
        print("NaN positions present")

    print("✔ File check complete")
