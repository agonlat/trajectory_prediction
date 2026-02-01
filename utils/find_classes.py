import os
import pandas as pd

RAW_DATA_DIR = "./data/raw/Trajectories_cleaned"
OUTPUT_FILE = "detected_classes.txt"

CANONICAL_CLASSES = [
    "bicycle",
    "bus",
    "car",
    "motorcycle",
    "person",
    "truck",
]


def find_all_classes(data_dir):
    classes = set()

    for f in os.listdir(data_dir):
        if not f.endswith(".csv"):
            continue

        df = pd.read_csv(
            os.path.join(data_dir, f),
            usecols=["class"],
            engine="python",
            on_bad_lines="skip",
        )

        classes |= set(
            df["class"]
            .astype(str)
            .str.lower()
            .str.strip()
            .dropna()
            .unique()
        )

    return sorted(classes)


classes = find_all_classes(RAW_DATA_DIR)
final_classes = [c for c in CANONICAL_CLASSES if c in classes]

with open(OUTPUT_FILE, "w") as f:
    for c in final_classes:
        f.write(f"{c}\n")

print("Detected classes:")
for c in final_classes:
    print(" ", c)
