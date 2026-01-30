import os
import pandas as pd

# Adjust this path to the folder containing your CSV files
RAW_DATA_DIR = './data/raw/Trajectories_cleaned'
OUTPUT_FILE = 'detected_classes.txt'

# Canonical classes you actually support
CANONICAL_CLASSES = [
    "bicycle",
    "bus",
    "car",
    "motorcycle",
    "person",
    "truck"
]

def find_all_unique_classes(data_dir):
    """Search for all unique values in the 'class' column across all CSV files."""
    
    unique_classes = set()
    print(f"Searching for classes in: {data_dir}\n")

    if not os.path.exists(data_dir):
        print(f"ERROR: Raw data directory not found at {data_dir}.")
        return None

    file_count = 0
    
    for file_name in os.listdir(data_dir):
        if not file_name.endswith('.csv'):
            continue

        file_path = os.path.join(data_dir, file_name)
        file_count += 1
        
        try:
            df = pd.read_csv(
                file_path,
                usecols=['class'],
                engine='python',
                on_bad_lines='skip'
            )

            classes_in_file = (
                df['class']
                .astype(str)
                .str.lower()
                .str.strip()
                .replace('', pd.NA)
                .dropna()
                .unique()
            )

            for cls in classes_in_file:
                # split accidental comma-separated entries
                if ',' in cls:
                    parts = [p.strip() for p in cls.split(',')]
                    unique_classes.update(parts)
                else:
                    unique_classes.add(cls)

        except Exception as e:
            print(f"ERROR reading {file_name}: {e}")

    print(f"\n{file_count} files processed.")
    return sorted(unique_classes)

def write_classes_to_file(classes, output_path):
    """Write ONE class per line, lowercase."""
    
    with open(output_path, 'w') as f:
        for cls in classes:
            f.write(f"{cls}\n")

    print(f"\nSuccessfully saved to file: {output_path}")

if __name__ == '__main__':
    all_classes = find_all_unique_classes(RAW_DATA_DIR)

    if not all_classes:
        raise RuntimeError("No classes found.")

    # Keep only canonical classes, in fixed order
    final_classes = [c for c in CANONICAL_CLASSES if c in all_classes]

    print("----------------------------------------------------------------")
    print("Final detected classes:")
    for c in final_classes:
        print(c)
    print("----------------------------------------------------------------")

    write_classes_to_file(final_classes, OUTPUT_FILE)
