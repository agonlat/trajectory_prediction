import os
import pandas as pd
import sys

# Adjust this path to the folder containing your CSV files
RAW_DATA_DIR = './data/raw/Trajectories' 
OUTPUT_FILE = 'detected_classes.txt' # Output file name

def find_all_unique_classes(data_dir):
    """Searches for all unique values in the 'class' column across all CSV files."""
    
    unique_classes = set()
    print(f"Searching for classes in: {data_dir}\n")

    if not os.path.exists(data_dir):
        print(f"ERROR: Raw data directory not found at {data_dir}.")
        return None

    file_count = 0
    
    for file_name in os.listdir(data_dir):
        if file_name.endswith('.csv'):
            file_path = os.path.join(data_dir, file_name)
            file_count += 1
            
            try:
                # Only read the 'class' column to save memory
                df = pd.read_csv(file_path, usecols=['class'])
                
                # Convert to lowercase, strip whitespace, and find unique classes
                classes_in_file = df['class'].astype(str).str.lower().str.strip().replace('', pd.NA).dropna().unique()
                
                unique_classes.update(classes_in_file)
                
            except KeyError:
                # Ignore files without a 'class' column
                pass
            except Exception as e:
                print(f"ERROR reading {file_name}: {e}")

    print(f"\n{file_count} files processed.")
    return sorted(list(unique_classes))

def write_classes_to_file(classes, output_path):
    """Writes the list of classes to a text file in minimal format."""
    
    # Content: 
    # 1. Comma-separated lowercase list (for NODE_TYPE_MAP keys)
    # 2. Python list of uppercase strings (for data_dict and configs)
    content = (
        f"{', '.join(classes)}\n"
        f"{[c.upper() for c in classes]}"
    )

    try:
        with open(output_path, 'w') as f:
            f.write(content)
        print(f"\nSuccessfully saved to file: {output_path}")
    except Exception as e:
        print(f"\nERROR saving file: {e}")

if __name__ == '__main__':
    all_classes = find_all_unique_classes(RAW_DATA_DIR)
    
    if all_classes:
        print("----------------------------------------------------------------")
        print("All unique classes found (lowercase):")
        # Print the list to the console in the desired format
        print(f"{', '.join(all_classes)}") 
        print("----------------------------------------------------------------")
        write_classes_to_file(all_classes, OUTPUT_FILE)