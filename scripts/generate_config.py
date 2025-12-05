import pandas as pd
import numpy as np
import json
import os
import sys

# --- CONFIGURATION ---
# !!! CRITICAL: CHANGE THIS PATH to the folder containing your 60 CSV files !!!
# Example: DATA_DIR = './data/raw_videos/'
DATA_DIR = './data/raw/Trajectories/'
OUTPUT_FILE = os.path.join('experiments', 'custom_config.json')
# ---------------------

def load_and_combine_data(data_dir, single_file_fallback=None):
    """Loads all CSV files in a directory and combines them for statistics."""
    
    all_data = []
    
    # Check for multiple files
    all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    # --- FALLBACK LOGIC FOR SINGLE FILE EXECUTION (REMOVE FOR 60 VIDEOS) ---
    if len(all_files) == 1 and single_file_fallback:
        print(f"Loading single file: {single_file_fallback} for demonstration.")
        all_files = [single_file_fallback]
        
    elif len(all_files) == 0:
        print(f"Error: No CSV files found in '{data_dir}'. Exiting.")
        sys.exit(1)
    # -----------------------------------------------------------------------

    print(f"Found {len(all_files)} files. Combining data...")
    
    for f_path in all_files:
        try:
            df = pd.read_csv(f_path)
            all_data.append(df)
        except Exception as e:
            print(f"Skipping file {f_path} due to error: {e}")

    if not all_data:
        print("No data loaded. Exiting.")
        sys.exit(1)
        
    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df

def generate_config_json(df):
    """Calculates standardization values and constructs the full Trajectron++ JSON structure."""
    
    # 1. CALCULATE STANDARDIZATION VALUES
    x_mean = float(df['x_center'].mean())
    x_std = float(df['x_center'].std())
    y_mean = float(df['y_center'].mean())
    y_std = float(df['y_center'].std())
    
    # 2. DEFINE THE FULL JSON STRUCTURE
    config = {
        "scene_freq": 1,
        "dt": 0.4,
        "max_scene_size": 1200,
        "data_format": ["x", "y"],
        
        "standardization": {
            "car": {
                "x": {"mean": x_mean, "std": x_std},
                "y": {"mean": y_mean, "std": y_std}
            },
            "truck": {
                "x": {"mean": x_mean, "std": x_std},
                "y": {"mean": y_mean, "std": y_std}
            }
        },
        
        "edge_types": [
            {"hierarchy": 1, "primary": "car", "secondary": "car", "name": "c_c"},
            {"hierarchy": 1, "primary": "truck", "secondary": "truck", "name": "t_t"},
            {"hierarchy": 1, "primary": "car", "secondary": "truck", "name": "c_t"},
            {"hierarchy": 1, "primary": "truck", "secondary": "car", "name": "t_c"}
        ],
        
        "node_type_parameters": {
            "car": {
                "model_type": "trajectron.model.Trajectron",
                "dynamic_model": "SingleIntegrator",
                "prediction_horizon": 12,
                "history_len": 8,
                "state_format": "x,y",
                "state_length": 2,
                "output_states": ["x", "y"],
                "map_encoder": None
            },
            "truck": {
                "model_type": "trajectron.model.Trajectron",
                "dynamic_model": "SingleIntegrator",
                "prediction_horizon": 12,
                "history_len": 8,
                "state_format": "x,y",
                "state_length": 2,
                "output_states": ["x", "y"],
                "map_encoder": None
            }
        }
    }
    return config

# --- MAIN EXECUTION ---
if __name__ == '__main__':
    combined_data = load_and_combine_data(DATA_DIR, single_file_fallback='Video 113.csv')

    config_dict = generate_config_json(combined_data)

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    # Save the JSON file
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(config_dict, f, indent=4)
        
    print(f"\nConfiguration file successfully generated at: {OUTPUT_FILE}")