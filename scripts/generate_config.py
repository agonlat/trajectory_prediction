import os
import sys
import json
import pandas as pd
import numpy as np

# -----------------------------
# Configuration
# -----------------------------
DATA_DIR = './data/raw/Trajectories/'
OUTPUT_FILE = os.path.join('experiments', 'custom_config.json')

# Load detected classes
DETECTED_CLASSES_FILE = './detected_classes.txt'

if not os.path.exists(DETECTED_CLASSES_FILE):
    print(f"ERROR: {DETECTED_CLASSES_FILE} not found.")
    sys.exit(1)

with open(DETECTED_CLASSES_FILE, 'r') as f:
    NODE_TYPE_LIST = [line.strip().upper() for line in f if line.strip()]

if not NODE_TYPE_LIST:
    print(f"ERROR: No classes found in {DETECTED_CLASSES_FILE}")
    sys.exit(1)

print(f"Using detected classes: {NODE_TYPE_LIST}")

# -----------------------------
# Load and combine CSVs
# -----------------------------
def load_all_csvs(data_dir):
    all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
    if not all_files:
        print(f"No CSV files found in {data_dir}")
        sys.exit(1)

    df_list = []
    for f in all_files:
        try:
            df = pd.read_csv(f)
            df_list.append(df)
        except Exception as e:
            print(f"Skipping file {f} due to error: {e}")

    combined_df = pd.concat(df_list, ignore_index=True)
    return combined_df

# -----------------------------
# Generate Trajectron++ Training Config
# -----------------------------
def generate_trajectronpp_config(df):
    """
    Generate the actual hyperparameters config that Trajectron++ training expects.
    This is different from the preprocessing config.
    """
    
    # Compute standardization per class
    standardization = {}
    for node_type in NODE_TYPE_LIST:
        subset = df[df['class'].str.upper() == node_type]
        if subset.empty:
            # fallback to global mean/std
            x_mean = float(df['x_center'].mean())
            x_std = float(df['x_center'].std())
            y_mean = float(df['y_center'].mean())
            y_std = float(df['y_center'].std())
        else:
            x_mean = float(subset['x_center'].mean())
            x_std = float(subset['x_center'].std())
            y_mean = float(subset['y_center'].mean())
            y_std = float(subset['y_center'].std())
        
        standardization[node_type] = {
            'position': {
                'x': {'mean': x_mean, 'std': x_std},
                'y': {'mean': y_mean, 'std': y_std}
            },
            'velocity': {
                'x': {'mean': 0.0, 'std': 2.0},
                'y': {'mean': 0.0, 'std': 2.0}
            },
            'acceleration': {
                'x': {'mean': 0.0, 'std': 1.0},
                'y': {'mean': 0.0, 'std': 1.0}
            }
        }

    # Dynamic models for each node type
    dynamic = {}
    state = {}
    pred_state = {}
    
    for node_type in NODE_TYPE_LIST:
        # Use SingleIntegrator for all types (simplest dynamics model)
        dynamic[node_type] = {
            "name": "SingleIntegrator",
            "distribution": False,
            "limits": {}
        }
        
        # State representation (what the model tracks)
        state[node_type] = {
            "position": ["x", "y"],
            "velocity": ["x", "y"],
            "acceleration": ["x", "y"]
        }
        
        # Prediction state (what we predict)
        pred_state[node_type] = {
            "position": ["x", "y"]
        }

    # Full Trajectron++ training configuration
    config = {
        # Training hyperparameters
        "batch_size": 256,
        "grad_clip": 1.0,
        "learning_rate_style": "exp",
        "learning_rate": 0.001,
        "min_learning_rate": 0.00001,
        "learning_decay_rate": 0.9999,
        
        # Prediction settings
        "prediction_horizon": 12,
        "maximum_history_length": 8,
        "minimum_history_length": 1,
        
        # Data standardization
        "standardization": standardization,
        
        # Latent variable settings
        "k": 1,
        "k_eval": 25,
        "kl_min": 0.07,
        "kl_weight": 100.0,
        "kl_weight_start": 0,
        "kl_decay_rate": 0.99995,
        "kl_crossover": 400,
        "kl_sigmoid_divisor": 4,
        
        # RNN settings
        "rnn_kwargs": {
            "dropout_keep_prob": 0.75
        },
        "MLP_dropout_keep_prob": 0.9,
        "enc_rnn_dim_edge": 32,
        "enc_rnn_dim_edge_influence": 32,
        "enc_rnn_dim_history": 32,
        "enc_rnn_dim_future": 32,
        "dec_rnn_dim": 128,
        "q_z_xy_MLP_dims": None,
        "p_z_x_MLP_dims": 32,
        "GMM_components": 1,
        "log_p_yt_xz_max": 6,
        
        # Sampling parameters
        "N": 1,
        "K": 25,
        
        # Temperature annealing
        "tau_init": 2.0,
        "tau_final": 0.05,
        "tau_decay_rate": 0.997,
        
        # Z logit clipping
        "use_z_logit_clipping": True,
        "z_logit_clip_start": 0.05,
        "z_logit_clip_final": 5.0,
        "z_logit_clip_crossover": 300,
        "z_logit_clip_divisor": 5,
        
        # Dynamic models per node type
        "dynamic": dynamic,
        
        # State definitions
        "state": state,
        "pred_state": pred_state,
        
        # Scene graph settings
        "dynamic_edges": "yes",
        "edge_state_combine_method": "sum",
        "edge_influence_combine_method": "attention",
        "edge_addition_filter": [0.25, 0.5, 0.75, 1.0],
        "edge_removal_filter": [1.0, 0.0],
        "offline_scene_graph": "yes",
        
        # Robot node settings
        "incl_robot_node": False,
        
        # Frequency multiplication
        "node_freq_mult_train": False,
        "node_freq_mult_eval": False,
        "scene_freq_mult_train": False,
        "scene_freq_mult_eval": False,
        "scene_freq_mult_viz": False,
        
        # Edge and map encoding
        "edge_encoding": True,
        "use_map_encoding": False,
        "map_encoder": {},
        
        # Data augmentation
        "augment": True,
        "override_attention_radius": [],
        
        # Logging
        "log_histograms": False
    }
    
    return config

# -----------------------------
# Main
# -----------------------------
if __name__ == '__main__':
    print("Loading CSV files...")
    combined_df = load_all_csvs(DATA_DIR)
    print(f"Loaded {len(combined_df)} total rows")
    
    print("Generating Trajectron++ training config...")
    config_dict = generate_trajectronpp_config(combined_df)
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"Config file saved to {OUTPUT_FILE}")
    print(f"Configuration includes {len(NODE_TYPE_LIST)} node types: {NODE_TYPE_LIST}")
    print("\nYou can now run training with this config!")