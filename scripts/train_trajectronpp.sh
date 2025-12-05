#!/bin/bash

# --- 1. SET PROJECT PATHS ---
# Define the root directory of your project (where the trajectronpp submodule is located)
PROJECT_ROOT=$(dirname "$0")/..

# Define paths to your custom files
CONF_FILE="${PROJECT_ROOT}/experiments/custom_config.json"
TRAIN_DATA="${PROJECT_ROOT}/data/processed_trajectronpp/custom_train.pkl"
EVAL_DATA="${PROJECT_ROOT}/data/processed_trajectronpp/custom_val.pkl"
LOG_DIR="${PROJECT_ROOT}/experiments/my_trajectron_run/models"

# Define the path to the Trajectron++ train script inside the submodule
TRAIN_SCRIPT="${PROJECT_ROOT}/trajectronpp/experiments/train.py"

# --- 2. EXECUTION ---
echo "Starting Trajectron++ Training with Custom Data..."
echo "Configuration: ${CONF_FILE}"
echo "Log Directory: ${LOG_DIR}"

# Run the training command
python "${TRAIN_SCRIPT}" \
    \
    # General Parameters
    --eval_every 10 \
    --vis_every 1 \
    --train_epochs 100 \
    --augment \
    \
    # Data and Configuration Paths
    --train_data_dict "${TRAIN_DATA}" \
    --eval_data_dict "${EVAL_DATA}" \
    --conf "${CONF_FILE}" \
    \
    # Hardware and Processing Options
    --offline_scene_graph yes \
    --preprocess_workers 5 \
    --log_dir "${LOG_DIR}" \
    --log_tag _video_data_si \
    --device cuda
    
echo "Training command executed."