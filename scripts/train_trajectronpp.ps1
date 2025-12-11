#!/bin/bash
# Trajectron++ Training Script (Bash Version)

# --- Colors for output ---
Red='\033[0;31m'
Green='\033[0;32m'
Yellow='\033[0;33m'
Cyan='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${Cyan}Checking if required files exist...${NC}"

# --- 1. SET PROJECT PATHS ---

# Resolve project root (parent of scripts/)
ScriptDir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ProjectRoot="$(dirname "$ScriptDir")"

# Paths
CONF_FILE="$ProjectRoot/experiments/custom_config.json"
TRAIN_DATA="$ProjectRoot/data/processed_trajectronpp/custom_train.pkl"
EVAL_DATA="$ProjectRoot/data/processed_trajectronpp/custom_val.pkl"
LOG_DIR="$ProjectRoot/experiments/my_trajectron_run/models"
TRAIN_SCRIPT="$ProjectRoot/trajectronpp/trajectron/train.py"

# --- 2. CHECK REQUIRED FILES ---
requiredFiles=("$CONF_FILE" "$TRAIN_DATA" "$EVAL_DATA" "$TRAIN_SCRIPT")

for f in "${requiredFiles[@]}"; do
    if [ ! -f "$f" ]; then
        echo -e "${Red}ERROR: Missing required file: $f${NC}"
        exit 1
    fi
done

echo -e "${Green}✓ All files found${NC}"
echo -e "${Green}✓ Train script: $TRAIN_SCRIPT${NC}"

# Create log directory if missing
if [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR"
    echo "Created log directory: $LOG_DIR"
fi

# --- 3. RUN TRAINING ---
echo
echo -e "${Yellow}==========================================${NC}"
echo -e "${Yellow}Starting Trajectron++ Training${NC}"
echo -e "${Yellow}==========================================${NC}"
echo "Configuration: $CONF_FILE"
echo "Training Data: $TRAIN_DATA"
echo "Validation Data: $EVAL_DATA"
echo "Log Directory: $LOG_DIR"
echo -e "${Yellow}==========================================${NC}"
echo

# Ensure Python venv is active (check VIRTUAL_ENV)
if [ -z "$VIRTUAL_ENV" ]; then
    echo -e "${Red}ERROR: Python venv is NOT active. Please run:${NC}"
    echo "       source .../trajectory_prediction/venv/bin/activate"
    exit 1
fi

# Run Python training
python "$TRAIN_SCRIPT" \
    --eval_every 10 \
    --vis_every 1 \
    --train_epochs 20 \
    --augment \
    --train_data_dict "$TRAIN_DATA" \
    --eval_data_dict "$EVAL_DATA" \
    --conf "$CONF_FILE" \
    --offline_scene_graph yes \
    --preprocess_workers 5 \
    --log_dir "$LOG_DIR" \
    --log_tag _video_data_si \
    --device cuda

# Capture exit code
if [ $? -eq 0 ]; then
    echo
    echo -e "${Green}==========================================${NC}"
    echo -e "${Green}Training completed successfully!${NC}"
    echo -e "${Green}==========================================${NC}"
else
    echo
    echo -e "${Red}==========================================${NC}"
    echo -e "${Red}Training failed.${NC}"
    echo -e "${Red}==========================================${NC}"
    exit 1
fi
