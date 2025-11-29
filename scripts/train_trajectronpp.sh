#!/bin/bash
set -e
# 1. Update submodule (falls vorhanden)
git submodule update --init --recursive || true

# 2. Create conda env if not exists (optional)
if ! conda env list | grep -q "trajectronpp"; then
  echo "Creating conda env 'trajectronpp'..."
  conda create -n trajectronpp python=3.8 -y
  conda activate trajectronpp
  pip install -r third_party/trajectronpp/requirements.txt || true
else
  conda activate trajectronpp
fi

# 3. Convert CSV -> PKL
python scripts/build_trajectronpp.py --input data/processed/trajectories.csv --out data/processed/trajectronpp --threshold 6.0

# 4. Train (smoke test)
cd third_party/trajectronpp/trajectron
python train.py \
  --train_data_dict ../../../data/processed/trajectronpp/trajectron_train.pkl \
  --eval_data_dict ../../../data/processed/trajectronpp/trajectron_val.pkl \
  --offline_scene_graph yes \
  --train_epochs 3 \
  --log_dir ../../../experiments/models/trajectronpp_smoke
