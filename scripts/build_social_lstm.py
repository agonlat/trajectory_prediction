
# =============================================================
# build_social_lstm.py
# =============================================================
"""
This script converts trajectories.csv into Social-LSTM format.

Social-LSTM requires:
- fixed observation window (e.g., 8 frames)
- prediction window (e.g., 12 frames)
- agent-aligned matrices

OUTPUT:
    social_lstm/<scene_id>/obs.npy
    social_lstm/<scene_id>/pred.npy
    social_lstm/<scene_id>/agents.npy

Run example:
    python build_social_lstm.py --input data/processed/trajectories.csv --out data/social_lstm
"""

import numpy as np
import argparse
import pandas as pd
from pathlib import Path

def build_windows(agent_df, obs_len=8, pred_len=12):
    """
    Converts a single agent trajectory into:
    - list of observation sequences
    - list of prediction sequences

    Example:
    frames: [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14]

    obs:  [0-7]
    pred: [8-19]

    Only generate if full window exists.
    """
    frames = agent_df.sort_values("frame")
    xs = frames["x"].values
    ys = frames["y"].values

    obs_seqs = []
    pred_seqs = []

    for i in range(len(xs) - obs_len - pred_len):
        obs = np.stack([xs[i:i+obs_len], ys[i:i+obs_len]], axis=1)
        pred = np.stack([xs[i+obs_len:i+obs_len+pred_len], ys[i+obs_len:i+obs_len+pred_len]], axis=1)
        obs_seqs.append(obs)
        pred_seqs.append(pred)

    return obs_seqs, pred_seqs


def build_social_lstm(input_csv, out_dir):
    df = pd.read_csv(input_csv)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Process per scene
    for scene, scene_df in df.groupby("scene_id"):
        print(f"Processing {scene}...")

        scene_folder = out_dir / scene
        scene_folder.mkdir(exist_ok=True)

        all_obs = []
        all_pred = []

        # Per agent window extraction
        for agent, agent_df in scene_df.groupby("id"):
            obs, pred = build_windows(agent_df)
            all_obs.extend(obs)
            all_pred.extend(pred)

        # Save
        np.save(scene_folder / "obs.npy", np.array(all_obs, dtype=float))
        np.save(scene_folder / "pred.npy", np.array(all_pred, dtype=float))

        # Save agent IDs for reference
        np.save(scene_folder / "agents.npy", scene_df["id"].unique())

    print("Social-LSTM formatting complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    build_social_lstm(args.input, args.out)
