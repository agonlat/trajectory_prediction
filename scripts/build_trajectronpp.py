# =============================================================
# build_trajectronpp.py
# =============================================================
"""
This script converts the unified trajectories.csv into a Trajectron++-compatible format.

OUTPUT per scene:
    trajectronpp/<scene_id>/data.pkl

Contains:
- normalized trajectories
- agent types (if available)
- adjacency matrices (interaction graph)

Run example:
    python build_trajectronpp.py --input data/processed/trajectories.csv --out data/trajectronpp
"""

import argparse
import pandas as pd
import numpy as np
import os
import pickle
from pathlib import Path

# -------------------------------------------------------------
# Helper: Build interaction graph
# -------------------------------------------------------------
def build_interaction_graph(agent_positions, threshold=50.0):
    """
    Creates adjacency matrix based on distance.

    agent_positions: dict {agent_id: (x,y)}
    threshold: max distance to connect two agents

    Output: NxN adjacency matrix
    """
    agents = list(agent_positions.keys())
    N = len(agents)
    A = np.zeros((N, N))

    for i, a in enumerate(agents):
        for j, b in enumerate(agents):
            if i == j:
                continue
            ax, ay = agent_positions[a]
            bx, by = agent_positions[b]
            dist = np.linalg.norm([ax - bx, ay - by])
            if dist < threshold:
                A[i,j] = 1
    return A

# -------------------------------------------------------------
# Main
# -------------------------------------------------------------
def build_trajectron(input_csv, out_dir):
    df = pd.read_csv(input_csv)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Group by scene
    for scene, scene_df in df.groupby("scene_id"):
        print(f"Processing {scene}...")

        scene_folder = out_dir / scene
        scene_folder.mkdir(exist_ok=True)

        # Normalize coordinates (centering)
        mean_x = scene_df["x"].mean()
        mean_y = scene_df["y"].mean()
        scene_df["x_norm"] = scene_df["x"] - mean_x
        scene_df["y_norm"] = scene_df["y"] - mean_y

        # Extract per-frame interaction graphs
        graphs = {}
        for frame, frame_df in scene_df.groupby("frame"):
            agent_positions = dict(zip(frame_df["id"], zip(frame_df["x_norm"], frame_df["y_norm"])))
            graphs[int(frame)] = build_interaction_graph(agent_positions)

        # Save pickle
        data = {
            "trajectories": scene_df,
            "graphs": graphs,
        }

        with open(scene_folder / "data.pkl", "wb") as f:
            pickle.dump(data, f)

    print("Trajectron++ formatting complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    build_trajectron(args.input, args.out)