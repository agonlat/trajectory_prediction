#!/usr/bin/env python3
"""
INPUT:
- Annotated frames: *.jpg
- Trajectories per video: e.g., "video113.csv"
  Contains columns like:
      frame, class, id, x_center, y_center, x, y, w, h, new_x_center, new_y_center

OUTPUT:
- data/processed/trajectories.csv
      scene_id, frame, id, x, y, vx, vy
- data/processed/frames/<scene_id>/frame_000123.png
- data/processed/quality_report.json

CLI example:
    python prepare_data.py --input data/raw --out data/processed --format csv
"""

import argparse
import os
import json
import cv2
import numpy as np
import pandas as pd
from pathlib import Path


# 1) Trajectory loader for OM format

def load_om_trajectories(csv_file: Path, scene_id: str):
    """
    Expected CSV format (example):
    frame, class, id, x_center, y_center, x, y, w, h, new_x_center, new_y_center

    We use new_x_center / new_y_center as the cleaned coordinates.
    If they do not exist, fallback to x_center/y_center.
    """
    df = pd.read_csv(csv_file)

    # Check if new_x_center exists, otherwise fallback
    if "new_x_center" in df.columns and "new_y_center" in df.columns:
        df["x"] = df["new_x_center"]
        df["y"] = df["new_y_center"]
    else:
        df["x"] = df["x_center"]
        df["y"] = df["y_center"]

    df["scene_id"] = scene_id

    # Relevant columns
    df = df[["scene_id", "frame", "id", "x", "y"]]
    return df


# 2) Compute velocities

def compute_velocities(df):
    df = df.sort_values(["scene_id", "id", "frame"])
    df[["vx", "vy"]] = 0.0

    for (scene, agent), group in df.groupby(["scene_id", "id"]):
        group = group.sort_values("frame")
        dx = group["x"].diff().fillna(0)
        dy = group["y"].diff().fillna(0)
        df.loc[group.index, "vx"] = dx
        df.loc[group.index, "vy"] = dy

    return df


# 3) Extract frames

def extract_frames_from_jpg(folder: Path, out_dir: Path):
    """
    OM dataset: Photos already extracted → just copy/normalize.
    Expected: frame_00001.jpg or numbered jpgs.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(list(folder.glob("*.jpg")))
    for img in images:
        # Extract frame number
        # Assumes: something_01234.jpg → number from the last numeric block
        stem = img.stem
        num = ''.join([c for c in stem if c.isdigit()])
        if num == "":
            continue
        frame_idx = int(num)
        out_path = out_dir / f"frame_{frame_idx:05d}.png"
        frame = cv2.imread(str(img))
        if frame is not None:
            cv2.imwrite(str(out_path), frame)


# 4) Quality checks

def check_missing_frames(df):
    report = {}
    for scene, group in df.groupby("scene_id"):
        frames = sorted(group["frame"].unique())
        missing = []
        if len(frames) > 0:
            expected = set(range(min(frames), max(frames) + 1))
            missing = sorted(list(expected - set(frames)))
        report[scene] = missing
    return report


def check_nans(df):
    return df.isna().sum().to_dict()


def check_id_jumps(df, threshold=10.0):
    report = {}
    for (scene, agent), group in df.groupby(["scene_id", "id"]):
        group = group.sort_values("frame")
        dx = group["x"].diff().abs().fillna(0)
        dy = group["y"].diff().abs().fillna(0)
        jumps = (dx > threshold) | (dy > threshold)
        if jumps.any():
            report[f"{scene}:{agent}"] = group[jumps][["frame", "x", "y"]].to_dict(orient="records")
    return report


# 5) Main function

def run(input_dir, out_dir, fmt):
    input_dir = Path(input_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # A) Load trajectories
    print("Loading OM trajectories...")
    all_dfs = []

    for csv_file in sorted(input_dir.rglob("*.csv")):
        scene_id = csv_file.stem.replace("video", "scene_")
        df_scene = load_om_trajectories(csv_file, scene_id)
        all_dfs.append(df_scene)

    if len(all_dfs) == 0:
        print("WARNING: No CSV trajectories found!")
        return

    traj_df = pd.concat(all_dfs, ignore_index=True)

    # Compute velocities
    traj_df = compute_velocities(traj_df)

    # Save
    out_csv = out_dir / "trajectories.csv"
    traj_df.to_csv(out_csv, index=False)
    print(f"Trajectories saved: {out_csv}")

    # B) Copy annotated frames
    print("Copying frames...")
    for scene_dir in sorted(input_dir.glob("scene_*")):
        jpg_folder = scene_dir / "frames"
        if jpg_folder.exists():
            out_frames = out_dir / "frames" / scene_dir.name
            extract_frames_from_jpg(jpg_folder, out_frames)

    # C) Quality report
    print("Creating quality report...")
    report = {
        "missing_frames": check_missing_frames(traj_df),
        "nans": check_nans(traj_df),
        "id_jumps": check_id_jumps(traj_df)
    }

    with open(out_dir / "quality_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("Done.")


# 6) CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--format", default="csv")
    args = parser.parse_args()

    run(args.input, args.out, args.format)
