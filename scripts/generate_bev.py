#!/usr/bin/env python3
"""
generate_bev_full.py

Full BEV / Raster Generator for DESIRE & HiVT with per-scene normalization

Input:
    CSV with columns: scene_id, frame, id, x, y, vx, vy[, class]

Output (per scene):
    <out>/<scene_id>/
        frame_00001.png    -> RGB visualization of agents
        bev_00001.npy      -> (C,H,W) tensor
        metadata.json      -> info (scale, img_size, channels)
        quality_report.json-> missing frames / id jumps

Example:
    python generate_bev_full.py \
      --traj data/processed/trajectories.csv \
      --out data/processed/bev \
      --img-size 256 --world-size 40 \
      --velocity --semantic
"""
from pathlib import Path
import argparse
import json
import logging
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import cv2

# ------------------ Default class colors for visualization ------------------
DEFAULT_CLASS_COLORS = {
    "car": (0, 0, 255),
    "truck": (0, 128, 255),
    "bus": (0, 255, 255),
    "pedestrian": (255, 255, 255),
    "bicycle": (255, 0, 255),
    "other": (200, 200, 200)
}

# ------------------ Utilities ------------------
def init_logger():
    """Initialize logger for console output"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

def safe_mkdir(path: Path):
    """Create directory if it does not exist"""
    path.mkdir(parents=True, exist_ok=True)

# ------------------ Coordinate transforms ------------------
def meters_per_pixel(img_size: int, world_size_m: float) -> float:
    """Compute meters per pixel based on image size and world size"""
    return world_size_m / float(img_size)

def world_to_pixel(
    x: float,
    y: float,
    img_size: int,
    world_size_m: float,
) -> Tuple[int, int]:
    """
    Convert world coordinates (x,y) to pixel coordinates (px,py)
    Image origin (0,0) is at center, y-axis inverted for image coords
    """
    mpp = meters_per_pixel(img_size, world_size_m)
    cx = img_size // 2
    cy = img_size // 2
    px = int(round(cx + (x / mpp)))
    py = int(round(cy - (y / mpp)))
    return px, py

# ------------------ Coordinate normalization ------------------
def normalize_scene_coordinates(df_scene: pd.DataFrame, world_size_m: float) -> pd.DataFrame:
    """
    Normalize x/y coordinates of a single scene to [-world_size/2, world_size/2]
    keeping the center of the scene at (0,0)
    """
    df = df_scene.copy()
    x_min, x_max = df['x'].min(), df['x'].max()
    y_min, y_max = df['y'].min(), df['y'].max()
    x_center = (x_max + x_min) / 2
    y_center = (y_max + y_min) / 2
    scale = max(x_max - x_min, y_max - y_min) / world_size_m
    if scale == 0:
        scale = 1.0  # avoid division by zero
    df['x'] = (df['x'] - x_center) / scale
    df['y'] = (df['y'] - y_center) / scale
    return df

# ------------------ Class inference ------------------
def infer_classes(traj_df: pd.DataFrame, max_classes: int = 32) -> Dict[str,int]:
    """
    Generate mapping from class name to channel index for semantic channels
    """
    if "class" not in traj_df.columns:
        return {"other": 0}
    classes = list(traj_df["class"].fillna("other").astype(str).str.lower().unique())
    if len(classes) > max_classes:
        logging.warning("Too many classes found, truncating to %d.", max_classes)
        classes = classes[:max_classes]
    return {c: i for i, c in enumerate(sorted(classes))}

# ------------------ BEV creation for one frame ------------------
def create_bev_for_frame(
    df_frame: pd.DataFrame,
    img_size: int,
    world_size_m: float,
    use_velocity: bool,
    use_semantic: bool,
    class_to_index: Dict[str,int]
):
    """
    Create BEV raster and visualization for one frame
    Returns:
        viz (H,W,3) uint8 RGB visualization
        bev_tensor (C,H,W) float32
    """
    H = W = img_size
    occupancy = np.zeros((H, W), dtype=np.float32)
    vx_map = np.zeros((H, W), dtype=np.float32) if use_velocity else None
    vy_map = np.zeros((H, W), dtype=np.float32) if use_velocity else None
    n_classes = len(class_to_index) if use_semantic else 0
    class_maps = np.zeros((n_classes, H, W), dtype=np.float32) if use_semantic else None

    viz = np.zeros((H, W, 3), dtype=np.uint8)

    for _, row in df_frame.iterrows():
        try:
            x = float(row["x"])
            y = float(row["y"])
        except Exception:
            continue

        px, py = world_to_pixel(x, y, img_size, world_size_m)
        if not (0 <= px < W and 0 <= py < H):
            continue

        occupancy[py, px] = 1.0

        if use_velocity:
            vx = float(row.get("vx", 0.0) or 0.0)
            vy = float(row.get("vy", 0.0) or 0.0)
            vx_map[py, px] = vx
            vy_map[py, px] = vy

        if use_semantic:
            cls = str(row.get("class", "other")).lower()
            idx = class_to_index.get(cls, class_to_index.get("other", 0))
            class_maps[idx, py, px] = 1.0
            color = DEFAULT_CLASS_COLORS.get(cls, DEFAULT_CLASS_COLORS["other"])
        else:
            color = (255, 255, 255)

        cv2.circle(viz, (px, py), 2, color, -1)

    # stack channels: occupancy first
    channels = [occupancy]
    if use_velocity:
        channels.append(vx_map)
        channels.append(vy_map)
    if use_semantic:
        for i in range(n_classes):
            channels.append(class_maps[i])

    bev_tensor = np.stack(channels, axis=0).astype(np.float32)  # C,H,W
    return viz, bev_tensor

# ------------------ Quality checks ------------------
def check_missing_frames(df_scene: pd.DataFrame) -> List[int]:
    frames = sorted(df_scene["frame"].unique())
    if not frames:
        return []
    expected = set(range(min(frames), max(frames)+1))
    missing = sorted(list(expected - set(frames)))
    return missing

def check_id_jumps(df_scene: pd.DataFrame, threshold: float = 50.0) -> Dict[str, List[dict]]:
    report = {}
    for agent, group in df_scene.groupby("id"):
        group = group.sort_values("frame")
        dx = group["x"].diff().abs().fillna(0)
        dy = group["y"].diff().abs().fillna(0)
        jumps = (dx > threshold) | (dy > threshold)
        if jumps.any():
            report[str(agent)] = group[jumps][["frame","x","y"]].to_dict(orient="records")
    return report

# ------------------ Process scene ------------------
def process_scene(
    scene_id: str,
    df_scene: pd.DataFrame,
    out_root: Path,
    img_size: int,
    world_size_m: float,
    use_velocity: bool,
    use_semantic: bool,
    class_to_index: Dict[str,int],
    save_png: bool,
    save_npy: bool
):
    scene_out = out_root / scene_id
    safe_mkdir(scene_out)

    frames = sorted(df_scene["frame"].unique())
    logging.info("Scene %s: %d frames", scene_id, len(frames))

    for frame in frames:
        df_frame = df_scene[df_scene["frame"]==frame]
        viz, bev = create_bev_for_frame(
            df_frame,
            img_size=img_size,
            world_size_m=world_size_m,
            use_velocity=use_velocity,
            use_semantic=use_semantic,
            class_to_index=class_to_index
        )

        if save_png:
            png_path = scene_out / f"frame_{frame:05d}.png"
            cv2.imwrite(str(png_path), viz)

        if save_npy:
            npy_path = scene_out / f"bev_{frame:05d}.npy"
            np.save(str(npy_path), bev.astype(np.float32))

    # Metadata
    channels = 1
    if use_velocity:
        channels += 2
    if use_semantic:
        channels += len(class_to_index)
    meta = {
        "scene_id": scene_id,
        "img_size": img_size,
        "world_size_m": world_size_m,
        "meters_per_pixel": meters_per_pixel(img_size, world_size_m),
        "channels": channels,
        "channel_description": (
            ["occupancy"] + (["vx","vy"] if use_velocity else []) +
            (["class_"+c for c in sorted(class_to_index.keys())] if use_semantic else [])
        )
    }
    with open(scene_out / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Quality report
    quality = {
        "missing_frames": check_missing_frames(df_scene),
        "id_jumps": check_id_jumps(df_scene)
    }
    with open(scene_out / "quality_report.json", "w") as f:
        json.dump(quality, f, indent=2)

    logging.info("Scene %s processed (output=%s)", scene_id, str(scene_out))

# ------------------ Main ------------------
def main():
    init_logger()
    parser = argparse.ArgumentParser(description="Full BEV / Raster generator")
    parser.add_argument("--traj", required=True, help="path to trajectories.csv")
    parser.add_argument("--out", required=True, help="output directory for BEV")
    parser.add_argument("--img-size", type=int, default=256, help="BEV image size (px, square)")
    parser.add_argument("--world-size", type=float, default=40.0, help="world size in meters covered by BEV")
    parser.add_argument("--velocity", action="store_true", help="include vx, vy channels")
    parser.add_argument("--semantic", action="store_true", help="include semantic one-hot channels")
    parser.add_argument("--save-png", action="store_true", help="save RGB visualization per frame")
    parser.add_argument("--save-npy", action="store_true", help="save BEV numpy tensor per frame")
    parser.add_argument("--min-frames", type=int, default=1, help="skip scenes with fewer frames")
    args = parser.parse_args()

    traj_path = Path(args.traj)
    out_root = Path(args.out)
    safe_mkdir(out_root)

    if not traj_path.exists():
        logging.error("Trajectories file not found: %s", str(traj_path))
        return

    traj_df = pd.read_csv(traj_path)
    required_columns = {"scene_id","frame","id","x","y"}
    if not required_columns.issubset(traj_df.columns):
        logging.error("Trajectories CSV missing required columns. Found: %s", list(traj_df.columns))
        return

    traj_df["frame"] = traj_df["frame"].astype(int)

    if args.semantic and "class" not in traj_df.columns:
        logging.warning("--semantic requested but no 'class' column found. Filling with 'other'.")
        traj_df["class"] = "other"

    class_to_index = infer_classes(traj_df) if args.semantic else {}

    save_png = args.save_png
    save_npy = args.save_npy or True

    for scene_id, df_scene in traj_df.groupby("scene_id"):
        if len(df_scene["frame"].unique()) < args.min_frames:
            logging.info("Skipping scene %s (only %d frames)", scene_id, len(df_scene["frame"].unique()))
            continue
        
        # --- normalize coordinates per scene ---
        df_scene = normalize_scene_coordinates(df_scene, world_size_m=args.world_size)

        process_scene(
            str(scene_id),
            df_scene,
            out_root,
            img_size=args.img_size,
            world_size_m=args.world_size,
            use_velocity=args.velocity,
            use_semantic=args.semantic,
            class_to_index=class_to_index,
            save_png=save_png,
            save_npy=save_npy
        )

    logging.info("All scenes processed.")

if __name__ == "__main__":
    main()
