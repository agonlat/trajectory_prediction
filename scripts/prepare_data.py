import argparse
import os
import json
import cv2
import numpy as np
import pandas as pd
from pathlib import Path


# 1) Trajectory-Loader für OM-Format
def load_om_trajectories(csv_file: Path, scene_id: str):

df = pd.read_csv(csv_file)


#
if "new_x_center" in df.columns and "new_y_center" in df.columns:
    df["x"] = df["new_x_center"]
    df["y"] = df["new_y_center"]
else:
    df["x"] = df["x_center"]
    df["y"] = df["y_center"]


df["scene_id"] = scene_id


# Relevante Spalten
df = df[["scene_id", "frame", "id", "x", "y"]]
return df

def compute_velocities(df):
    df = df.sort_values(["scene_id", "id", "frame"])
    df[["vx","vy"]] = 0.0

    for (scene,agent),group in df.groupby(["scene_id", "id"]):
        dx = group["x"].diff().fillna(0)
        dy = group["y"].diff().fillna(0)

        df.loc[group.index, "vx"] = dx
        df.loc[group.index,"vy"] = dy

        
