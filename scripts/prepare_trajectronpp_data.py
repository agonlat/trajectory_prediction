import os
import sys
import glob
import dill
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# =========================
# CONFIG
# =========================

DT = 0.4
HISTORY_LENGTH = 8
PREDICTION_HORIZON = 6
MIN_NODE_LENGTH = HISTORY_LENGTH + PREDICTION_HORIZON

# =========================
# PATHS
# =========================

ROOT_DIR = os.getcwd()
TRAJECTRON_PATH = os.path.join(ROOT_DIR, "trajectronpp", "trajectron")
sys.path.insert(0, TRAJECTRON_PATH)

RAW_DATA_DIR = os.path.join(ROOT_DIR, "data", "raw", "Trajectories")
OUT_DIR = os.path.join(ROOT_DIR, "data", "processed_trajectronpp")
os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# TRAJECTRON IMPORTS
# =========================

from environment.scene import Scene
from environment.node import Node
from environment.environment import Environment
from environment.node_type import NodeTypeEnum

# =========================
# NODE TYPES
# =========================

DETECTED_CLASSES_FILE = os.path.join(ROOT_DIR, "detected_classes.txt")
with open(DETECTED_CLASSES_FILE) as f:
    NODE_TYPE_LIST = [l.strip().lower() for l in f if l.strip()]

NodeType = NodeTypeEnum(NODE_TYPE_LIST)

def get_node_type(cls):
    if cls is None or pd.isna(cls):
        return NodeType.car
    return getattr(NodeType, str(cls).lower(), NodeType.car)

# =========================
# HELPERS
# =========================

def compute_derivatives(pos, dt):
    v = np.zeros_like(pos)
    a = np.zeros_like(pos)
    v[1:] = (pos[1:] - pos[:-1]) / dt
    v[0] = v[1]
    a[1:] = (v[1:] - v[:-1]) / dt
    a[0] = a[1]
    return v.astype(np.float32), a.astype(np.float32)

def is_contiguous(frames):
    return np.all(np.diff(frames) == 1)

def scene_has_valid_window(scene):
    for n in scene.nodes:
        if n.last_timestep - n.first_timestep + 1 >= MIN_NODE_LENGTH:
            return True
    return False

# =========================
# CREATE SCENE
# =========================

def create_scene(csv_path):
    df = pd.read_csv(csv_path, engine="python", on_bad_lines="skip")

    if df.empty or not {"frame", "id", "new_x_center", "new_y_center"}.issubset(df.columns):
        return None

    df["frame"] = df["frame"].astype(int)

    min_f = df.frame.min()
    max_f = df.frame.max()

    scene = Scene(
        name=os.path.splitext(os.path.basename(csv_path))[0],
        timesteps=(max_f - min_f + 1),
        dt=DT,
    )

    for tid, track in df.groupby("id"):
        track = track.sort_values("frame")

        if len(track) < MIN_NODE_LENGTH:
            continue

        frames = track["frame"].values
        if not is_contiguous(frames):
            continue

        x = track["new_x_center"].values.astype(np.float32)
        y = track["new_y_center"].values.astype(np.float32)
        pos = np.column_stack((x, y))

        vel, acc = compute_derivatives(pos, DT)

        data = pd.DataFrame(
            np.concatenate([pos, vel, acc], axis=1),
            columns=pd.MultiIndex.from_tuples([
                ("position", "x"), ("position", "y"),
                ("velocity", "x"), ("velocity", "y"),
                ("acceleration", "x"), ("acceleration", "y"),
            ]),
        )

        first_timestep = frames[0] - min_f  # 🔑 critical fix

        node = Node(
            node_type=get_node_type(track["class"].iloc[0] if "class" in track else None),
            node_id=f"{scene.name}_{tid}",
            data=data,
            first_timestep=first_timestep,
        )

        scene.nodes.append(node)

    if not scene.nodes:
        return None

    if not scene_has_valid_window(scene):
        return None

    return scene

# =========================
# ATTENTION RADIUS
# =========================

attention_radius = {
    (a.name, b.name): 30.0
    for a in NodeType
    for b in NodeType
}

# =========================
# STANDARDIZATION
# =========================

standardization = {
    nt: {
        "position": {"x": {"mean": 0, "std": 50}, "y": {"mean": 0, "std": 50}},
        "velocity": {"x": {"mean": 0, "std": 5}, "y": {"mean": 0, "std": 5}},
        "acceleration": {"x": {"mean": 0, "std": 2}, "y": {"mean": 0, "std": 2}},
    }
    for nt in NODE_TYPE_LIST
}

def make_env(scenes):
    env = Environment(
        node_type_list=NODE_TYPE_LIST,
        standardization=standardization,
    )
    env.scenes = scenes
    env.attention_radius = attention_radius
    return env

# =========================
# MAIN
# =========================

files = sorted(glob.glob(os.path.join(RAW_DATA_DIR, "*.csv")))
print(f"Found {len(files)} CSV files")

scenes = []
for f in tqdm(files, desc="Processing scenes"):
    scene = create_scene(f)
    if scene:
        scenes.append(scene)

print(f"Valid scenes: {len(scenes)}")

if len(scenes) < 10:
    print("WARNING: Dataset is very small")

train, tmp = train_test_split(scenes, test_size=0.3, random_state=42)
val, test = train_test_split(tmp, test_size=0.5, random_state=42)

for name, split in [("train", train), ("val", val), ("test", test)]:
    env = make_env(split)
    out = os.path.join(OUT_DIR, f"custom_{name}.pkl")
    with open(out, "wb") as f:
        dill.dump(env, f)
    print(f"{name.upper():5} → {len(split):3} scenes → {out}")
