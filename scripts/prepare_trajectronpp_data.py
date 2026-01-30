import os
import sys
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import dill

# =============================
# CONFIG (HIER MUSS ES STIMMEN)
# =============================
DT = 0.4
HISTORY_LENGTH = 8
PREDICTION_HORIZON = 6
MIN_NODE_LENGTH = HISTORY_LENGTH + PREDICTION_HORIZON

# =============================
# PATHS
# =============================
ROOT_DIR = os.getcwd()
TRAJECTRON_PATH = os.path.join(ROOT_DIR, 'trajectronpp', 'trajectron')
sys.path.insert(0, TRAJECTRON_PATH)

RAW_DATA_DIR = os.path.join(ROOT_DIR, 'data', 'raw', 'Trajectories')
PROCESSED_DATA_DIR = os.path.join(ROOT_DIR, 'data', 'processed_trajectronpp')
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# =============================
# LOAD CLASSES
# =============================
DETECTED_CLASSES_FILE = os.path.join(ROOT_DIR, 'detected_classes.txt')
with open(DETECTED_CLASSES_FILE, 'r') as f:
    NODE_TYPE_LIST = [l.strip().lower() for l in f if l.strip()]

print(f"Detected classes: {NODE_TYPE_LIST}")

# =============================
# TRAJECTRON IMPORTS
# =============================
from environment.node_type import NodeTypeEnum
from environment.scene import Scene
from environment.node import Node
from environment.environment import Environment

NodeType = NodeTypeEnum(NODE_TYPE_LIST)

# =============================
# HELPERS
# =============================
def compute_derivatives(pos, dt):
    v = np.zeros_like(pos)
    a = np.zeros_like(pos)
    v[1:] = (pos[1:] - pos[:-1]) / dt
    v[0] = v[1]
    a[1:] = (v[1:] - v[:-1]) / dt
    a[0] = a[1]
    return v.astype(np.float32), a.astype(np.float32)


def get_node_type(cls):
    if cls is None or pd.isna(cls):
        return NodeType.car
    cls = str(cls).lower()
    return getattr(NodeType, cls, NodeType.car)


# =============================
# CREATE SCENE
# =============================
def create_scene(csv_path, scene_name):
    df = pd.read_csv(
    csv_path,
    engine="python",      
    on_bad_lines="skip"   
)

    if df.empty or 'frame' not in df or 'id' not in df:
        return None

    df['frame'] = df['frame'].astype(int)
    min_f, max_f = df.frame.min(), df.frame.max()

    scene = Scene(
        timesteps=max_f - min_f + 1,
        dt=DT,
        name=scene_name
    )

    for tid, track in df.groupby('id'):
        track = track.sort_values('frame')

        if len(track) < MIN_NODE_LENGTH:
            continue

        if not {'new_x_center', 'new_y_center'}.issubset(track.columns):
            continue

        x = track['new_x_center'].values.astype(np.float32)
        y = track['new_y_center'].values.astype(np.float32)
        pos = np.column_stack((x, y))

        vel, acc = compute_derivatives(pos, DT)

        data = pd.DataFrame(
            np.concatenate([pos, vel, acc], axis=1),
            columns=pd.MultiIndex.from_tuples([
                ('position', 'x'), ('position', 'y'),
                ('velocity', 'x'), ('velocity', 'y'),
                ('acceleration', 'x'), ('acceleration', 'y')
            ])
        )

        if len(data) < MIN_NODE_LENGTH:
            continue

        cls = track['class'].iloc[0] if 'class' in track else None
        node = Node(
            node_type=get_node_type(cls),
            node_id=f"{scene_name}_{tid}",
            data=data,
            first_timestep=track['frame'].iloc[0]
        )

        scene.nodes.append(node)

    if len(scene.nodes) == 0:
        return None

    return scene


# =============================
# ATTENTION RADIUS
# =============================
attention_radius = {}
for a in NodeType:
    for b in NodeType:
        attention_radius[(a.name, b.name)] = 30.0


# =============================
# STANDARDIZATION
# =============================
standardization = {
    nt: {
        'position': {'x': {'mean': 0.0, 'std': 50.0},
                     'y': {'mean': 0.0, 'std': 50.0}},
        'velocity': {'x': {'mean': 0.0, 'std': 5.0},
                     'y': {'mean': 0.0, 'std': 5.0}},
        'acceleration': {'x': {'mean': 0.0, 'std': 2.0},
                         'y': {'mean': 0.0, 'std': 2.0}}
    } for nt in NODE_TYPE_LIST
}


def make_env(scenes):
    env = Environment(
        node_type_list=NODE_TYPE_LIST,
        standardization=standardization
    )
    env.scenes = scenes
    env.attention_radius = attention_radius
    return env


# =============================
# PROCESS ALL FILES
# =============================
files = sorted(glob.glob(os.path.join(RAW_DATA_DIR, "*.csv")))
print(f"Found {len(files)} CSV files")

scenes = []
for fp in tqdm(files, desc="Processing scenes"):
    name = os.path.splitext(os.path.basename(fp))[0]
    scene = create_scene(fp, name)
    if scene:
        scenes.append(scene)

print(f"Created {len(scenes)} valid scenes")

if len(scenes) < 10:
    print("⚠️ WARNING: Dataset is very small – evaluation may be unstable")

# =============================
# SPLIT & SAVE
# =============================
train, tmp = train_test_split(scenes, test_size=0.3, random_state=42)
val, test = train_test_split(tmp, test_size=0.5, random_state=42)

for name, split in [('train', train), ('val', val), ('test', test)]:
    env = make_env(split)
    out = os.path.join(PROCESSED_DATA_DIR, f"custom_{name}.pkl")
    with open(out, 'wb') as f:
        dill.dump(env, f)
    print(f"{name.upper():5} → {len(split):3} scenes → {out}")

