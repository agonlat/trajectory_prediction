import sys
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# -----------------------------
# 1. Pfade setzen
# -----------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
TRAJECTRON_PATH = os.path.join(ROOT_DIR, 'trajectronpp', 'trajectron')

if not os.path.exists(TRAJECTRON_PATH):
    print(f"FATAL: Pfad {TRAJECTRON_PATH} existiert nicht.")
    sys.exit(1)

sys.path.insert(0, TRAJECTRON_PATH)
print(f"DEBUG: TRAJECTRON_PATH = {TRAJECTRON_PATH}")

# -----------------------------
# 2. Imports
# -----------------------------
try:
    from environment.scene import Scene
    from environment.node import Node
    from environment.node_type import NodeTypeEnum as NodeType
    from model.dynamics import SingleIntegrator
    print("DEBUG: Imports erfolgreich")
except ImportError as e:
    print(f"FATALER FEHLER BEI DEN IMPORTS: {e}")
    sys.exit(1)

# -----------------------------
# 3. Konfiguration
# -----------------------------
RAW_DATA_DIR = os.path.join(ROOT_DIR, 'data', 'raw', 'Trajectories')
PROCESSED_DATA_DIR = os.path.join(ROOT_DIR, 'data', 'processed_trajectronpp')
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# -----------------------------
# 4. Szenen erstellen
# -----------------------------
def create_scenes_from_video_data(df: pd.DataFrame, video_id: str, node_type_map: dict):
    min_frame = df['frame'].min()
    max_frame = df['frame'].max()
    
    DT = 0.4
    scene = Scene(timesteps=max_frame - min_frame + 1, dt=DT, name=video_id, map=None)

    for track_id, track_df in df.groupby('id'):
        track_df = track_df.sort_values('frame')
        data_columns = ['x_center', 'y_center']

        # Padding
        full_frame_range = np.arange(min_frame, max_frame + 1)
        full_track_df = pd.DataFrame({'frame': full_frame_range})
        track_df = full_track_df.merge(track_df, on='frame', how='left')

        trajectory = track_df[data_columns].values.astype('float32')
        agent_type_str = track_df['class'].dropna().iloc[0].lower() if 'class' in track_df.columns else 'car'

        if agent_type_str in node_type_map:
            node_type = node_type_map[agent_type_str]
            # Node nur mit node_type und node_id
            node = Node(node_type=node_type, node_id=f"{video_id}_{track_id}")
            # Trajektorie zuweisen
            node.set_data_array(node_data_array=trajectory, frequency_multiplier=1)
            scene.add_node(node)

    return scene if len(scene.nodes) > 0 else None

# -----------------------------
# 5. Main
# -----------------------------
if __name__ == '__main__':
    # NodeTypes definieren
    CAR = NodeType("car")
    TRUCK = NodeType("truck")
    BUS = NodeType("bus")
    PERSON = NodeType("person")
    BICYCLE = NodeType("bicycle")
    MOTORCYCLE = NodeType("motorcycle")

    NODE_TYPE_MAP = {
        'car': CAR,
        'truck': TRUCK,
        'bus': BUS,
        'person': PERSON,
        'bicycle': BICYCLE,
        'motorcycle': MOTORCYCLE,
    }

    all_scenes = []

    if not os.path.exists(RAW_DATA_DIR):
        print(f"Fehler: Rohdatenverzeichnis nicht gefunden unter {RAW_DATA_DIR}.")
        sys.exit(1)

    for file_name in tqdm(os.listdir(RAW_DATA_DIR)):
        if file_name.endswith('.csv'):
            video_id = os.path.splitext(file_name)[0]
            try:
                df = pd.read_csv(os.path.join(RAW_DATA_DIR, file_name))
                scene = create_scenes_from_video_data(df, video_id, NODE_TYPE_MAP)
                if scene:
                    all_scenes.append(scene)
            except Exception as e:
                print(f"Fehler bei {file_name}: {e}")

    if not all_scenes:
        print("Keine Szenen erstellt. Beende Skript.")
        sys.exit(1)

    print(f"\nErfolgreich {len(all_scenes)} Szenen erstellt.")

    # Split
    train_scenes, temp_scenes = train_test_split(all_scenes, test_size=0.3, random_state=42)
    val_scenes, test_scenes = train_test_split(temp_scenes, test_size=0.5, random_state=42)

    data_splits = {'train': train_scenes, 'val': val_scenes, 'test': test_scenes}

    # Speichern
    for split_name, scene_list in data_splits.items():
        data_dict = {k: scene_list for k in NODE_TYPE_MAP.keys()}
        output_path = os.path.join(PROCESSED_DATA_DIR, f'custom_{split_name}.pkl')
        print(f"Speichere {len(scene_list)} Szenen in {output_path}")
        torch.save(data_dict, output_path)
