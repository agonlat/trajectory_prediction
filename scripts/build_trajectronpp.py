#!/usr/bin/env python3
# scripts/build_trajectronpp.py
"""
Robuster CSV -> Trajectron++-like PKL Converter.

Erwartete CSV-Spalten (mindestens):
    scene_id, frame, agent_id (oder id), x, y [, category]

Output (default):
    data/processed/trajectron_train.pkl
    data/processed/trajectron_val.pkl
    data/processed/trajectron_test.pkl

Optional: --per_scene to erzeugen per-scene/data.pkl Files.
"""
import argparse
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from collections import OrderedDict
import math
import sys

def build_interaction_graph_ordered(agent_positions, threshold):
    """
    Returns adjacency matrix (NxN) and ordered agent list
    agent_positions: dict {agent_id: (x,y)}
    """
    agents = list(agent_positions.keys())
    N = len(agents)
    A = np.zeros((N, N), dtype=np.uint8)
    for i, a in enumerate(agents):
        ax, ay = agent_positions[a]
        for j, b in enumerate(agents):
            if i == j:
                continue
            bx, by = agent_positions[b]
            dist = math.hypot(ax - bx, ay - by)
            if dist <= threshold:
                A[i, j] = 1
    return A, agents

def scene_to_struct(scene_df, distance_threshold=5.0, normalize=True):
    """
    Convert one scene DataFrame into structured dict:
    {
      'frames': [f0,f1,...],
      'agents': {
         agent_id: {'frames':[...], 'x':[...], 'y':[...], 'type': 'pedestrian' or None}
      },
      'graphs': { frame: {'adj': ndarray (NxN), 'agent_list':[id0,id1,...]} }
    }
    distance_threshold in same units as x,y (default 5.0)
    """
    sdf = scene_df.copy()
    sdf = sdf.sort_values(['frame','agent_id'])

    if normalize:
        mean_x = sdf['x'].mean()
        mean_y = sdf['y'].mean()
        sdf['x_norm'] = sdf['x'] - mean_x
        sdf['y_norm'] = sdf['y'] - mean_y
    else:
        sdf['x_norm'] = sdf['x']
        sdf['y_norm'] = sdf['y']

    frames = sorted(sdf['frame'].unique().tolist())

    agents = {}
    for aid, adf in sdf.groupby('agent_id'):
        adf = adf.sort_values('frame')
        agents[int(aid)] = {
            'frames': adf['frame'].tolist(),
            'x': adf['x_norm'].tolist(),
            'y': adf['y_norm'].tolist(),
            'type': adf['category'].iloc[0] if 'category' in adf.columns else None
        }

    graphs = {}
    for frame, fdf in sdf.groupby('frame'):
        # build positions dict for agents **present in this frame**
        pos = OrderedDict()
        for _, row in fdf.sort_values('agent_id').iterrows():
            pos[int(row['agent_id'])] = (row['x_norm'], row['y_norm'])
        if len(pos)==0:
            continue
        adj, agent_list = build_interaction_graph_ordered(pos, threshold=distance_threshold)
        graphs[int(frame)] = {'adj': adj, 'agent_list': agent_list}

    return {
        'frames': frames,
        'agents': agents,
        'graphs': graphs
    }

def split_dict_by_keys(d, keys):
    return {k: d[k] for k in keys}

def build_from_csv(csv_path, out_dir, split=(0.8,0.1,0.1), per_scene=False, threshold=5.0, normalize=True):
    df = pd.read_csv(csv_path)
    # normalize column names
    if 'id' in df.columns and 'agent_id' not in df.columns:
        df = df.rename(columns={'id':'agent_id'})

    required = ['scene_id','frame','agent_id','x','y']
    for r in required:
        if r not in df.columns:
            print(f"ERROR: required column '{r}' not in CSV", file=sys.stderr)
            sys.exit(1)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scenes_struct = {}
    for scene, scene_df in df.groupby('scene_id'):
        scenes_struct[scene] = scene_to_struct(scene_df, distance_threshold=threshold, normalize=normalize)

    if per_scene:
        # save per-scene data.pkl
        for scene_id, sdict in scenes_struct.items():
            scene_folder = out_dir / scene_id
            scene_folder.mkdir(parents=True, exist_ok=True)
            with open(scene_folder / 'data.pkl','wb') as fh:
                pickle.dump(sdict, fh)
        print(f"Wrote {len(scenes_struct)} per-scene files to {out_dir}")
        return

    # else produce train/val/test splits by scene
    scene_ids = sorted(list(scenes_struct.keys()))
    total = len(scene_ids)
    n_train = int(total * split[0])
    n_val = int(total * split[1])
    train_keys = scene_ids[:n_train]
    val_keys = scene_ids[n_train:n_train+n_val]
    test_keys = scene_ids[n_train+n_val:]

    train = {k: scenes_struct[k] for k in train_keys}
    val = {k: scenes_struct[k] for k in val_keys}
    test = {k: scenes_struct[k] for k in test_keys}

    with open(out_dir / 'trajectron_train.pkl','wb') as fh:
        pickle.dump(train, fh)
    with open(out_dir / 'trajectron_val.pkl','wb') as fh:
        pickle.dump(val, fh)
    with open(out_dir / 'trajectron_test.pkl','wb') as fh:
        pickle.dump(test, fh)

    print(f"Wrote train/val/test to {out_dir}: {len(train)}/{len(val)}/{len(test)} scenes")

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True, help='path to trajectories.csv')
    p.add_argument('--out', required=True, help='output folder')
    p.add_argument('--per_scene', action='store_true', help='write per-scene data.pkl files')
    p.add_argument('--threshold', type=float, default=5.0, help='distance threshold for edges')
    p.add_argument('--no_normalize', action='store_true', help='disable centering normalization')
    args = p.parse_args()

    build_from_csv(args.input, args.out, per_scene=args.per_scene, threshold=args.threshold, normalize=not args.no_normalize)

if __name__ == '__main__':
    main()
