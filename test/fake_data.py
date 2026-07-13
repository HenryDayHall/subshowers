import os
import numpy as np
import pandas as pd


def mock_history_dict_1():
    """Simplist possible history dict"""
    history_dict = {
        "shower": [0],
        "pdg": [0],
        "px": [0],
        "py": [0],
        "pz": [0],
        "x": [0],
        "y": [0],
        "z": [0],
        "kinetic_energy": [0],
        "time": [0],
        "label": [0],
        "parent_label": [-1],
    }
    return history_dict


def mock_history_dict_2():
    """History dict with a small tree of
    parent -> child child -> grandchild
    """
    history_dict = {
        "shower": [0, 0, 0, 0],
        "pdg": [0, 1, 2, 3],
        "px": [0, 0.5, 0, 0],
        "py": [1, 0, 0.5, 0],
        "pz": [2, 0, 0, 0.5],
        "x": [0, 2, 0, 0],
        "y": [0, 1.1, 0, 0],
        "z": [0, 0, 0.1, 0],
        "kinetic_energy": [10, 5, 7, 0.1],
        "time": [0.1, 0, 0.5, 0],
        "label": [0, 1, 2, 3],
        "parent_label": [0, 0, 0, 1],
    }
    return history_dict


COLUMNS = [
    "shower",
    "pdg",
    "px",
    "py",
    "pz",
    "x",
    "y",
    "z",
    "kinetic_energy",
    "time",
    "label",
    "parent_label",
]


def grow_cascade(
    rng, root_energy, root_height, lateral=0.12, min_split_ke=2.0, min_height=1600
):
    rows = {c: [] for c in COLUMNS}

    def add(pos, mom, ke, t, parent_label):
        label = len(rows["label"])
        rows["shower"].append(0)
        rows["pdg"].append(int(rng.choice([22, 11, -11, 13, -13])))
        rows["px"].append(mom[0])
        rows["py"].append(mom[1])
        rows["pz"].append(mom[2])
        rows["x"].append(pos[0])
        rows["y"].append(pos[1])
        rows["z"].append(pos[2])
        rows["kinetic_energy"].append(ke)
        rows["time"].append(t)
        rows["label"].append(label)
        rows["parent_label"].append(parent_label)
        return label

    root_dir = np.array([rng.normal(0, 0.05), rng.normal(0, 0.05), -1.0])
    root_dir /= np.linalg.norm(root_dir)
    root = add(np.array([0.0, 0.0, root_height]), root_dir, root_energy, 0.0, -1)
    # Reader flips the root's parent_label to -1 if absent; -1 given directly.
    stack = [(root, np.array([0.0, 0.0, root_height]), root_dir, root_energy, 0.0)]

    while stack:
        parent, pos, direction, ke, t = stack.pop()
        eps = 1e-6
        if ke < min_split_ke or pos[2] <= min_height + eps:
            continue  # leaf: no children recorded
        # propagate a random fraction of the remaining height, then split
        step = rng.uniform(0.2, 0.5) * (pos[2]-min_height) / max(-direction[2], 0.2)
        new_pos = pos + direction * step
        new_pos[2] = max(new_pos[2], min_height)
        n_children = int(rng.choice([2, 2, 3]))
        fractions = rng.dirichlet(np.ones(n_children)) * rng.uniform(0.85, 0.98)
        for frac in fractions:
            child_dir = direction + np.array(
                [
                    rng.normal(0, lateral),
                    rng.normal(0, lateral),
                    -abs(rng.normal(0, 0.05)),
                ]
            )
            child_dir /= np.linalg.norm(child_dir)
            child_ke = ke * frac
            child = add(new_pos, child_dir, child_ke, t + step / 3e8, parent)
            stack.append((child, new_pos, child_dir, child_ke, t + step / 3e8))
    for c in COLUMNS:
        rows[c] = np.array(rows[c])
    return rows


def fake_history(folder, history_dict):
    hist_dir = os.path.join(folder, "history")
    if not os.path.exists(hist_dir):
        os.makedirs(hist_dir)
    history_path = os.path.join(hist_dir, "history.parquet")
    pd.DataFrame(history_dict).to_parquet(history_path)
