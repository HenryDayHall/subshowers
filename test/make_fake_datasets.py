# NOTE major functionality impelmented in fake_data
"""
Build small synthetic shower histories in the format expected by
subshowers.showers.Reader: <folder>/history/history.parquet with columns
shower, pdg, px, py, pz, x, y, z, kinetic_energy, time, label, parent_label.

Each folder holds one branching cascade: the root splits repeatedly,
children losing energy and drifting downwards with lateral spread, until
particles reach ground level and become leaves.
"""
import os
import numpy as np
import pandas as pd

COLUMNS = ["shower", "pdg", "px", "py", "pz", "x", "y", "z",
           "kinetic_energy", "time", "label", "parent_label"]


def grow_cascade(rng, root_energy, root_height, lateral=0.12, min_split_ke=2.0):
    rows = {c: [] for c in COLUMNS}

    def add(pos, mom, ke, t, parent_label):
        label = len(rows["label"])
        rows["shower"].append(0)
        rows["pdg"].append(int(rng.choice([22, 11, -11, 13, -13])))
        rows["px"].append(mom[0]); rows["py"].append(mom[1]); rows["pz"].append(mom[2])
        rows["x"].append(pos[0]); rows["y"].append(pos[1]); rows["z"].append(pos[2])
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
        if ke < min_split_ke or pos[2] <= 0:
            continue  # leaf: no children recorded
        # propagate a random fraction of the remaining height, then split
        step = rng.uniform(0.2, 0.5) * pos[2] / max(-direction[2], 0.2)
        new_pos = pos + direction * step
        new_pos[2] = max(new_pos[2], 0.0)
        n_children = int(rng.choice([2, 2, 3]))
        fractions = rng.dirichlet(np.ones(n_children)) * rng.uniform(0.85, 0.98)
        for frac in fractions:
            child_dir = direction + np.array(
                [rng.normal(0, lateral), rng.normal(0, lateral), -abs(rng.normal(0, 0.05))]
            )
            child_dir /= np.linalg.norm(child_dir)
            child_ke = ke * frac
            child = add(new_pos, child_dir, child_ke, t + step / 3e8, parent)
            stack.append((child, new_pos, child_dir, child_ke, t + step / 3e8))
    return pd.DataFrame(rows)


def write_folder(folder, data):
    hist_dir = os.path.join(folder, "history")
    os.makedirs(hist_dir, exist_ok=True)
    data.to_parquet(os.path.join(hist_dir, "history.parquet"))


if __name__ == "__main__":
    base = "/home/claude/data"
    specs = {  # folder -> (seed, root energy [GeV], root height [m])
        "example_photon_hist1": (1, 4.0e4, 12_000.0),
        "example_photon_hist2": (2, 7.0e4, 15_000.0),
    }
    for name, (seed, energy, height) in specs.items():
        rng = np.random.default_rng(seed)
        data = grow_cascade(rng, energy, height)
        folder = os.path.join(base, name)
        write_folder(folder, data)
        print(f"{name}: {len(data)} particles, "
              f"{(data['kinetic_energy'] < 100).sum()} below 100 GeV")
