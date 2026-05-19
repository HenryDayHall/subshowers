import os
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


def fake_history(folder, history_dict):
    hist_dir = os.path.join(folder, "history")
    if not os.path.exists(hist_dir):
        os.mkdir(hist_dir)
    history_path = os.path.join(hist_dir, "history.parquet")
    pd.DataFrame(history_dict).to_parquet(history_path)
