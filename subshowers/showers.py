"""
Methods for reading and handling whole showers
"""

import os as _os
import pandas as _pd
import yaml as _yaml


def shower_exists(folder):
    try:
        Reader(folder)
        return True
    except ValueError:
        return False


class Reader:
    """
    Read data created by a modified CORSIKA8 from file
    """

    history_columns = [
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

    def __init__(self, folder: str):
        """
        Parameters
        ----------
        folder : str
            Path to the folder containing the subshowers
        """
        self.folder = folder
        self._history_path = _os.path.join(folder, "history", "history.parquet")
        self._check_expected_format()
        self.data = _pd.read_parquet(self._history_path)
        self._length = len(self.data[self.history_columns[0]])
        if -1 not in self.data["parent_label"].unique():
            root = self.data[self.data["parent_label"] == self.data["label"]].index[0]
            self.data.at[root, "parent_label"] = -1

    def _check_expected_format(self):
        """
        Check that the folder given contains the history file with the expected columns
        """
        has_history_file = _os.path.exists(self._history_path)
        if not has_history_file:
            raise ValueError(f"History file not found at {self._history_path}")
        columns = _pd.read_parquet(self._history_path).columns
        not_found = [column for column in self.history_columns if column not in columns]
        if not_found:
            raise ValueError(
                f"history file at {self._history_path}"
                f" is missing columns: {not_found}"
            )

    def __len__(self) -> int:
        return self._length


class ShowerFolders:
    def __init__(self, configs):
        self.configs = configs
        storage_path = configs["storage_path"]
        showers_folder = configs["showers"]["folder"]
        self._base_path = _os.path.join(storage_path, showers_folder)
        self._folders = [
            _os.path.join(self._base_path, folder)
            for folder in configs["showers"]["subfolders"]
        ]
        self.__length = len(self._folders)

    def __getitem__(self, key):
        return self._folders[key]

    def __len__(self):
        return self.__length

    def __iter__(self):
        for folder in self._folders:
            yield folder

    def __str__(self):
        return f"ShowerFiles[{self.__length}]"

    def __repr__(self):
        return f"ShowerFiles[{self._folders}]"


if __name__ == "__main__":
    conf_path = "/home/dayhallh/eas/subshowers/configs/default.yaml"
    with open(conf_path, "r") as f:
        conf = _yaml.safe_load(f)
    print(conf)
    shower_files = ShowerFolders(conf)
    for folder in shower_files:
        print(folder)

    reader = Reader(shower_files[0])
    print(reader.data)
