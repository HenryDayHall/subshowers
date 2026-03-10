"""
TODO currently, this assumes that all folders contain a single shower.
May need to alter that.
"""

import os as _os
import warnings as _warnings
import pandas as _pd


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


def energy_cut(energy: float):
    """
    Simple start_condition, that begins a subshower when the kinetic energy
    is below the given value.

    Parameters
    ----------
    energy : float
        Energy in GeV
    """
    return lambda **kwargs: kwargs["kinetic_energy"] < energy


class ShowerStarts:
    """
    Locations that subshowers start
    """

    def __init__(
        self, reader: Reader, start_condition: callable, verbose: bool = False
    ):
        """
        Parameters
        ----------
        data : Reader
            Reader object to read data from
        start_condition : callable
            Function to determine when a subshower starts.
            Must take the columns specified in Reader.history_columns
            as keyname arguments and return a bool.
        verbose : bool
            Print progress
        """
        self.reader = reader
        self.start_condition = start_condition
        self.verbose = verbose
        self._identify_root_starts()
        self._sort_starts()

    def _save_dict(self) -> dict:
        return {
            "folder": self.data.folder,
            "starts": self._starts,
            "loose_ends": self._loose_ends,
            "root": self._root,
        }

    @classmethod
    def _load_dict(cls, dictionary: dict, start_condition: callable = None):
        reader = Reader(dictionary["folder"])
        new = cls(reader, start_condition)
        new._starts = dictionary["starts"]
        new._loose_ends = dictionary["loose_ends"]
        new._root = dictionary["root"]
        new.start_condition = start_condition
        return new

    def _identify_root_starts(self):
        """
        Find the root particle, the start of each subshower,
        and the leaf particles with no valid subshower.
        """
        data = self.reader.data
        root = data.index[data["parent_label"] == -1][0]
        self._root = data["label"][root]
        self._starts = []
        self._loose_ends = []
        stack = [root]
        i = 0
        while stack:
            i += 1
            if i % 1000 == 0 and self.verbose:
                fraction = i / len(self.data)
                print(
                    f"Processed {fraction:.1%} of total particles."
                    f" Found {len(self._starts)} starts.",
                    end="\r",
                )

            parent = stack.pop()
            child_indices = data.index[data["parent_label"] == data.at[parent, "label"]]
            child_values = data.loc[child_indices]
            if child_values.empty:
                self._loose_ends.append(parent)
                continue
            is_start = self.start_condition(**child_values)
            self._starts += list(child_indices[is_start])
            stack += list(child_indices[~is_start])
        if self.verbose:
            fraction = i / len(self.data)
            print(
                f"Found all {len(self._starts)} starts after "
                f"processing {fraction:.1%} of total particles"
            )
        self._starts = _pd.Index(self._starts)
        self._loose_ends = _pd.Index(self._loose_ends)
        if self._loose_ends.any():
            _warnings.warn(
                f"Found {len(self._loose_ends)} loose ends,"
                f"with {len(self._starts)} starts.",
                UserWarning,
            )
        if root in self._starts:
            _warnings.warn(
                "The root particle is also a start particle,"
                " therefore the subshower may not be representative.",
                UserWarning,
            )

    def _sort_starts(self):
        kinetic_energy = self.reader.at[self._starts, "kinetic_energy"]
        self._starts = self._starts.sort_values(kinetic_energy)

    def __len__(self) -> int:
        """
        Number of subshowers
        """
        return len(self._starts)

    def __iter__(self) -> _pd.Index:
        for start in self._starts:
            yield start

    def __getitem__(self, item) -> _pd.Index:
        return self._starts[item]

    def get_loose_ends(self, index=None) -> _pd.Index:
        """
        Return the data of leaf particles with no valid subshower
        """
        with self.reader as data:
            if index is None:
                return data.loc[self._loose_ends]
            return self._loose_ends[index]


def get_subshower(data, start_index: int, leaves_only: bool = True) -> list:
    """
    Get data for particles in a subshower, either all particles,
    or just leaf particles.
    """
    subshower = []
    stack = [start_index]
    while stack:
        parent = stack.pop()
        child_indices = data.index[data["parent_label"] == data.at[parent, "label"]]
        if leaves_only and child_indices.empty:
            subshower.append(parent)
        else:
            subshower.append(parent)
        stack += list(child_indices)
    return subshower


class Subshowers:
    def __init__(self, reader: Reader, leaves_only: bool = True, verbose: bool = False):
        self.showerstarts = []
        self.subshowers = []
        self.reader = reader
        self.leaves_only = leaves_only
        self.verbose = verbose

    def __len__(self):
        return len(self.showerstarts)

    def add_subshower(self, start_index: list | _pd.Index):
        for start in start_index:
            if start in self.showerstarts:
                continue
            subshower = get_subshower(self.reader.data, start, self.leaves_only)
            self.subshowers.append(subshower)
            self.showerstarts.append(start)

    def __getitem__(self, item):
        return self.showerstarts[item], self.subshowers[item]

    def __iter__(self):
        for start, subshower in zip(self.showerstarts, self.subshowers):
            yield start, subshower

    def _save_dict(self) -> dict:
        flat_starts = [
            [start] * len(subshower)
            for start, subshower in zip(self.showerstarts, self.subshowers)
        ]
        flat_subshowers = sum(self.subshowers, [])
        return {
            "folder": self.reader.folder,
            "starts": flat_starts,
            "subshowers": flat_subshowers,
            "leaves_only": self.leaves_only,
        }

    @classmethod
    def _load_dict(dictionary: dict):
        reader = Reader(dictionary["folder"])
        new = Subshowers(reader, dictionary["leaves_only"])
        flat_starts = dictionary["starts"]
        flat_subshowers = dictionary["subshowers"]
        new.showerstarts = sorted(set(flat_starts))
        new.subshowers = [[] for _ in new.showerstarts]
        for start, subshower in zip(flat_starts, flat_subshowers):
            new.subshowers[new.showerstarts.index(start)].append(subshower)
        return new
