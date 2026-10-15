"""
TODO currently, this assumes that all folders contain a single shower.
May need to alter that.
"""

import sys as _sys
import os as _os
import warnings as _warnings
import numpy as _np
import pandas as _pd

from subshowers.showers import Reader as _Reader, HistoryTree as _HistoryTree
from subshowers import raw_save_load as _raw


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


def _history_tree(reader) -> _HistoryTree:
    """The reader's HistoryTree, or a new one for readers that don't keep one."""
    tree = getattr(reader, "tree", None)
    return tree if tree is not None else _HistoryTree(reader.data)


def _evaluate_condition(start_condition: callable, data: _pd.DataFrame):
    """start_condition for every particle at once, as an array of bools."""
    result = _np.asarray(start_condition(**data), dtype=bool)
    if result.shape != (len(data),):
        # a condition that doesn't depend on the particles
        result = _np.broadcast_to(result, (len(data),))
    return result


class ShowerStarts:
    """
    Locations that subshowers start
    """

    def __init__(
        self, reader: _Reader, start_condition: callable, verbose: bool = False
    ):
        """
        Parameters
        ----------
        reader : Reader
            Reader object to read data from
        start_condition : callable
            Function to determine when a subshower starts.
            Must take the columns specified in Reader.history_columns
            as keyname arguments and return a bool.
            It's called once with whole columns, so it must work element
            by element, returning one bool per particle (as energy_cut does).
        verbose : bool
            Print progress
        """
        self.reader = reader
        self.start_condition = start_condition
        self.verbose = verbose
        if start_condition is not None:
            self._identify_root_starts()
            self._sort_starts()

    def _save_dict(self) -> dict:
        return {
            "folder": self.reader.folder,
            "starts": self._starts,
            "loose_ends": self._loose_ends,
            "root": int(self._root),
        }

    @classmethod
    def _load_dict(cls, dictionary: dict, start_condition: callable = None):
        reader = _Reader(dictionary["folder"])
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

        Walking down from the root, the first particle on each branch that
        meets the start condition starts a subshower, and nothing below it
        is considered for another start. Leaves reached without meeting the
        condition are loose ends.
        """
        data = self.reader.data
        tree = _history_tree(self.reader)
        if tree.root is None:
            raise ValueError("The history has no root particle (parent_label -1)")
        root = tree.root
        self._root = data["label"].iloc[root]
        is_start = _evaluate_condition(self.start_condition, data)
        if is_start[root]:
            self._starts = tree.index[[root]]
            self._loose_ends = _pd.Index([])
            _warnings.warn(
                "The root particle is also a start particle,"
                " therefore the subshower may not be representative.",
                UserWarning,
            )
            return
        in_shower = _np.sort(tree.subtree(root))
        owner = tree.topmost_marked(is_start)[in_shower]
        starts = in_shower[owner == in_shower]
        loose_ends = in_shower[(owner < 0) & tree.is_leaf[in_shower]]
        self._starts = _pd.Index(tree.labels_of(starts))
        self._loose_ends = _pd.Index(tree.labels_of(loose_ends))
        if self.verbose:
            print(
                f"Found {len(self._starts)} starts and {len(self._loose_ends)}"
                f" loose ends among {len(in_shower)} particles"
            )
        if len(self._loose_ends):
            _warnings.warn(
                f"Found {len(self._loose_ends)} loose ends,"
                f" with {len(self._starts)} starts.",
                UserWarning,
            )

    def _sort_starts(self):
        kinetic_energy = self.reader.data["kinetic_energy"][self._starts]
        self._starts = self._starts[kinetic_energy.argsort()]

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
        return self._loose_ends


def get_subshower(
    data, start_index: int, leaves_only: bool = True, tree: _HistoryTree = None
) -> _np.ndarray:
    """
    Get data for particles in a subshower, either all particles,
    or just leaf particles, as index labels of data.

    Building the HistoryTree takes a pass over the whole history, so when
    collecting many subshowers, build it once and pass it as tree.
    """
    if tree is None:
        tree = _HistoryTree(data)
    position = tree.positions_of([start_index])[0]
    return tree.labels_of(tree.subtree(position, leaves_only))


class Subshowers:
    """
    Locate and retain particles in subshowers given a starting particle.
    """

    def __init__(
        self, reader: _Reader, leaves_only: bool = True, verbose: bool = False
    ):
        self.showerstarts = []
        self.subshowers = []
        self.reader = reader
        self.leaves_only = leaves_only
        self.verbose = verbose

    def __len__(self):
        return len(self.showerstarts)

    def add_subshower(self, start_index: list | _pd.Index):
        """
        Add the subshower below each of these start particles (index labels
        of the history), skipping starts already added.
        """
        known = set(self.showerstarts)
        new_starts = []
        for start in start_index:
            if start not in known:
                known.add(start)
                new_starts.append(start)
        if not new_starts:
            return
        tree = _history_tree(self.reader)
        for start, position in zip(new_starts, tree.positions_of(new_starts)):
            members = tree.subtree(position, self.leaves_only)
            self.subshowers.append(tree.labels_of(members))
            self.showerstarts.append(start)
        if self.verbose:
            n_particles = sum(len(subshower) for subshower in self.subshowers)
            print(
                f"Collected {len(new_starts)} subshowers;"
                f" {len(self)} hold {n_particles} particles in all"
            )

    def __getitem__(self, item):
        return self.showerstarts[item], self.subshowers[item]

    def __iter__(self):
        for start, subshower in zip(self.showerstarts, self.subshowers):
            yield start, subshower

    def _save_dict(self) -> dict:
        """Starts and particles as two flat arrays, one row per particle."""
        lengths = [len(subshower) for subshower in self.subshowers]
        flat_starts = _np.repeat(
            _np.asarray(self.showerstarts, dtype=_np.int64), lengths
        )
        if self.subshowers:
            flat_subshowers = _np.concatenate(self.subshowers).astype(_np.int64)
        else:
            flat_subshowers = _np.zeros(0, dtype=_np.int64)
        return {
            "folder": self.reader.folder,
            "starts": flat_starts,
            "subshowers": flat_subshowers,
            "leaves_only": self.leaves_only,
        }

    @classmethod
    def _load_dict(cls, dictionary: dict, reader: _Reader = None):
        """
        Rebuild from _save_dict's output, with starts in increasing order.
        reader, if given, is used rather than reading the folder's history.
        """
        if reader is None:
            reader = _Reader(dictionary["folder"])
        new = Subshowers(reader, dictionary["leaves_only"])
        flat_starts = _np.asarray(dictionary["starts"], dtype=_np.int64)
        flat_subshowers = _np.asarray(dictionary["subshowers"], dtype=_np.int64)
        if len(flat_starts) == 0:
            return new
        order = _np.argsort(flat_starts, kind="stable")
        starts, first_rows = _np.unique(flat_starts[order], return_index=True)
        new.showerstarts = starts.tolist()
        new.subshowers = _np.split(flat_subshowers[order], first_rows[1:])
        return new


class Output:
    """
    Read and write both shower starts and subshowers, along with metadata
    """

    SHOWER_STARTS_PREFIX = "starts"
    SUBSHOWERS_PREFIX = "subshowers"
    METADATA_PREFIX = "metadata"

    def __init__(self, shower_starts: ShowerStarts, subshowers: Subshowers, **metadata):
        self.shower_starts = shower_starts
        self.subshowers = subshowers
        self.metadata = metadata

    @staticmethod
    def basepath(path):
        parts = path.split("/")
        return "/".join(parts[:-2])

    def _prep_dataframes(self):
        common_dict = {}
        shower_starts_dict = self.shower_starts._save_dict()
        for key, value in shower_starts_dict.items():
            prefixed_key = f"{self.SHOWER_STARTS_PREFIX}_{key}"
            common_dict[prefixed_key] = value
        subshowers_dict = self.subshowers._save_dict()
        for key in subshowers_dict:
            prefixed_key = f"{self.SUBSHOWERS_PREFIX}_{key}"
            common_dict[prefixed_key] = subshowers_dict[key]
        for key, value in self.metadata.items():
            common_dict[f"{self.METADATA_PREFIX}_{key}"] = value
        return common_dict

    @classmethod
    def _unpack_dataframes(cls, loaded: dict):
        shower_starts_dict = {}
        subshowers_dict = {}
        metadata = {}
        for key in loaded:
            value = loaded[key]
            if key.startswith(cls.SHOWER_STARTS_PREFIX):
                tail = key[len(cls.SHOWER_STARTS_PREFIX) + 1 :]
                shower_starts_dict[tail] = value
            elif key.startswith(cls.SUBSHOWERS_PREFIX):
                tail = key[len(cls.SUBSHOWERS_PREFIX) + 1 :]
                subshowers_dict[tail] = value
            elif key.startswith(cls.METADATA_PREFIX):
                tail = key[len(cls.METADATA_PREFIX) + 1 :]
                metadata[tail] = value
            else:
                print(f"Unknown key: {key}")

        shower_starts = ShowerStarts._load_dict(shower_starts_dict)
        # both normally come from the same history, so only read it once
        reader = shower_starts.reader
        same_folder = _os.path.normpath(subshowers_dict["folder"]) == _os.path.normpath(
            reader.folder
        )
        subshowers = Subshowers._load_dict(
            subshowers_dict, reader=reader if same_folder else None
        )
        new = cls(shower_starts, subshowers, **metadata)
        return new

    def save(self, path: str):
        common_dict = self._prep_dataframes()
        base_path = self.basepath(path)
        for key, value in common_dict.items():
            if key.endswith("folder"):
                common_dict[key] = _os.path.relpath(value, base_path)
        _raw.save(path, **common_dict)

    @classmethod
    def load(cls, path: str):
        loaded = _raw.load(path)
        base_path = cls.basepath(path)
        for key in loaded:
            if key.endswith("folder"):
                loaded[key] = _os.path.join(base_path, loaded[key])
            elif isinstance(loaded[key], _pd.DataFrame):
                loaded[key] = loaded[key].values
                if len(loaded[key].shape) == 2 and loaded[key].shape[1] == 1:
                    loaded[key] = loaded[key][:, 0]

        return cls._unpack_dataframes(loaded)


def subshower_exists(folder, energy_cut_value):
    output_path = _os.path.join(folder, f"subshowers_e{energy_cut_value}.h5")
    return _os.path.exists(output_path)


def run(
    folder: str,
    energy_cut_value: float = 1000,
    leaves_only: bool = False,
    output_path: str = None,
    overwrite: bool = False,
    verbose: bool = False,
):
    if output_path is None:
        output_path = _os.path.join(folder, f"subshowers_e{energy_cut_value}.h5")
    if not overwrite and _os.path.exists(output_path):
        if verbose:
            print(f"{output_path} already exists. Skipping.")
        _os.remove(output_path)
    reader = _Reader(folder)
    start_condition = energy_cut(energy_cut_value)
    shower_starts = ShowerStarts(reader, start_condition, verbose=verbose)
    subshowers = Subshowers(reader, leaves_only, verbose=verbose)
    subshowers.add_subshower(shower_starts)
    output = Output(
        shower_starts, subshowers, energy_cut=energy_cut_value, leaves_only=leaves_only
    )
    output.save(output_path)
    if verbose:
        print(f"Saved to {output_path}")


if __name__ == "__main__":
    if len(_sys.argv) < 3:
        print("Usage: python subshowers.py folder energy_cut_value")
    else:
        folder = _sys.argv[1]
        energy_cut_value = float(_sys.argv[2])
        if len(_sys.argv) > 3:
            output_path = _sys.argv[3]
        else:
            output_path = None
        run(folder, energy_cut_value, output_path=output_path, verbose=True)
