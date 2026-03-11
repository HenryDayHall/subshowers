"""
TODO currently, this assumes that all folders contain a single shower.
May need to alter that.
"""
import sys as _sys
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
        reader : Reader
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
        if start_condition is not None:
            self._identify_root_starts()
            self._sort_starts()

    def _save_dict(self) -> dict:
        return {
            "folder": self.reader.folder,
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
        root_values = data.loc[root]
        if self.start_condition(**root_values):
            self._starts = _pd.Index([root])
            self._loose_ends = _pd.Index([])
            _warnings.warn(
                "The root particle is also a start particle,"
                " therefore the subshower may not be representative.",
                UserWarning,
            )
            return
        self._starts = []
        self._loose_ends = []
        stack = [root]
        i = 0
        while stack:
            i += 1
            if i % 1000 == 0 and self.verbose:
                fraction = i / len(self.reader)
                print(
                    f"Processed {fraction:00.1%} of total particles."
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
            fraction = i / len(self.reader)
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
        if child_indices.empty:
            subshower.append(parent)
        elif not leaves_only:
            subshower.append(parent)
        stack += list(child_indices)
    return subshower


class Subshowers:
    """
    Locate and retain particles in subshowers given a starting particle.
    """

    def __init__(self, reader: Reader, leaves_only: bool = True, verbose: bool = False):
        self.showerstarts = []
        self.subshowers = []
        self.reader = reader
        self.leaves_only = leaves_only
        self.verbose = verbose

    def __len__(self):
        return len(self.showerstarts)

    def add_subshower(self, start_index: list | _pd.Index):
        n_to_add = len(start_index)
        for i, start in enumerate(start_index):
            if start in self.showerstarts:
                continue
            if self.verbose:
                fraction = i / n_to_add
                print(
                    f"Processed {fraction:00.1%} of total starts.",
                    end="\r",
                )
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
            start
            for start, subshower in zip(self.showerstarts, self.subshowers)
            for _ in subshower
        ]
        flat_subshowers = sum(self.subshowers, [])
        return {
            "folder": self.reader.folder,
            "starts": flat_starts,
            "subshowers": flat_subshowers,
            "leaves_only": self.leaves_only,
        }

    @classmethod
    def _load_dict(cls, dictionary: dict):
        reader = Reader(dictionary["folder"])
        new = Subshowers(reader, dictionary["leaves_only"])
        flat_starts = dictionary["starts"]
        flat_subshowers = dictionary["subshowers"]
        new.showerstarts = sorted(set(flat_starts))
        new.subshowers = [[] for _ in new.showerstarts]
        for start, subshower in zip(flat_starts, flat_subshowers):
            new.subshowers[new.showerstarts.index(start)].append(subshower)
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

    def _prep_dataframes(self):
        to_save = {}
        shower_starts_dict = self.shower_starts._save_dict()
        for key, value in shower_starts_dict.items():
            if key in ["folder", "root"]:
                value = [value]
            df = _pd.DataFrame.from_dict({key: value})
            to_save[f"{self.SHOWER_STARTS_PREFIX}_{key}"] = df
        subshowers_dict = self.subshowers._save_dict()
        df = _pd.DataFrame.from_dict(
            {
                "starts": subshowers_dict["starts"],
                "subshowers": subshowers_dict["subshowers"],
            }
        )
        to_save[f"{self.SUBSHOWERS_PREFIX}"] = df
        for key in ["folder", "leaves_only"]:
            df = _pd.DataFrame.from_dict({key: [subshowers_dict[key]]})
            to_save[f"{self.SUBSHOWERS_PREFIX}_{key}"] = df
        for key, value in self.metadata.items():
            df = _pd.DataFrame.from_dict({key: [value]})
            to_save[f"{self.METADATA_PREFIX}_{key}"] = df
        return to_save

    @classmethod
    def _unpack_dataframes(cls, loaded: dict):
        shower_starts_dict = {}
        for key in loaded:
            if key.startswith(cls.SHOWER_STARTS_PREFIX):
                tail = key[len(cls.SHOWER_STARTS_PREFIX) + 1 :]
                value = loaded[key][tail]
                if tail in ["folder", "root"]:
                    value = value[0]
                shower_starts_dict[tail] = value
        shower_starts = ShowerStarts._load_dict(shower_starts_dict)
        subshowers_dict = {}
        for tail in ["folder", "leaves_only"]:
            key = f"{cls.SUBSHOWERS_PREFIX}_{tail}"
            value = loaded[key][tail]
            if tail in ["leaves_only", "folder"]:
                value = value[0]
            subshowers_dict[tail] = value
        subshowers_dict["starts"] = loaded[f"{cls.SUBSHOWERS_PREFIX}"]["starts"]
        subshowers_dict["subshowers"] = loaded[f"{cls.SUBSHOWERS_PREFIX}"]["subshowers"]
        subshowers = Subshowers._load_dict(subshowers_dict)
        metadata = {}
        for key in loaded:
            if key.startswith(cls.METADATA_PREFIX):
                tail = key[len(cls.METADATA_PREFIX) + 1 :]
                metadata[tail] = loaded[key][tail][0]
        new = cls(shower_starts, subshowers, **metadata)
        return new

    def save(self, path: str):
        assert path.endswith(".h5"), "Path must end with .h5"
        to_save = self._prep_dataframes()
        for key, df in to_save.items():
            df.to_hdf(path, key=key, mode="a")

    @classmethod
    def load(cls, path: str):
        assert path.endswith(".h5"), "Path must end with .h5"
        loaded = {}
        with _pd.HDFStore(path) as store:
            for key in store.keys():
                loaded[key.strip("/")] = store.get(key)
        return cls._unpack_dataframes(loaded)


def run(
    folder: str,
    energy_cut_value: float = 1000,
    leaves_only: bool = False,
    output_path: str = None,
    overwrite: bool = False,
    verbose: bool = False,
):
    if output_path is None:
        output_path = _os.path.join(folder, "subshowers.h5")
    if not overwrite and _os.path.exists(output_path):
        if verbose:
            print(f"{output_path} already exists. Skipping.")
        _os.remove(output_path)
    reader = Reader(folder)
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

