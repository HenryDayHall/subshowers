"""
Generate and store detector responses to subshower sets
"""

import json
import os
from dataclasses import dataclass, field

import numpy as np


def sample_energy(reader, subshower_idxs, detector_array):
    start_positions = np.stack(
        [reader.data[c][subshower_idxs] for c in ["x", "y", "z"]], axis=-1
    )
    directions = np.stack(
        [reader.data[c][subshower_idxs] for c in ["px", "py", "pz"]], axis=-1
    )
    hits = detector_array.hit_any(start_positions, directions)
    energies = np.array(reader.data["kinetic_energy"][subshower_idxs])
    energy_per_detector = np.sum(hits * energies[:, None], axis=0)
    return energy_per_detector


@dataclass
class StoredSample:
    """
    One subshower of a Responses, with the attributes of
    sampling.DetectedSample that observables use. source is a name here,
    rather than a DatasetSource.
    """

    source: str
    subshower_number: int
    start_pdg: int
    start_kinetic_energy: float
    start_time: float
    start_position: np.ndarray  # (3,)
    start_direction: np.ndarray  # (3,)
    n_leaves: int
    n_detected_leaves: int
    total_leaf_energy: float
    energy_per_detector: np.ndarray = field(repr=False)  # (N_detectors,)
    # history index labels of the leaves that hit some detector; None if unknown
    detected_leaf_indices: np.ndarray = field(default=None, repr=False)

    @property
    def total_detected_energy(self) -> float:
        return float(self.energy_per_detector.sum())

    @property
    def n_hit_detectors(self) -> int:
        return int(np.count_nonzero(self.energy_per_detector))

    @property
    def detected_fraction(self) -> float:
        """Fraction of leaf kinetic energy caught by the array."""
        total = self.total_leaf_energy
        return self.total_detected_energy / total if total > 0 else 0.0


class Responses:
    """
    What a detector array saw of a population of subshowers: each
    subshower's start particle, and the energy each detector recorded from
    its leaves, kept only for detectors that recorded some.

    The population can be subshowers sampled from the dataset by
    sampling.SubshowerSampler, or made by a generative model: from_samples
    takes anything with the attributes it reads. observables.record_of works
    out observables from a Responses, so populations measured with the same
    array can be compared.

    For subshowers from the dataset, it also keeps which of their leaves hit
    some detector (as history index labels), so that what reaches the
    detectors can be summarised like what reaches the ground (see
    ground.GroundSummaries.detected_in).
    """

    # one value per subshower, and its type
    scalar_columns = {
        "source": str,  # e.g. the shower, or the model, it came from
        "subshower_number": np.int64,  # within its source; -1 if unknown
        "start_pdg": np.int64,
        "start_kinetic_energy": np.float64,
        "start_time": np.float64,
        "n_leaves": np.int64,  # -1 if unknown
        "n_detected_leaves": np.int64,  # -1 if unknown
        "total_leaf_energy": np.float64,  # nan if unknown
    }
    # three values per subshower
    vector_columns = ["start_position", "start_direction"]
    # 2 added which leaves hit a detector; files of version 1 are still read
    FORMAT_VERSION = 2

    def __init__(
        self,
        n_detectors,
        values,
        hit_offsets,
        hit_detectors,
        hit_energies,
        metadata=None,
        detected_offsets=None,
        detected_leaves=None,
        detected_known=None,
    ):
        """
        Parameters
        ----------
        n_detectors : int
            Detectors in the array.
        values : dict
            An array for each of scalar_columns and vector_columns, with
            one entry (or row of three) per subshower.
        hit_offsets : array (N_subshowers + 1,)
            Subshower i's hits are hit_offsets[i]:hit_offsets[i + 1] of
            hit_detectors and hit_energies.
        hit_detectors : array (N_hits,)
            Index of each detector that recorded energy.
        hit_energies : array (N_hits,)
            Energy it recorded, in GeV.
        metadata : dict
            Anything that can be written as json, e.g. how the responses
            were made.
        detected_offsets : array (N_subshowers + 1,), optional
            Subshower i's leaves that hit some detector are
            detected_offsets[i]:detected_offsets[i + 1] of detected_leaves.
        detected_leaves : array (N_detected,), optional
            Their index labels in the subshower's history.
        detected_known : array (N_subshowers,) of bool, optional
            Whether each subshower's detected leaves are known; those that
            aren't, such as a generative model's, have none listed. Without
            detected_offsets, no subshower's are known.
        """
        self.n_detectors = int(n_detectors)
        self.hit_offsets = np.asarray(hit_offsets, dtype=np.int64)
        self.hit_detectors = np.asarray(hit_detectors, dtype=np.int64)
        self.hit_energies = np.asarray(hit_energies, dtype=np.float64)
        n_subshowers = len(self.hit_offsets) - 1
        if detected_offsets is None:
            detected_offsets = np.zeros(n_subshowers + 1)
            detected_leaves = []
            detected_known = np.zeros(n_subshowers, dtype=bool)
        self.detected_offsets = np.asarray(detected_offsets, dtype=np.int64)
        self.detected_leaves = np.asarray(detected_leaves, dtype=np.int64)
        if detected_known is None:
            detected_known = np.ones(n_subshowers, dtype=bool)
        self.detected_known = np.asarray(detected_known, dtype=bool).reshape(-1)
        self.values = {}
        for name, kind in self.scalar_columns.items():
            self.values[name] = np.asarray(values[name], dtype=kind).reshape(
                n_subshowers
            )
        for name in self.vector_columns:
            self.values[name] = np.asarray(values[name], dtype=np.float64).reshape(
                n_subshowers, 3
            )
        self.metadata = dict(metadata or {})
        self._check()

    def _check(self):
        offsets = self.hit_offsets
        if len(offsets) == 0 or offsets[0] != 0 or np.any(np.diff(offsets) < 0):
            raise ValueError("hit_offsets must start at 0 and never decrease")
        if not offsets[-1] == len(self.hit_detectors) == len(self.hit_energies):
            raise ValueError("hit_offsets must end at the number of hits")
        if len(self.hit_detectors) and not (
            0 <= self.hit_detectors.min()
            and self.hit_detectors.max() < self.n_detectors
        ):
            raise ValueError(f"hit detectors must be below {self.n_detectors}")
        detected = self.detected_offsets
        if (
            len(detected) != len(offsets)
            or detected[0] != 0
            or np.any(np.diff(detected) < 0)
            or detected[-1] != len(self.detected_leaves)
        ):
            raise ValueError(
                "detected_offsets must have one more entry than there are"
                " subshowers, start at 0, never decrease, and end at the"
                " number of detected leaves"
            )
        if len(self.detected_known) != len(offsets) - 1:
            raise ValueError("detected_known must have an entry for each subshower")
        if np.any(np.diff(detected)[~self.detected_known] > 0):
            raise ValueError("subshowers whose detected leaves aren't known list none")

    # ----------------------------------------------------------- building

    @classmethod
    def from_samples(cls, samples, n_detectors, source=None, metadata=None):
        """
        Responses to some samples: sampling.DetectedSamples, or any objects
        with start_pdg, start_kinetic_energy, start_time, start_position,
        start_direction and energy_per_detector (one value per detector).
        Their subshower_number, n_leaves, n_detected_leaves,
        total_leaf_energy and detected_leaf_indices are kept if they have
        them. source names the population; by default it's each sample's
        source, or its source's name.
        """
        values = {name: [] for name in list(cls.scalar_columns) + cls.vector_columns}
        offsets, detectors, energies = [0], [], []
        detected_offsets, detected_leaves, detected_known = [0], [], []
        for sample in samples:
            leaves = getattr(sample, "detected_leaf_indices", None)
            detected_known.append(leaves is not None)
            leaves = np.asarray([] if leaves is None else leaves, dtype=np.int64)
            detected_leaves.append(leaves.reshape(-1))
            detected_offsets.append(detected_offsets[-1] + leaves.size)
            energy = np.asarray(sample.energy_per_detector, dtype=np.float64)
            if energy.shape != (n_detectors,):
                raise ValueError(
                    f"energy_per_detector has shape {energy.shape},"
                    f" not ({n_detectors},)"
                )
            hit = np.flatnonzero(energy)
            detectors.append(hit)
            energies.append(energy[hit])
            offsets.append(offsets[-1] + len(hit))
            if source is None:
                name = getattr(sample, "source", "")
                name = name if isinstance(name, str) else getattr(name, "name", "")
            else:
                name = source
            values["source"].append(name)
            values["subshower_number"].append(getattr(sample, "subshower_number", -1))
            for column in [
                "start_pdg",
                "start_kinetic_energy",
                "start_time",
                "start_position",
                "start_direction",
            ]:
                values[column].append(getattr(sample, column))
            values["n_leaves"].append(getattr(sample, "n_leaves", -1))
            values["n_detected_leaves"].append(getattr(sample, "n_detected_leaves", -1))
            values["total_leaf_energy"].append(
                getattr(sample, "total_leaf_energy", np.nan)
            )
        return cls(
            n_detectors,
            values,
            offsets,
            np.concatenate(detectors) if detectors else [],
            np.concatenate(energies) if energies else [],
            metadata,
            detected_offsets,
            np.concatenate(detected_leaves) if detected_leaves else [],
            detected_known,
        )

    @classmethod
    def concatenate(cls, parts, metadata=None):
        """All the subshowers of several Responses, measured with one array."""
        parts = list(parts)
        if not parts:
            raise ValueError("nothing to concatenate")
        n_detectors = parts[0].n_detectors
        if any(part.n_detectors != n_detectors for part in parts):
            raise ValueError("the parts were measured with different arrays")
        values = {
            name: np.concatenate([part.values[name] for part in parts])
            for name in parts[0].values
        }
        hit_counts = np.concatenate([np.diff(part.hit_offsets) for part in parts])
        detected_counts = np.concatenate(
            [np.diff(part.detected_offsets) for part in parts]
        )
        return cls(
            n_detectors,
            values,
            np.concatenate(([0], np.cumsum(hit_counts))),
            np.concatenate([part.hit_detectors for part in parts]),
            np.concatenate([part.hit_energies for part in parts]),
            metadata,
            np.concatenate(([0], np.cumsum(detected_counts))),
            np.concatenate([part.detected_leaves for part in parts]),
            np.concatenate([part.detected_known for part in parts]),
        )

    @staticmethod
    def _picked(offsets, indices):
        """New offsets, and positions in the flat array, of some subshowers' entries."""
        counts = np.diff(offsets)[indices]
        new_offsets = np.concatenate(([0], np.cumsum(counts)))
        positions = np.repeat(offsets[indices] - new_offsets[:-1], counts)
        positions += np.arange(new_offsets[-1])
        return new_offsets, positions

    def select(self, which):
        """The subshowers a boolean mask, or array of indices, picks out."""
        indices = np.arange(len(self))[which]
        offsets, positions = self._picked(self.hit_offsets, indices)
        detected_offsets, detected = self._picked(self.detected_offsets, indices)
        return Responses(
            self.n_detectors,
            {name: column[indices] for name, column in self.values.items()},
            offsets,
            self.hit_detectors[positions],
            self.hit_energies[positions],
            self.metadata,
            detected_offsets,
            self.detected_leaves[detected],
            self.detected_known[indices],
        )

    # ------------------------------------------------------------ reading

    def __len__(self):
        return len(self.hit_offsets) - 1

    @property
    def n_hit_detectors(self):
        """Detectors that recorded energy, for each subshower."""
        return np.diff(self.hit_offsets)

    @property
    def detected(self):
        """True for subshowers that some detector recorded energy from."""
        return self.n_hit_detectors > 0

    @property
    def total_detected_energy(self):
        """Energy recorded by the whole array, for each subshower."""
        owner = np.repeat(np.arange(len(self)), self.n_hit_detectors)
        return np.bincount(owner, weights=self.hit_energies, minlength=len(self))

    def energy_per_detector(self, i):
        """Energy recorded by every detector from subshower i."""
        energy = np.zeros(self.n_detectors)
        start, end = self.hit_offsets[i], self.hit_offsets[i + 1]
        energy[self.hit_detectors[start:end]] = self.hit_energies[start:end]
        return energy

    @property
    def keeps_detected_leaves(self):
        """True if every subshower's detected leaves are known."""
        return bool(self.detected_known.all())

    def detected_leaf_indices(self, i):
        """
        History index labels of subshower i's leaves that hit some
        detector, or None if they aren't known.
        """
        if not self.detected_known[i]:
            return None
        start, end = self.detected_offsets[i], self.detected_offsets[i + 1]
        return self.detected_leaves[start:end].copy()

    def sample(self, i):
        """Subshower i, as a StoredSample."""
        values = {name: column[i] for name, column in self.values.items()}
        return StoredSample(
            source=str(values["source"]),
            subshower_number=int(values["subshower_number"]),
            start_pdg=int(values["start_pdg"]),
            start_kinetic_energy=float(values["start_kinetic_energy"]),
            start_time=float(values["start_time"]),
            start_position=values["start_position"].copy(),
            start_direction=values["start_direction"].copy(),
            n_leaves=int(values["n_leaves"]),
            n_detected_leaves=int(values["n_detected_leaves"]),
            total_leaf_energy=float(values["total_leaf_energy"]),
            energy_per_detector=self.energy_per_detector(i),
            detected_leaf_indices=self.detected_leaf_indices(i),
        )

    def samples(self):
        """Every subshower, in order, as StoredSamples."""
        for i in range(len(self)):
            yield self.sample(i)

    # ------------------------------------------------------- saving, loading

    def save(self, path):
        """Write to a .npz file, which load reads back."""
        arrays = {
            "format_version": np.array(self.FORMAT_VERSION),
            "n_detectors": np.array(self.n_detectors),
            "metadata": np.array(json.dumps(self.metadata)),
            "hit_offsets": self.hit_offsets,
            "hit_detectors": self.hit_detectors,
            "hit_energies": self.hit_energies,
            "detected_offsets": self.detected_offsets,
            "detected_leaves": self.detected_leaves,
            "detected_known": self.detected_known,
        }
        for name, column in self.values.items():
            arrays[f"value_{name}"] = column
        folder = os.path.dirname(os.path.abspath(path))
        os.makedirs(folder, exist_ok=True)
        temporary = path + ".tmp"
        with open(temporary, "wb") as f:
            np.savez_compressed(f, **arrays)
        os.replace(temporary, path)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            version = int(data["format_version"])
            if version not in (1, cls.FORMAT_VERSION):
                raise ValueError(
                    f"{path} has format version {version}, not {cls.FORMAT_VERSION}"
                )
            names = list(cls.scalar_columns) + cls.vector_columns
            detected = [None, None, None]  # version 1 didn't keep them
            if version >= 2:
                detected = [
                    data["detected_offsets"],
                    data["detected_leaves"],
                    data["detected_known"],
                ]
            return cls(
                int(data["n_detectors"]),
                {name: data[f"value_{name}"] for name in names},
                data["hit_offsets"],
                data["hit_detectors"],
                data["hit_energies"],
                json.loads(data["metadata"].item()),
                *detected,
            )

    @classmethod
    def read_metadata(cls, path):
        """The metadata of a saved Responses, without reading the rest."""
        with np.load(path, allow_pickle=False) as data:
            return json.loads(data["metadata"].item())
