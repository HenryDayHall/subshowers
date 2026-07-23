"""
Sample the detected components of random subshowers.

A "sample" is what a detector array sees of one subshower: which leaf
particles hit a detector, and how much kinetic energy each detector
accumulated (as in responses.sample_energy and the final cells of
scripts/ViewDetectorArrays.ipynb).

The dataset may be spread over several folders (see ShowerFolders in
subshowers.showers, and configs/default.yaml), each folder holding one
shower history and one or more sets of subshowers (e.g. produced by
subshowers.run with different energy cuts).  DatasetSource abstracts one
such (folder, subshower set) pair; SubshowerSampler draws random
subshowers across all sources.

Because each Reader holds a full particle history in memory, sources are
opened lazily and evicted LRU-style, and the epoch iterator visits sources
one at a time (shuffled) rather than hopping between files per sample.

Example
-------
>>> from subshowers import detector
>>> from subshowers.sampling import DatasetSource, SubshowerSampler
>>> array = detector.generate_pierre_auger()
>>> sources = [DatasetSource(folder, energy_cut_value=100.0)
...            for folder in shower_folders]
>>> sampler = SubshowerSampler(sources, array, seed=42)
>>> for sample in sampler.iter_epoch(shuffle=True):
...     print(sample.source.name, sample.total_detected_energy)
"""

import os as _os
import warnings as _warnings
from collections import OrderedDict as _OrderedDict
from dataclasses import dataclass as _dataclass, field as _field

import numpy as _np

from subshowers.showers import Reader as _Reader
from subshowers import subshowers as _subshowers


class DatasetSource:
    """
    One shower folder together with one set of subshowers.

    The subshower set can come from two places:

    * a saved Output file (``output_path``), as written by
      ``subshowers.run(folder, energy_cut_value)``; or
    * computed on the fly from an energy cut (``energy_cut_value``),
      for folders where no Output file has been generated yet.

    Heavy state (the Reader's history DataFrame, the Output) is only
    loaded on first use; ``release()`` drops it again.  The subshower
    index lists are kept after first load - they are just lists of ints.
    """

    def __init__(
        self,
        folder: str,
        output_path: str = None,
        energy_cut_value: float = None,
        leaves_only: bool = True,
        name: str = None,
    ):
        if output_path is None and energy_cut_value is None:
            raise ValueError(
                "Provide output_path (saved Output) or energy_cut_value"
                " (compute subshowers on the fly)."
            )
        self.folder = folder
        self.output_path = output_path
        self.energy_cut_value = energy_cut_value
        self.leaves_only = leaves_only
        self.name = (
            name if name is not None else _os.path.basename(_os.path.normpath(folder))
        )
        self._reader = None
        self._subshower_pairs = None  # list of (start_idx, [leaf idxs])

    # -- lazy loading ------------------------------------------------

    @property
    def reader(self) -> _Reader:
        if self._reader is None:
            self._reader = _Reader(self.folder)
        return self._reader

    def _load_subshowers(self):
        if self._subshower_pairs is not None:
            return
        if self.output_path is not None:
            # Saved starts + subshowers written by subshowers.run(...)
            from subshowers.subshowers import Output as _Output

            output = _Output.load(self.output_path)
            self._subshower_pairs = list(output.subshowers)
            # reuse the Output's reader so the history isn't read twice
            self._reader = output.subshowers.reader
        else:
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore", UserWarning)
                start_condition = _subshowers.energy_cut(self.energy_cut_value)
                starts = _subshowers.ShowerStarts(self.reader, start_condition)
                subs = _subshowers.Subshowers(self.reader, self.leaves_only)
                subs.add_subshower(starts)
            self._subshower_pairs = list(subs)

    def __len__(self) -> int:
        self._load_subshowers()
        return len(self._subshower_pairs)

    def __getitem__(self, item):
        """Return (start_index, leaf_indices) for subshower ``item``."""
        self._load_subshowers()
        return self._subshower_pairs[item]

    def release(self):
        """Drop the in-memory history (keeps the subshower index lists)."""
        self._reader = None

    @property
    def is_open(self) -> bool:
        return self._reader is not None


@_dataclass
class DetectedSample:
    """The detected components of one subshower."""

    source: DatasetSource
    subshower_number: int  # index of the subshower within its source
    start_index: int  # reader.data index of the start particle
    leaf_indices: list  # reader.data indices of all leaf particles
    start_pdg: int
    start_time: float
    start_kinetic_energy: float
    start_position: _np.ndarray  # (3,)
    start_direction: _np.ndarray  # (3,)
    energy_per_detector: _np.ndarray = _field(repr=False)  # (N_detectors,)
    detected_leaf_indices: _np.ndarray = _field(repr=False)  # leaves that hit

    @property
    def total_detected_energy(self) -> float:
        return float(self.energy_per_detector.sum())

    @property
    def n_leaves(self) -> int:
        return len(self.leaf_indices)

    @property
    def n_detected_leaves(self) -> int:
        return len(self.detected_leaf_indices)

    @property
    def n_hit_detectors(self) -> int:
        return int(_np.count_nonzero(self.energy_per_detector))

    @property
    def detected_fraction(self) -> float:
        """Fraction of leaf kinetic energy caught by the array."""
        total = self.total_leaf_energy
        return self.total_detected_energy / total if total > 0 else 0.0

    # filled in by the sampler
    total_leaf_energy: float = 0.0


class SubshowerSampler:
    """
    Iterate over DetectedSamples for random subshowers drawn from one or
    more DatasetSources.

    Parameters
    ----------
    sources : list of DatasetSource
    detector_array : SensitiveRegions
        e.g. detector.generate_pierre_auger() or detector.Triangles(...)
    seed : int or numpy Generator, optional
    max_open : int
        Maximum number of sources whose histories are held in memory at
        once (LRU eviction).
    chunk_size : int
        Leaves are pushed through detector_array.hit_any in chunks of
        this many particles, bounding the (chunk x N_detector) memory.
    """

    def __init__(
        self,
        sources,
        detector_array,
        seed=None,
        max_open: int = 2,
        chunk_size: int = 20_000,
    ):
        if isinstance(sources, DatasetSource):
            sources = [sources]
        self.sources = list(sources)
        if not self.sources:
            raise ValueError("Need at least one DatasetSource")
        self.detector_array = detector_array
        self.rng = (
            seed
            if isinstance(seed, _np.random.Generator)
            else _np.random.default_rng(seed)
        )
        self.max_open = max(1, max_open)
        self.chunk_size = chunk_size
        self._lru = _OrderedDict()
        # counting subshowers touches every source once; readers are
        # evicted again afterwards so only index lists remain resident.
        self._counts = []
        for source in self.sources:
            self._counts.append(len(self._checkout(source)))
        self._cumulative = _np.cumsum(self._counts)

    # -- bookkeeping ---------------------------------------------------

    def __len__(self) -> int:
        """Total number of subshowers across all sources."""
        return int(self._cumulative[-1])

    def _checkout(self, source: DatasetSource) -> DatasetSource:
        """Mark a source as recently used, evicting the least recent."""
        key = id(source)
        if key in self._lru:
            self._lru.move_to_end(key)
        else:
            self._lru[key] = source
            while len(self._lru) > self.max_open:
                _, evicted = self._lru.popitem(last=False)
                evicted.release()
        return source

    def _split_global_index(self, index: int):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        source_number = int(_np.searchsorted(self._cumulative, index, side="right"))
        previous = 0 if source_number == 0 else self._cumulative[source_number - 1]
        return source_number, int(index - previous)

    # -- building one sample -------------------------------------------

    def _detect(self, reader, leaf_indices):
        """energy per detector and the indices of leaves that hit."""
        n_detectors = len(self.detector_array)
        energy_per_detector = _np.zeros(n_detectors)
        detected = []
        leaf_indices = _np.asarray(leaf_indices)
        for chunk_start in range(0, len(leaf_indices), self.chunk_size):
            chunk = leaf_indices[chunk_start : chunk_start + self.chunk_size]
            start_positions = _np.stack(
                [reader.data[c][chunk] for c in ["x", "y", "z"]], axis=-1
            )
            directions = _np.stack(
                [reader.data[c][chunk] for c in ["px", "py", "pz"]], axis=-1
            )
            hits = self.detector_array.hit_any(start_positions, directions)
            energies = _np.array(reader.data["kinetic_energy"][chunk])
            energy_per_detector += _np.sum(hits * energies[:, None], axis=0)
            detected.append(chunk[hits.any(axis=-1)])
        return energy_per_detector, _np.concatenate(detected)

    def make_sample(self, source_number: int, subshower_number: int) -> DetectedSample:
        source = self._checkout(self.sources[source_number])
        start_index, leaf_indices = source[subshower_number]
        reader = source.reader
        energy_per_detector, detected_leaves = self._detect(reader, leaf_indices)
        start_kinetic_energy = float(reader.data["kinetic_energy"][start_index])
        start_position = _np.array(
            [reader.data[c][start_index] for c in ["x", "y", "z"]]
        )
        start_direction = _np.array(
            [reader.data[c][start_index] for c in ["px", "py", "pz"]]
        )
        total_leaf_energy = float(_np.sum(reader.data["kinetic_energy"][leaf_indices]))
        sample = DetectedSample(
            source=source,
            subshower_number=subshower_number,
            start_index=start_index,
            leaf_indices=leaf_indices,
            start_pdg=reader.data["pdg"][start_index],
            start_time=reader.data["time"][start_index],
            start_kinetic_energy=start_kinetic_energy,
            start_position=start_position,
            start_direction=start_direction,
            energy_per_detector=energy_per_detector,
            detected_leaf_indices=detected_leaves,
            total_leaf_energy=total_leaf_energy,
        )
        return sample

    def __getitem__(self, global_index: int) -> DetectedSample:
        return self.make_sample(*self._split_global_index(global_index))

    # -- iteration -------------------------------------------------------

    def iter_epoch(self, shuffle: bool = True, detected_only: bool = False):
        """
        Yield each subshower exactly once (one epoch).

        Sources are visited one after another (in shuffled order if
        requested) and subshowers shuffled within each source, so at most
        one history has to be resident per step.  With detected_only,
        subshowers that miss the array entirely are skipped.
        """
        source_order = _np.arange(len(self.sources))
        if shuffle:
            self.rng.shuffle(source_order)
        for source_number in source_order:
            subshower_order = _np.arange(self._counts[source_number])
            if shuffle:
                self.rng.shuffle(subshower_order)
            for subshower_number in subshower_order:
                sample = self.make_sample(int(source_number), int(subshower_number))
                if detected_only and sample.n_hit_detectors == 0:
                    continue
                yield sample

    def iter_random(self, n_samples: int = None, detected_only: bool = False):
        """
        Yield subshowers i.i.d. uniformly across all sources, with
        replacement; runs forever if n_samples is None.  Statistically
        clean, but may reload histories when consecutive draws land in
        different sources - keep max_open >= number of sources if the
        histories fit in memory.
        """
        produced = 0
        sequential_skips = 0
        while n_samples is None or produced < n_samples:
            global_index = int(self.rng.integers(len(self)))
            sample = self[global_index]
            if detected_only and sample.n_hit_detectors == 0:
                sequential_skips += 1
                if sequential_skips % 100 == 0:
                    detector_mins = self.detector_array.centers.min(axis=0)
                    detector_maxs = self.detector_array.centers.max(axis=0)
                    random_source = self.sources[self.rng.choice(len(self.sources))]
                    range_description = "Coordinates from; [detector, random shower]"
                    for i, coord in enumerate(["x", "y", "z"]):
                        detector_min = detector_mins[i]
                        detector_max = detector_maxs[i]
                        random_min = random_source.reader.data[coord].min()
                        random_max = random_source.reader.data[coord].max()
                        range_description += f"\n{coord}: {detector_min} to {detector_max}, {random_min} to {random_max}"

                    _warnings.warn(
                        f"We have skipped {sequential_skips} sequential subshowers"
                        " that didn't reach a detector.\n" + range_description
                    )
                continue
            sequential_skips = 0
            yield sample
            produced += 1

    def __iter__(self):
        return self.iter_epoch(shuffle=True)
