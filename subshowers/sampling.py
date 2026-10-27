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

A subshower is much the same wherever it falls: the atmosphere changes
with height, not with x and y. So with shift=True the sampler moves each
subshower it samples across the ground, by a random shift in x and y,
before working out what the array sees of it. Each time a subshower is
sampled it can reach other detectors, which augments the dataset. How far
it can move depends on the array: by default, up to the spacing of its
detectors (see detector.max_shift).

Example
-------
>>> from subshowers import detector
>>> from subshowers.sampling import DatasetSource, SubshowerSampler
>>> array = detector.generate_pierre_auger()
>>> sources = [DatasetSource(folder, energy_cut_value=100.0)
...            for folder in shower_folders]
>>> sampler = SubshowerSampler(sources, array, seed=42, shift=True)
>>> for sample in sampler.iter_epoch(shuffle=True):
...     print(sample.source.name, sample.shift, sample.total_detected_energy)
"""

import os as _os
import warnings as _warnings
from collections import OrderedDict as _OrderedDict
from dataclasses import dataclass as _dataclass, field as _field

import numpy as _np
import pandas as _pd

from subshowers.showers import Reader as _Reader
from subshowers import detector as _detector
from subshowers import subshowers as _subshowers

# history columns a sample is made from
_SAMPLED_COLUMNS = ["pdg", "x", "y", "z", "px", "py", "pz", "kinetic_energy", "time"]


def history_columns(reader, names=_SAMPLED_COLUMNS):
    """
    The history's columns as numpy arrays, and a function giving the
    positions in them of reader.data index labels. Indexing these is much
    faster than indexing the DataFrame by label.
    """
    data = reader.data
    columns = {name: data[name].to_numpy() for name in names}
    index = data.index
    if isinstance(index, _pd.RangeIndex) and index.start == 0 and index.step == 1:
        return columns, _np.asarray

    def positions_of(labels):
        positions = index.get_indexer(_np.atleast_1d(labels))
        if (positions < 0).any():
            raise KeyError(f"not in the history: {labels}")
        return positions

    return columns, positions_of


def _as_shift(shift):
    """
    A shift across the ground, given as (dx, dy) or (dx, dy, 0) in metres,
    as an array of three. Subshowers only move across the ground: the
    atmosphere changes with height, so moving one up or down would change it.
    """
    values = _np.array(shift, dtype=float).reshape(-1)  # a copy, kept by the sample
    if len(values) == 2:
        values = _np.append(values, 0.0)
    if len(values) != 3 or values[2] != 0 or not _np.all(_np.isfinite(values)):
        raise ValueError(
            f"a shift is (dx, dy) in metres, across the ground, not {shift!r}"
        )
    return values


def _generator_for_shifts(rng):
    """
    A random number generator for shifts, apart from rng, so that drawing
    shifts doesn't change what rng draws (such as the order of an epoch).
    """
    try:
        return rng.spawn(1)[0]
    except (AttributeError, TypeError):  # older numpy, or a legacy seeding
        return rng


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
    """
    The detected components of one subshower.

    If the sampler shifted the subshower, start_position is where the
    shifted subshower starts, and the deposits are those of its shifted
    leaves. The history still holds the subshower where it was simulated:
    add shift to the positions of its leaves there to get where they were
    for this sample, and subtract shift from start_position to get where
    the subshower was simulated to start.
    """

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
    # how far the subshower was moved across the ground, (dx, dy, 0) in m
    shift: _np.ndarray = _field(default_factory=lambda: _np.zeros(3))


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
    shift : bool
        Move each subshower sampled across the ground, by a random shift in
        x and y spread evenly over a disc of radius max_shift, before
        working out what the array sees of it. A subshower is much the same
        wherever it falls, so this augments the dataset: each time a
        subshower is sampled, in each epoch, it can reach other detectors.
        The start particle and the leaves move together, so each sample's
        start_position is where the shifted subshower starts, and its shift
        says how far it was moved. Shifts don't change the order subshowers
        are drawn in, for a given seed.
    max_shift : float, optional
        The furthest (m) a subshower is moved, with shift. By default it's
        detector.max_shift(detector_array), the spacing of the array's
        detectors, which is enough for a subshower to land anywhere
        relative to them.
    """

    def __init__(
        self,
        sources,
        detector_array,
        seed=None,
        max_open: int = 2,
        chunk_size: int = 20_000,
        shift: bool = False,
        max_shift: float = None,
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
        self.shift = bool(shift)
        # how far subshowers can be moved (m); 0 without shift
        self.max_shift = self._checked_max_shift(max_shift)
        self._shift_rng = _generator_for_shifts(self.rng) if self.shift else None
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

    # -- shifting --------------------------------------------------------

    def _checked_max_shift(self, max_shift):
        """How far subshowers can be moved (m), from the settings given."""
        if not self.shift:
            if max_shift is not None:
                raise ValueError(
                    "max_shift is the furthest subshowers are shifted,"
                    " so it needs shift=True"
                )
            return 0.0
        if max_shift is None:
            return _detector.max_shift(self.detector_array)
        try:
            distance = float(max_shift)
        except (TypeError, ValueError):
            distance = _np.nan
        if not (_np.isfinite(distance) and distance >= 0):
            raise ValueError(
                f"max_shift must be a distance in metres, at least 0, not {max_shift!r}"
            )
        return distance

    def random_shift(self) -> _np.ndarray:
        """
        A shift for one subshower, (dx, dy, 0) in metres, spread evenly over
        the disc of radius max_shift; no shift at all without shift=True.
        """
        if not self.shift or self.max_shift == 0:
            return _np.zeros(3)
        if self._shift_rng is None:  # shift was turned on after __init__
            self._shift_rng = _generator_for_shifts(self.rng)
        radius = self.max_shift * _np.sqrt(self._shift_rng.random())
        angle = 2 * _np.pi * self._shift_rng.random()
        return _np.array([radius * _np.cos(angle), radius * _np.sin(angle), 0.0])

    # -- building one sample -------------------------------------------

    @staticmethod
    def _columns(reader):
        """The history's columns, as history_columns gives them."""
        return history_columns(reader)

    def _detect(self, reader, leaf_indices, shift=None):
        """
        energy per detector and the indices of leaves that hit, with the
        leaves moved by shift, (dx, dy, 0), if it's given.
        """
        n_detectors = len(self.detector_array)
        energy_per_detector = _np.zeros(n_detectors)
        detected = []
        leaf_indices = _np.asarray(leaf_indices)
        columns, positions_of = self._columns(reader)
        for chunk_start in range(0, len(leaf_indices), self.chunk_size):
            chunk = leaf_indices[chunk_start : chunk_start + self.chunk_size]
            positions = positions_of(chunk)
            start_positions = _np.stack(
                [columns[c][positions] for c in ["x", "y", "z"]], axis=-1
            )
            if shift is not None:
                # a new array, so the history itself isn't moved
                start_positions = start_positions + shift
            directions = _np.stack(
                [columns[c][positions] for c in ["px", "py", "pz"]], axis=-1
            )
            hits = self.detector_array.hit_any(start_positions, directions)
            energies = columns["kinetic_energy"][positions]
            energy_per_detector += _np.sum(hits * energies[:, None], axis=0)
            detected.append(chunk[hits.any(axis=-1)])
        return energy_per_detector, _np.concatenate(detected)

    def make_sample(
        self, source_number: int, subshower_number: int, shift=None
    ) -> DetectedSample:
        """
        What the array sees of one subshower, moved across the ground by
        shift, (dx, dy) in metres, if it's given. Otherwise it's moved by a
        random_shift, which is no shift at all without shift=True.
        """
        source = self._checkout(self.sources[source_number])
        start_index, leaf_indices = source[subshower_number]
        reader = source.reader
        shift = self.random_shift() if shift is None else _as_shift(shift)
        shifted = bool(shift.any())
        energy_per_detector, detected_leaves = self._detect(
            reader, leaf_indices, shift if shifted else None
        )
        columns, positions_of = self._columns(reader)
        start = int(_np.atleast_1d(positions_of(start_index))[0])
        start_kinetic_energy = float(columns["kinetic_energy"][start])
        start_position = _np.array([columns[c][start] for c in ["x", "y", "z"]])
        if shifted:
            start_position = start_position + shift
        start_direction = _np.array([columns[c][start] for c in ["px", "py", "pz"]])
        total_leaf_energy = float(
            _np.sum(columns["kinetic_energy"][positions_of(leaf_indices)])
        )
        sample = DetectedSample(
            source=source,
            subshower_number=subshower_number,
            start_index=start_index,
            leaf_indices=leaf_indices,
            start_pdg=columns["pdg"][start],
            start_time=columns["time"][start],
            start_kinetic_energy=start_kinetic_energy,
            start_position=start_position,
            start_direction=start_direction,
            energy_per_detector=energy_per_detector,
            detected_leaf_indices=detected_leaves,
            total_leaf_energy=total_leaf_energy,
            shift=shift,
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
        subshowers that miss the array entirely are skipped. With shift,
        each epoch moves every subshower by a new shift, and those that
        miss the array are the ones that miss it where they're moved to.
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
                    if self.shift:
                        range_description += (
                            "\nThe subshowers are shifted across the ground"
                            f" by up to {self.max_shift:.6g} m."
                        )

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
