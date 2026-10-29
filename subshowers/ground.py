"""
What each subshower puts on the ground, summarised for the ground-level
observables used to compare air shower simulations: lateral distributions,
energy spectra, particle numbers, arrival times, and readings at a reference
distance from the axis (like the S800 of the Telescope Array).

The particles a subshower puts on the ground are its leaves, carried in a
straight line from where they were made to the ground plane, as the detector
sampler does (see sampling.SubshowerSampler), and slowed by their mass.
CORSIKA 8 writes a particle to the history before deciding whether to track
it, so particles below its tracking cuts are leaves too; those below
min_kinetic_energy (by default the cuts c8_air_shower_with_history uses)
aren't counted. Neutrinos aren't counted either. Energies are as the
particles were made: losses on the way down aren't known.

Distances and arrival times are measured about an axis, in two frames:

* "subshower": the line through the subshower's start particle along its
  direction;
* "shower": the line through the primary particle along its direction,
  which is the shower's axis.

The distance from the axis is measured in the shower plane (perpendicular to
the axis). A particle's arrival delay is the time it reaches the ground less
the time a plane front, moving along the axis at the speed of light and
passing the axis particle at that particle's time, reaches the same place.

Each subshower is summarised by histograms with fixed bins (RADIUS_EDGES,
ENERGY_EDGES), so populations summarised apart, such as the dataset and a
generative model's output, can be compared. GroundSummaries holds them for
many subshowers. Positions are in metres, as in the history, with the ground
at z = height (0 is CORSIKA's observation level); times in the history are
in seconds, and delays here are in nanoseconds.
"""

import json
import os
from dataclasses import dataclass, field

import numpy as np

from subshowers import detector as _detector
from subshowers.sampling import history_columns as _history_columns

# kinds of particle on the ground; neutrinos are left out
SPECIES = ["electrons", "photons", "muons", "hadrons"]
SPECIES_LABELS = {
    "electrons": "e±",
    "photons": "γ",
    "muons": "μ±",
    "hadrons": "hadrons",
}
NEUTRINOS = (12, 14, 16)

# the frames distances and delays are measured in, as above
FRAMES = ["subshower", "shower"]

# kinetic energy (GeV) below which a particle isn't counted: the defaults of
# c8_air_shower_with_history's --emcut, --mucut and --hadcut
DEFAULT_MIN_KINETIC_ENERGY = {
    "electrons": 5.0e-4,
    "photons": 5.0e-4,
    "muons": 0.3,
    "hadrons": 0.3,
}
# distance (m) from the axis at which readings are taken, as the Telescope
# Array reads S800 at 800 m from the shower's axis; subshowers are narrower
DEFAULT_REFERENCE_DISTANCE = {"subshower": 100.0, "shower": 800.0}
# readings at a reference distance r are taken over r / 10**w to r * 10**w
REFERENCE_HALF_WIDTH = 0.1  # w, in decades

# Bins of distance from the axis (m): the disc within 10 cm, then ten a
# decade out to 10 km. Particles further out are left out of the lateral
# distributions and arrival times, but are counted.
RADIUS_EDGES = np.concatenate(([0.0], np.logspace(-1, 4, 51)))
# Bins of kinetic energy (GeV), ten a decade from 0.1 MeV to 100 TeV
ENERGY_EDGES = np.logspace(-4, 5, 91)

SPEED_OF_LIGHT = 299_792_458.0  # m/s

MASSES = {  # GeV, by |pdg|
    11: 0.000510999,
    12: 0.0,
    13: 0.105658,
    14: 0.0,
    15: 1.77686,
    16: 0.0,
    22: 0.0,
    111: 0.134977,
    130: 0.497611,
    211: 0.139570,
    310: 0.497611,
    321: 0.493677,
    2112: 0.939565,
    2212: 0.938272,
    3112: 1.197449,
    3122: 1.115683,
    3212: 1.192642,
    3222: 1.189370,
    3312: 1.321710,
    3322: 1.314860,
    3334: 1.672450,
}
NUCLEON_MASS = 0.931494  # GeV, for nuclei (pdg 10LZZZAAAI), by mass number

# The histograms summarising one subshower, and their shapes, where S is the
# number of species, F of frames, R of radius bins and E of energy bins:
#   number        (S,)       particles reaching the ground
#   spectrum      (S, E)     of them, by kinetic energy
#   lateral       (F, S, R)  by distance from the axis
#   delay_sum     (F, S, R)  sum of their arrival delays (ns)
#   delay_sum2    (F, S, R)  sum of the squares of their delays (ns²)
#   at_reference  (F, S)     within REFERENCE_HALF_WIDTH of the reference distance
# Counts are float32 (exact to 16 million), so a frame whose axis isn't
# known can be nan; sums of delays are float64.
HISTOGRAMS = [
    "number",
    "spectrum",
    "lateral",
    "delay_sum",
    "delay_sum2",
    "at_reference",
]


def _histogram_shapes():
    n_species, n_frames = len(SPECIES), len(FRAMES)
    n_radii, n_energies = len(RADIUS_EDGES) - 1, len(ENERGY_EDGES) - 1
    return {
        "number": ((n_species,), np.float32),
        "spectrum": ((n_species, n_energies), np.float32),
        "lateral": ((n_frames, n_species, n_radii), np.float32),
        "delay_sum": ((n_frames, n_species, n_radii), np.float64),
        "delay_sum2": ((n_frames, n_species, n_radii), np.float64),
        "at_reference": ((n_frames, n_species), np.float32),
    }


class GroundConfigError(ValueError):
    """A config's ground section can't be used."""


@dataclass
class GroundSettings:
    """
    What decides which particles count as reaching the ground, and where
    readings are taken. Missing entries take the defaults.

    height : float
        Height of the ground, in the history's coordinates (m).
    min_kinetic_energy : dict
        {species: GeV} below which particles aren't counted.
    reference_distance : dict
        {frame: m} at which readings are taken.
    """

    height: float = 0.0
    min_kinetic_energy: dict = field(default_factory=dict)
    reference_distance: dict = field(default_factory=dict)

    def __post_init__(self):
        try:
            self.height = float(self.height)
        except (TypeError, ValueError):
            raise GroundConfigError(f"height must be a number, not {self.height!r}")
        if not np.isfinite(self.height):
            raise GroundConfigError(f"height must be finite, not {self.height}")
        self.min_kinetic_energy = self._filled(
            "min_kinetic_energy", self.min_kinetic_energy, DEFAULT_MIN_KINETIC_ENERGY
        )
        self.reference_distance = self._filled(
            "reference_distance", self.reference_distance, DEFAULT_REFERENCE_DISTANCE
        )
        for frame, distance in self.reference_distance.items():
            if distance <= 0:
                raise GroundConfigError(
                    f"reference_distance: {frame} must be above 0, not {distance}"
                )

    @staticmethod
    def _filled(name, given, defaults):
        if given is None:
            given = {}
        if not isinstance(given, dict):
            raise GroundConfigError(
                f"{name} must map {sorted(defaults)} to numbers, not {given!r}"
            )
        unknown = sorted(set(given) - set(defaults))
        if unknown:
            raise GroundConfigError(
                f"{name} has {unknown}, which aren't among {list(defaults)}"
            )
        filled = dict(defaults)
        for key, value in given.items():
            try:
                filled[key] = float(value)
            except (TypeError, ValueError):
                raise GroundConfigError(
                    f"{name}: {key} must be a number, not {value!r}"
                )
            if not (np.isfinite(filled[key]) and filled[key] >= 0):
                raise GroundConfigError(
                    f"{name}: {key} must be a finite number, at least 0, not {value!r}"
                )
        return filled

    @classmethod
    def from_config(cls, config):
        """The settings in a config's ground section, which may be left out."""
        section = config.get("ground") if isinstance(config, dict) else None
        if section is None:
            return cls()
        if not isinstance(section, dict):
            raise GroundConfigError(
                f"the ground section must be a mapping, not {section!r}"
            )
        unknown = sorted(
            set(section) - {"height", "min_kinetic_energy", "reference_distance"}
        )
        if unknown:
            raise GroundConfigError(
                f"the ground section has {unknown}; it can have height,"
                " min_kinetic_energy and reference_distance"
            )
        return cls(**section)

    def as_dict(self):
        return {
            "height": self.height,
            "min_kinetic_energy": dict(self.min_kinetic_energy),
            "reference_distance": dict(self.reference_distance),
        }

    def reference_annulus(self, frame):
        """(inner, outer) radius (m) over which readings are taken in a frame."""
        return annulus(self.reference_distance[frame])


def annulus(distance):
    """(inner, outer) radius (m) over which readings at a distance are taken."""
    return distance * 10**-REFERENCE_HALF_WIDTH, distance * 10**REFERENCE_HALF_WIDTH


# -------------------------------------------------------------- particles


def species_of(pdg):
    """The index in SPECIES of each particle, or -1 for neutrinos."""
    pdg = np.abs(np.asarray(pdg, dtype=np.int64))
    species = np.full(pdg.shape, SPECIES.index("hadrons"))
    species[pdg == 11] = SPECIES.index("electrons")
    species[pdg == 22] = SPECIES.index("photons")
    species[pdg == 13] = SPECIES.index("muons")
    species[np.isin(pdg, NEUTRINOS)] = -1
    return species


def masses_of(pdg):
    """
    The mass of each particle (GeV), from MASSES or, for nuclei, its mass
    number; nan if it's neither.
    """
    pdg = np.abs(np.asarray(pdg, dtype=np.int64))
    codes, inverse = np.unique(pdg, return_inverse=True)
    masses = np.full(len(codes), np.nan)
    for i, code in enumerate(codes):
        if code in MASSES:
            masses[i] = MASSES[code]
        elif code >= 1_000_000_000:
            masses[i] = ((code // 10) % 1000) * NUCLEON_MASS
    return masses[inverse].reshape(pdg.shape)


def speeds(pdg, kinetic_energy, momentum=None):
    """
    Each particle's speed as a fraction of the speed of light, from its
    kinetic energy and mass. For particles of unknown mass, the mass is
    found from the size of the momentum (GeV) if it's given, and otherwise
    they're taken to move at the speed of light.
    """
    kinetic_energy = np.asarray(kinetic_energy, dtype=float)
    mass = masses_of(pdg)
    with np.errstate(divide="ignore", invalid="ignore"):
        speed = np.sqrt(kinetic_energy * (kinetic_energy + 2 * mass)) / (
            kinetic_energy + mass
        )
        unknown = np.isnan(mass)
        if unknown.any():
            if momentum is None:
                speed[unknown] = 1.0
            else:
                # E = T + m and p^2 = E^2 - m^2, so p / E = 2 T p / (T^2 + p^2)
                p = np.linalg.norm(np.asarray(momentum, dtype=float), axis=-1)[unknown]
                t = kinetic_energy[unknown]
                speed[unknown] = 2 * t * p / (t * t + p * p)
    return np.where(np.isfinite(speed), np.clip(speed, 0.0, 1.0), 0.0)


def to_ground(position, momentum, time, speed, height=0.0):
    """
    Where and when particles, moving in straight lines from position along
    momentum at speed (a fraction of the speed of light), cross the plane
    z = height on the way down.

    Returns
    -------
    reaches : array (N,) of bool
        True for particles that cross the plane: moving down, from on or
        above it, and not at rest.
    ground_position : array (M, 3)
        Where those that reach it do.
    ground_time : array (M,)
        When, in the units of time (seconds in a history).
    """
    position = np.asarray(position, dtype=float).reshape(-1, 3)
    momentum = np.asarray(momentum, dtype=float).reshape(-1, 3)
    time = np.asarray(time, dtype=float).reshape(-1)
    speed = np.asarray(speed, dtype=float).reshape(-1)
    norm = np.linalg.norm(momentum, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        direction = momentum / norm[:, None]
        reaches = (direction[:, 2] < 0) & (position[:, 2] >= height) & (speed > 0)
        length = (position[reaches, 2] - height) / -direction[reaches, 2]
    ground_position = position[reaches] + direction[reaches] * length[:, None]
    ground_position[:, 2] = height
    ground_time = time[reaches] + length / (speed[reaches] * SPEED_OF_LIGHT)
    return reaches, ground_position, ground_time


def axis_coordinates(points, times, axis_point, axis_direction, axis_time):
    """
    Each point's distance (m) from an axis, in the shower plane, and the
    delay (ns) of each time after a plane front that moves along the axis
    at the speed of light and passes axis_point at axis_time (both times in
    seconds). Everything is nan if the axis has no direction; delays are
    nan if axis_time is.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    direction = np.asarray(axis_direction, dtype=float).reshape(3)
    norm = np.linalg.norm(direction)
    if not (np.isfinite(norm) and norm > 0):
        nothing = np.full(len(points), np.nan)
        return nothing, nothing.copy()
    direction = direction / norm
    offset = points - np.asarray(axis_point, dtype=float).reshape(3)
    radius = np.linalg.norm(np.cross(offset, direction), axis=-1)
    along = offset @ direction
    times = np.asarray(times, dtype=float).reshape(-1)
    delay = (times - axis_time - along / SPEED_OF_LIGHT) * 1e9
    return radius, delay


def leaves_to_ground(pdg, position, momentum, kinetic_energy, time, height=0.0):
    """
    The leaves of a subshower that reach the ground, as recorded in a
    history: where each was made, its momentum (GeV), kinetic energy (GeV)
    and the time it was made (s).

    Returns pdg, ground_position (M, 3), kinetic_energy, ground_time of
    those that reach the ground.
    """
    pdg = np.asarray(pdg).reshape(-1)
    kinetic_energy = np.asarray(kinetic_energy, dtype=float).reshape(-1)
    speed = speeds(pdg, kinetic_energy, momentum)
    reaches, ground_position, ground_time = to_ground(
        position, momentum, time, speed, height
    )
    return pdg[reaches], ground_position, kinetic_energy[reaches], ground_time


def _empty_summary():
    return {
        name: np.zeros(shape, dtype=dtype)
        for name, (shape, dtype) in _histogram_shapes().items()
    }


def _valid_axis(axis):
    if axis is None:
        return False
    point, direction, _ = axis
    point = np.asarray(point, dtype=float)
    direction = np.asarray(direction, dtype=float)
    return bool(
        np.all(np.isfinite(point))
        and np.all(np.isfinite(direction))
        and np.linalg.norm(direction) > 0
    )


def summarise(
    pdg, ground_position, kinetic_energy, ground_time=None, axes=None, settings=None
):
    """
    The histograms (see HISTOGRAMS) of the particles one subshower puts on
    the ground.

    Parameters
    ----------
    pdg, kinetic_energy : arrays (N,)
        Of each particle reaching the ground; kinetic energy in GeV.
    ground_position : array (N, 3), or (N, 2) of x and y
        Where each reaches the ground, in metres.
    ground_time : array (N,), optional
        When, in seconds. Without times, the sums of delays are nan.
    axes : dict
        {frame: (point, direction, time)} for the frames in FRAMES. A
        frame left out, or None, isn't known, and its histograms are nan.
    settings : GroundSettings

    Returns
    -------
    {name: array} for each name in HISTOGRAMS
    """
    settings = settings or GroundSettings()
    axes = axes or {}
    pdg = np.asarray(pdg).reshape(-1)
    n_particles = len(pdg)
    kinetic_energy = np.asarray(kinetic_energy, dtype=float).reshape(-1)
    position = np.asarray(ground_position, dtype=float)
    if position.size:
        position = position.reshape(n_particles, -1)
    else:
        position = np.zeros((0, 3))  # reshape can't work out a width from nothing
    if position.shape[1] == 2:
        position = np.column_stack((position, np.full(n_particles, settings.height)))
    time = None
    if ground_time is not None:
        time = np.asarray(ground_time, dtype=float).reshape(-1)

    species = species_of(pdg)
    minimum = np.array([settings.min_kinetic_energy[name] for name in SPECIES])
    counted = species >= 0
    counted[counted] = kinetic_energy[counted] >= minimum[species[counted]]
    species = species[counted]
    kinetic_energy = kinetic_energy[counted]
    position = position[counted]
    if time is not None:
        time = time[counted]

    n_species = len(SPECIES)
    n_radii, n_energies = len(RADIUS_EDGES) - 1, len(ENERGY_EDGES) - 1
    summary = _empty_summary()
    summary["number"][:] = np.bincount(species, minlength=n_species)
    energy_bin = np.searchsorted(ENERGY_EDGES, kinetic_energy, side="right") - 1
    in_range = (energy_bin >= 0) & (energy_bin < n_energies)
    summary["spectrum"][:] = np.bincount(
        species[in_range] * n_energies + energy_bin[in_range],
        minlength=n_species * n_energies,
    ).reshape(n_species, n_energies)

    for f, frame in enumerate(FRAMES):
        axis = axes.get(frame)
        if not _valid_axis(axis):
            for name in ["lateral", "delay_sum", "delay_sum2", "at_reference"]:
                summary[name][f] = np.nan
            continue
        point, direction, axis_time = axis
        times = time if time is not None else np.full(len(species), np.nan)
        radius, delay = axis_coordinates(position, times, point, direction, axis_time)
        radius_bin = np.searchsorted(RADIUS_EDGES, radius, side="right") - 1
        in_range = radius_bin < n_radii
        index = species[in_range] * n_radii + radius_bin[in_range]
        size = n_species * n_radii
        summary["lateral"][f] = np.bincount(index, minlength=size).reshape(
            n_species, n_radii
        )
        if time is None or not np.isfinite(axis_time):
            summary["delay_sum"][f] = np.nan
            summary["delay_sum2"][f] = np.nan
        else:
            delay = delay[in_range]
            summary["delay_sum"][f] = np.bincount(
                index, weights=delay, minlength=size
            ).reshape(n_species, n_radii)
            summary["delay_sum2"][f] = np.bincount(
                index, weights=delay * delay, minlength=size
            ).reshape(n_species, n_radii)
        inner, outer = settings.reference_annulus(frame)
        near = (radius >= inner) & (radius < outer)
        summary["at_reference"][f] = np.bincount(species[near], minlength=n_species)
    return summary


# ------------------------------------------------------------ populations


class GroundSummaries:
    """
    What a population of subshowers puts on the ground: each subshower's
    start particle, the primary of its shower, and the histograms summarise
    gives. Like responses.Responses, but for the ground rather than a
    detector array, so it can be made from the dataset (from_source) or from
    a generative model's subshowers (from_samples), and compared.
    """

    # one value per subshower, and its type
    scalar_columns = {
        "source": str,
        "subshower_number": np.int64,  # within its source; -1 if unknown
        "start_pdg": np.int64,
        "start_kinetic_energy": np.float64,
        "start_time": np.float64,
        "primary_time": np.float64,  # nan if unknown
    }
    # three values per subshower; the primary's are nan if unknown
    vector_columns = [
        "start_position",
        "start_direction",
        "primary_position",
        "primary_direction",
    ]
    FORMAT_VERSION = 1

    def __init__(self, values, histograms, settings=None, metadata=None):
        """
        Parameters
        ----------
        values : dict
            An array for each of scalar_columns and vector_columns, with
            one entry (or row of three) per subshower.
        histograms : dict
            An array for each of HISTOGRAMS, with one more dimension in
            front than summarise gives, for the subshowers.
        settings : GroundSettings
            The settings they were made with.
        metadata : dict
            Anything that can be written as json.
        """
        shapes = _histogram_shapes()
        n_subshowers = len(np.asarray(histograms["number"]))
        self.histograms = {}
        for name, (shape, dtype) in shapes.items():
            self.histograms[name] = np.asarray(histograms[name], dtype=dtype).reshape(
                (n_subshowers,) + shape
            )
        self.values = {}
        for name, kind in self.scalar_columns.items():
            self.values[name] = np.asarray(values[name], dtype=kind).reshape(
                n_subshowers
            )
        for name in self.vector_columns:
            self.values[name] = np.asarray(values[name], dtype=np.float64).reshape(
                n_subshowers, 3
            )
        self.settings = settings or GroundSettings()
        self.metadata = dict(metadata or {})

    def __len__(self):
        return len(self.histograms["number"])

    @property
    def reaches_ground(self):
        """True for subshowers that put some counted particle on the ground."""
        return self.histograms["number"].sum(axis=1) > 0

    def axes(self, frame):
        """(points, directions) of each subshower's axis in a frame."""
        prefix = {"subshower": "start", "shower": "primary"}[frame]
        return self.values[f"{prefix}_position"], self.values[f"{prefix}_direction"]

    # ----------------------------------------------------------- building

    @classmethod
    def _build(cls, rows, settings, metadata):
        """From a list of (values, histograms) for each subshower."""
        names = list(cls.scalar_columns) + cls.vector_columns
        values = {name: [row[0][name] for row in rows] for name in names}
        histograms = {}
        for name, (shape, dtype) in _histogram_shapes().items():
            if rows:
                histograms[name] = np.stack([row[1][name] for row in rows])
            else:
                histograms[name] = np.zeros((0,) + shape, dtype=dtype)
        if not rows:
            values = {
                name: np.zeros((0, 3)) if name in cls.vector_columns else []
                for name in names
            }
        return cls(values, histograms, settings, metadata)

    @classmethod
    def from_samples(cls, samples, settings=None, source=None, metadata=None):
        """
        Summaries of subshowers given as particles on the ground, such as a
        generative model makes. Each sample needs

        * start_pdg, start_kinetic_energy, start_time, start_position and
          start_direction, as for responses.Responses.from_samples;
        * ground_pdg, ground_x, ground_y and ground_kinetic_energy, with a
          value for each particle reaching the ground (at settings.height),
          and ground_time if arrival times are known (in seconds, the
          history's units);
        * primary_position, primary_direction and primary_time, for
          distances and delays about the shower's axis; without them, those
          histograms are nan.

        source and subshower_number are kept, as in Responses.from_samples.
        """
        settings = settings or GroundSettings()
        rows = []
        for sample in samples:
            if source is None:
                name = getattr(sample, "source", "")
                name = name if isinstance(name, str) else getattr(name, "name", "")
            else:
                name = source
            values = {
                "source": name,
                "subshower_number": getattr(sample, "subshower_number", -1),
                "start_pdg": sample.start_pdg,
                "start_kinetic_energy": sample.start_kinetic_energy,
                "start_time": sample.start_time,
                "start_position": sample.start_position,
                "start_direction": sample.start_direction,
                "primary_time": getattr(sample, "primary_time", np.nan),
                "primary_position": getattr(
                    sample, "primary_position", np.full(3, np.nan)
                ),
                "primary_direction": getattr(
                    sample, "primary_direction", np.full(3, np.nan)
                ),
            }
            x = np.asarray(sample.ground_x, dtype=float).reshape(-1)
            y = np.asarray(sample.ground_y, dtype=float).reshape(-1)
            histograms = summarise(
                sample.ground_pdg,
                np.column_stack((x, y)),
                sample.ground_kinetic_energy,
                getattr(sample, "ground_time", None),
                axes=cls._axes_of(values),
                settings=settings,
            )
            rows.append((values, histograms))
        return cls._build(rows, settings, metadata)

    @staticmethod
    def _axes_of(values):
        return {
            "subshower": (
                values["start_position"],
                values["start_direction"],
                values["start_time"],
            ),
            "shower": (
                values["primary_position"],
                values["primary_direction"],
                values["primary_time"],
            ),
        }

    @classmethod
    def from_source(
        cls, source, subshower_numbers, settings=None, metadata=None, progress=None
    ):
        """
        Summaries of some of a sampling.DatasetSource's subshowers, picked
        by their number within it (as Responses.values["subshower_number"]
        records them), in that order. progress, if given, wraps the numbers
        as they're used, e.g. to print how far it has got.
        """
        settings = settings or GroundSettings()
        reader = source.reader
        columns, positions_of = _history_columns(reader)
        roots = np.flatnonzero(reader.data["parent_label"].to_numpy() == -1)
        primary = {
            "primary_time": np.nan,
            "primary_position": np.full(3, np.nan),
            "primary_direction": np.full(3, np.nan),
        }
        if len(roots):
            root = roots[0]
            primary = {
                "primary_time": float(columns["time"][root]),
                "primary_position": np.array([columns[c][root] for c in "xyz"]),
                "primary_direction": np.array(
                    [columns[c][root] for c in ["px", "py", "pz"]]
                ),
            }
        numbers = np.asarray(subshower_numbers, dtype=np.int64).reshape(-1)
        rows = []
        for number in progress(numbers) if progress is not None else numbers:
            start_index, leaf_indices = source[int(number)]
            start = int(np.atleast_1d(positions_of(start_index))[0])
            values = {
                "source": source.name,
                "subshower_number": int(number),
                "start_pdg": columns["pdg"][start],
                "start_kinetic_energy": columns["kinetic_energy"][start],
                "start_time": columns["time"][start],
                "start_position": np.array([columns[c][start] for c in "xyz"]),
                "start_direction": np.array(
                    [columns[c][start] for c in ["px", "py", "pz"]]
                ),
                **primary,
            }
            leaves = positions_of(np.asarray(leaf_indices))
            pdg, ground_position, kinetic_energy, ground_time = leaves_to_ground(
                columns["pdg"][leaves],
                np.stack([columns[c][leaves] for c in "xyz"], axis=-1),
                np.stack([columns[c][leaves] for c in ["px", "py", "pz"]], axis=-1),
                columns["kinetic_energy"][leaves],
                columns["time"][leaves],
                settings.height,
            )
            histograms = summarise(
                pdg,
                ground_position,
                kinetic_energy,
                ground_time,
                axes=cls._axes_of(values),
                settings=settings,
            )
            rows.append((values, histograms))
        return cls._build(rows, settings, metadata)

    @classmethod
    def concatenate(cls, parts, metadata=None):
        """All the subshowers of several GroundSummaries made with one setting."""
        parts = list(parts)
        if not parts:
            raise ValueError("nothing to concatenate")
        settings = parts[0].settings
        if any(part.settings.as_dict() != settings.as_dict() for part in parts):
            raise ValueError("the parts were made with different ground settings")
        return cls(
            {
                name: np.concatenate([part.values[name] for part in parts])
                for name in parts[0].values
            },
            {
                name: np.concatenate([part.histograms[name] for part in parts])
                for name in HISTOGRAMS
            },
            settings,
            metadata,
        )

    def select(self, which):
        """The subshowers a boolean mask, or array of indices, picks out."""
        indices = np.arange(len(self))[which]
        return GroundSummaries(
            {name: column[indices] for name, column in self.values.items()},
            {name: array[indices] for name, array in self.histograms.items()},
            self.settings,
            self.metadata,
        )

    # ------------------------------------------------------- saving, loading

    def save(self, path):
        """Write to a .npz file, which load reads back."""
        arrays = {
            "format_version": np.array(self.FORMAT_VERSION),
            "settings": np.array(json.dumps(self.settings.as_dict())),
            "metadata": np.array(json.dumps(self.metadata)),
            "species": np.array(json.dumps(SPECIES)),
            "frames": np.array(json.dumps(FRAMES)),
            "radius_edges": RADIUS_EDGES,
            "energy_edges": ENERGY_EDGES,
        }
        for name, column in self.values.items():
            arrays[f"value_{name}"] = column
        for name, array in self.histograms.items():
            arrays[f"histogram_{name}"] = array
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
            if version != cls.FORMAT_VERSION:
                raise ValueError(
                    f"{path} has format version {version}, not {cls.FORMAT_VERSION}"
                )
            same_bins = (
                json.loads(data["species"].item()) == SPECIES
                and json.loads(data["frames"].item()) == FRAMES
                and np.array_equal(data["radius_edges"], RADIUS_EDGES)
                and np.array_equal(data["energy_edges"], ENERGY_EDGES)
            )
            if not same_bins:
                raise ValueError(f"{path} was made with other bins or species")
            names = list(cls.scalar_columns) + cls.vector_columns
            return cls(
                {name: data[f"value_{name}"] for name in names},
                {name: data[f"histogram_{name}"] for name in HISTOGRAMS},
                GroundSettings(**json.loads(data["settings"].item())),
                json.loads(data["metadata"].item()),
            )

    @classmethod
    def read_metadata(cls, path):
        """The metadata of a saved GroundSummaries, without reading the rest."""
        with np.load(path, allow_pickle=False) as data:
            return json.loads(data["metadata"].item())


def check_aligned(summaries, responses):
    """
    Raise ValueError unless a GroundSummaries and a responses.Responses are
    of the same subshowers, in the same order, starting in the same places:
    the detector signal about each subshower's axis needs its deposits
    measured where the summary has it, so not shifted across the ground (as
    sampling.SubshowerSampler does, unless given shift=False) for one and
    not the other.
    """
    if len(summaries) != len(responses):
        raise ValueError(
            f"{len(summaries)} ground summaries, but {len(responses)} responses"
        )
    numbers = summaries.values["subshower_number"]
    other = responses.values["subshower_number"]
    known = (numbers >= 0) & (other >= 0)
    if np.any(numbers[known] != other[known]) or np.any(
        summaries.values["source"][known] != responses.values["source"][known]
    ):
        raise ValueError(
            "the ground summaries and responses are of different subshowers"
        )
    if not np.allclose(
        summaries.values["start_position"],
        responses.values["start_position"],
        rtol=0.0,
        atol=1e-6,
        equal_nan=True,
    ):
        raise ValueError(
            "the ground summaries and responses have subshowers starting in"
            " different places, as if the responses were of shifted subshowers"
        )


# ----------------------------------------------------------- the detectors


def detector_areas(array):
    """The area (m²) of each detector in an array."""
    if isinstance(array, _detector.Triangles):
        edges = np.asarray(array.edges, dtype=float)
        cross = np.cross(edges[:, 1] - edges[:, 0], edges[:, 2] - edges[:, 0])
        return 0.5 * np.linalg.norm(cross, axis=-1)
    if isinstance(array, _detector.HorizontalCircles):
        return np.pi * np.asarray(array.radii2, dtype=float)
    return np.ones(len(array))


def signal_about_axes(responses, array, points, directions, settings, frame):
    """
    What the detectors read about each subshower's axis: the energy
    deposited (GeV) in the detectors in each bin of distance from the axis
    (RADIUS_EDGES), the area (m²) of those detectors, and the energy
    deposited per m² of the detectors within the reference annulus of the
    frame (nan if none is).

    Returns deposit (n, R), area (n, R), at_reference (n,)
    """
    if len(array) != responses.n_detectors:
        raise ValueError(
            f"the responses were measured with {responses.n_detectors} detectors,"
            f" not {len(array)}"
        )
    centres = np.asarray(array.centers, dtype=float)
    areas = detector_areas(array)
    inner, outer = settings.reference_annulus(frame)
    n_subshowers, n_radii = len(responses), len(RADIUS_EDGES) - 1
    deposit = np.full((n_subshowers, n_radii), np.nan)
    area = np.full((n_subshowers, n_radii), np.nan)
    at_reference = np.full(n_subshowers, np.nan)
    offsets = responses.hit_offsets
    for i in range(n_subshowers):
        point = np.asarray(points[i], dtype=float)
        direction = np.asarray(directions[i], dtype=float)
        norm = np.linalg.norm(direction)
        if not (np.all(np.isfinite(point)) and np.isfinite(norm) and norm > 0):
            continue
        radius = np.linalg.norm(np.cross(centres - point, direction / norm), axis=-1)
        radius_bin = np.searchsorted(RADIUS_EDGES, radius, side="right") - 1
        in_range = radius_bin < n_radii
        area[i] = np.bincount(
            radius_bin[in_range], weights=areas[in_range], minlength=n_radii
        )
        hit = responses.hit_detectors[offsets[i] : offsets[i + 1]]
        energy = responses.hit_energies[offsets[i] : offsets[i + 1]]
        counted = in_range[hit]
        deposit[i] = np.bincount(
            radius_bin[hit][counted], weights=energy[counted], minlength=n_radii
        )
        near = (radius >= inner) & (radius < outer)
        near_area = areas[near].sum()
        if near_area > 0:
            at_reference[i] = energy[near[hit]].sum() / near_area
    return deposit, area, at_reference
