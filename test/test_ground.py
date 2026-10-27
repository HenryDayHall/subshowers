"""
Tests for subshowers.ground: carrying leaves to the ground, measuring them
about an axis, summarising them, and the GroundSummaries that holds the
summaries of many subshowers.
"""

import os
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from subshowers import detector, ground, subshowers
from subshowers.ground import GroundSettings, GroundSummaries
from subshowers.responses import Responses
from subshowers.sampling import DatasetSource, SubshowerSampler
from fake_data import fake_history, grow_cascade

C = ground.SPEED_OF_LIGHT
DOWN = np.array([0.0, 0.0, -1.0])
TOP = np.array([0.0, 0.0, 1000.0])


def radius_bin(radius):
    """The bin of ground.RADIUS_EDGES a distance falls in, found by hand."""
    edges = ground.RADIUS_EDGES
    return next(b for b in range(len(edges) - 1) if edges[b] <= radius < edges[b + 1])


# ---------------------------------------------------------------- particles


def test_species_and_masses():
    pdg = [11, -11, 22, 13, -13, 211, 2212, 12, -14, 16, 1000260560]
    assert ground.species_of(pdg).tolist() == [0, 0, 1, 2, 2, 3, 3, -1, -1, -1, 3]
    assert ground.SPECIES[0] == "electrons" and ground.SPECIES[3] == "hadrons"
    masses = ground.masses_of([22, -13, 2212, 1000260560, 99999])
    assert masses[0] == 0
    assert masses[1] == pytest.approx(0.105658)
    assert masses[2] == pytest.approx(0.938272)
    assert masses[3] == pytest.approx(56 * ground.NUCLEON_MASS)  # iron 56
    assert np.isnan(masses[4])


def test_speeds():
    muon_mass = ground.MASSES[13]
    speeds = ground.speeds([22, 13, 11, 13], [1.0, muon_mass, 1e3, 0.0])
    assert speeds[0] == 1.0
    # kinetic energy equal to the mass: gamma 2, so beta is sqrt(3) / 2
    assert speeds[1] == pytest.approx(np.sqrt(3) / 2)
    assert speeds[2] == pytest.approx(1.0)
    assert speeds[3] == 0.0  # at rest
    # unknown particles: from the momentum if given, else the speed of light
    assert ground.speeds([99999], [1.0])[0] == 1.0
    # a mass of 1 GeV and a kinetic energy of 1 GeV: p = sqrt(3), E = 2
    p = np.sqrt(3)
    assert ground.speeds([99999], [1.0], [[0, 0, -p]])[0] == pytest.approx(p / 2)


def test_to_ground():
    position = [
        [10, 20, 1000],  # straight down
        [0, 0, 1000],  # down at 45 degrees, half the speed of light
        [0, 0, 1000],  # up
        [0, 0, -5],  # below the ground
        [0, 0, 1000],  # no momentum
        [0, 0, 1000],  # at rest
        [3, 4, 0],  # on the ground, going down
    ]
    momentum = [DOWN, [1, 0, -1], -DOWN, DOWN, [0, 0, 0], DOWN, DOWN]
    time = [1e-6, 0, 0, 0, 0, 0, 2e-6]
    speed = [1, 0.5, 1, 1, 1, 0, 1]
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no warnings about dividing by 0
        reaches, where, when = ground.to_ground(position, momentum, time, speed)
    assert reaches.tolist() == [True, True, False, False, False, False, True]
    np.testing.assert_allclose(where, [[10, 20, 0], [1000, 0, 0], [3, 4, 0]])
    np.testing.assert_allclose(
        when, [1e-6 + 1000 / C, 1000 * np.sqrt(2) / (0.5 * C), 2e-6]
    )
    # a raised ground
    reaches, where, when = ground.to_ground(position[:1], momentum[:1], [0], [1], 400)
    np.testing.assert_allclose(where, [[10, 20, 400]])
    np.testing.assert_allclose(when, [600 / C])


def test_axis_coordinates():
    points = [[3, 4, 0], [0, 0, 0], [0, 0, 500]]
    times = [1000 / C + 7e-9, 1000 / C, 0]
    radius, delay = ground.axis_coordinates(points, times, TOP, DOWN * 2, 0.0)
    np.testing.assert_allclose(radius, [5, 0, 0], atol=1e-12)
    # the plane front passes the ground at 1000 / C, and z = 500 at 500 / C
    np.testing.assert_allclose(delay, [7, 0, -500 / C * 1e9], atol=1e-6)
    # an inclined axis: distance is measured in the shower plane
    slant = np.array([1.0, 0.0, -1.0])
    radius, _ = ground.axis_coordinates([[0, 0, 0]], [0], TOP, slant, 0.0)
    assert radius[0] == pytest.approx(1000 / np.sqrt(2))
    radius, delay = ground.axis_coordinates(points, times, TOP, [0, 0, 0], 0.0)
    assert np.all(np.isnan(radius)) and np.all(np.isnan(delay))


def test_leaves_to_ground():
    pdg, where, kinetic_energy, when = ground.leaves_to_ground(
        [22, 13, 22],
        [[0, 0, 1000], [0, 0, 1000], [0, 0, 1000]],
        [DOWN, DOWN, -DOWN],
        [1.0, ground.MASSES[13], 1.0],
        [0.0, 0.0, 0.0],
    )
    assert pdg.tolist() == [22, 13]
    assert kinetic_energy.tolist() == [1.0, ground.MASSES[13]]
    np.testing.assert_allclose(when, [1000 / C, 1000 / (C * np.sqrt(3) / 2)])


# ---------------------------------------------------------------- summaries


def particles():
    """Particles on the ground, with the bins they should land in."""
    return {
        # pdg, x, y, kinetic energy (GeV), delay (ns)
        "pdg": np.array([11, 22, -13, 13, 14, 2212, -11]),
        "x": np.array([0.0, 50, 100, 0, 5, 1000, 20_000]),
        "y": np.array([0.0, 0, 0, 100, 5, 0, 0]),
        "kinetic_energy": np.array([0.01, 0.5, 1.0, 0.1, 1.0, 2.0, 1.0]),
        "delay": np.array([5.0, 7, 30, 1, 1, 400, 9000]),
    }


def test_summarise():
    p = particles()
    times = 1000 / C + p["delay"] * 1e-9
    summary = ground.summarise(
        p["pdg"],
        np.column_stack((p["x"], p["y"])),
        p["kinetic_energy"],
        times,
        axes={"subshower": (TOP, DOWN, 0.0)},
    )
    assert set(summary) == set(ground.HISTOGRAMS)
    # the muon below 0.3 GeV and the neutrino aren't counted
    assert summary["number"].tolist() == [2, 1, 1, 1]
    spectrum = summary["spectrum"]
    assert spectrum.sum(axis=1).tolist() == [2, 1, 1, 1]
    for species, energy in [(0, 0.01), (1, 0.5), (2, 1.0), (3, 2.0)]:
        edges = ground.ENERGY_EDGES
        b = int(np.flatnonzero(spectrum[species])[0])
        assert edges[b] <= energy < edges[b + 1]

    lateral = summary["lateral"][0]
    # the positron 20 km out is past the last bin
    assert lateral.sum(axis=1).tolist() == [1, 1, 1, 1]
    for species, radius in [(0, 0.0), (1, 50.0), (2, 100.0), (3, 1000.0)]:
        assert lateral[species, radius_bin(radius)] == 1
    delay_sum, delay_sum2 = summary["delay_sum"][0], summary["delay_sum2"][0]
    for species, radius, delay in [(0, 0, 5), (1, 50, 7), (2, 100, 30), (3, 1000, 400)]:
        assert delay_sum[species, radius_bin(radius)] == pytest.approx(delay, abs=1e-3)
        assert delay_sum2[species, radius_bin(radius)] == pytest.approx(
            delay**2, rel=1e-3
        )
    # the reference distance about the subshower axis is 100 m
    assert summary["at_reference"][0].tolist() == [0, 0, 1, 0]
    # no shower axis was given
    for name in ["lateral", "delay_sum", "delay_sum2", "at_reference"]:
        assert np.all(np.isnan(summary[name][1]))

    # without times, there are no delays; with other settings, other counts
    settings = GroundSettings(
        min_kinetic_energy={"muons": 0.05}, reference_distance={"subshower": 50}
    )
    summary = ground.summarise(
        p["pdg"],
        np.column_stack((p["x"], p["y"])),
        p["kinetic_energy"],
        axes={"subshower": (TOP, DOWN, 0.0), "shower": (TOP, DOWN, 0.0)},
        settings=settings,
    )
    assert summary["number"].tolist() == [2, 1, 2, 1]
    assert np.all(np.isnan(summary["delay_sum"]))
    # the photon 50 m out, and the proton 1000 m out (about 800 m)
    assert summary["at_reference"].tolist() == [[0, 1, 0, 0], [0, 0, 0, 1]]
    np.testing.assert_array_equal(summary["lateral"][0], summary["lateral"][1])


def test_summarise_nothing():
    summary = ground.summarise(
        [], np.zeros((0, 2)), [], [], {"subshower": (TOP, DOWN, 0)}
    )
    assert summary["number"].sum() == 0
    assert np.nansum(summary["lateral"]) == 0


def ground_sample(rng, source="model", n_particles=20, primary=True):
    """A subshower as a generative model might give it: particles on the ground."""
    sample = SimpleNamespace(
        source=source,
        start_pdg=int(rng.choice([22, 11, -11])),
        start_kinetic_energy=float(rng.uniform(1, 100)),
        start_time=0.0,
        start_position=np.array([rng.normal(0, 10), rng.normal(0, 10), 1000.0]),
        start_direction=DOWN + rng.normal(0, 0.05, 3),
        ground_pdg=rng.choice([11, -11, 22, 13, 2212, 12], n_particles),
        ground_x=rng.normal(0, 50, n_particles),
        ground_y=rng.normal(0, 50, n_particles),
        ground_kinetic_energy=10 ** rng.uniform(-3, 1, n_particles),
        ground_time=1000 / C + rng.exponential(20e-9, n_particles),
    )
    if primary:
        sample.primary_position = np.array([0.0, 0.0, 100_000.0])
        sample.primary_direction = DOWN
        sample.primary_time = 0.0
    return sample


def test_from_samples_select_concatenate_save_load(tmp_path):
    rng = np.random.default_rng(1)
    samples = [ground_sample(rng) for _ in range(4)] + [
        ground_sample(rng, primary=False)
    ]
    summaries = GroundSummaries.from_samples(samples, metadata={"made": "here"})
    assert len(summaries) == 5
    assert set(summaries.values["source"]) == {"model"}
    assert summaries.values["subshower_number"].tolist() == [-1] * 5
    for i, sample in enumerate(samples):
        expected = ground.summarise(
            sample.ground_pdg,
            np.column_stack((sample.ground_x, sample.ground_y)),
            sample.ground_kinetic_energy,
            sample.ground_time,
            axes=GroundSummaries._axes_of(
                {
                    "start_position": sample.start_position,
                    "start_direction": sample.start_direction,
                    "start_time": sample.start_time,
                    "primary_position": getattr(sample, "primary_position", None),
                    "primary_direction": getattr(sample, "primary_direction", None),
                    "primary_time": getattr(sample, "primary_time", np.nan),
                }
            ),
        )
        for name in ground.HISTOGRAMS:
            np.testing.assert_array_equal(summaries.histograms[name][i], expected[name])
    # the last has no primary, so nothing about the shower's axis
    assert np.all(np.isnan(summaries.histograms["lateral"][4, 1]))
    assert np.all(np.isnan(summaries.values["primary_position"][4]))
    assert summaries.reaches_ground.all()

    picked = summaries.select([4, 0])
    np.testing.assert_array_equal(
        picked.histograms["spectrum"][0], summaries.histograms["spectrum"][4]
    )
    both = GroundSummaries.concatenate([summaries, picked])
    assert len(both) == 7
    np.testing.assert_array_equal(
        both.histograms["lateral"][6], summaries.histograms["lateral"][0]
    )
    other = GroundSummaries.from_samples(samples[:1], GroundSettings(height=5))
    with pytest.raises(ValueError, match="settings"):
        GroundSummaries.concatenate([summaries, other])

    path = str(tmp_path / "ground.npz")
    summaries.save(path)
    assert GroundSummaries.read_metadata(path) == {"made": "here"}
    loaded = GroundSummaries.load(path)
    assert loaded.settings.as_dict() == summaries.settings.as_dict()
    for name in summaries.values:
        np.testing.assert_array_equal(loaded.values[name], summaries.values[name])
    for name in ground.HISTOGRAMS:
        np.testing.assert_array_equal(
            loaded.histograms[name], summaries.histograms[name]
        )
    assert loaded.histograms["lateral"].dtype == np.float32

    GroundSummaries.from_samples([]).save(path)
    assert len(GroundSummaries.load(path)) == 0


def test_load_other_bins(tmp_path, monkeypatch):
    path = str(tmp_path / "ground.npz")
    summaries = GroundSummaries.from_samples([ground_sample(np.random.default_rng(2))])
    monkeypatch.setattr(ground, "RADIUS_EDGES", ground.RADIUS_EDGES * 2)
    summaries.save(path)
    monkeypatch.undo()
    with pytest.raises(ValueError, match="other bins"):
        GroundSummaries.load(path)


def test_settings():
    defaults = GroundSettings.from_config({"storage_path": "/somewhere"})
    assert defaults.height == 0.0
    assert defaults.min_kinetic_energy == ground.DEFAULT_MIN_KINETIC_ENERGY
    assert defaults.reference_distance == ground.DEFAULT_REFERENCE_DISTANCE
    config = {
        "ground": {
            "height": 1400,
            "min_kinetic_energy": {"muons": "1.0e+0"},
            "reference_distance": {"shower": 1000},
        }
    }
    settings = GroundSettings.from_config(config)
    assert settings.height == 1400.0
    assert settings.min_kinetic_energy["muons"] == 1.0
    assert settings.min_kinetic_energy["electrons"] == 5.0e-4
    assert settings.reference_distance == {"subshower": 100.0, "shower": 1000.0}
    assert GroundSettings(**settings.as_dict()).as_dict() == settings.as_dict()
    inner, outer = settings.reference_annulus("shower")
    assert inner == pytest.approx(1000 / 10**0.1) and outer == pytest.approx(
        1000 * 10**0.1
    )
    for bad, match in [
        ({"heigth": 0}, "heigth"),
        ({"height": "high"}, "height"),
        ({"min_kinetic_energy": {"pions": 1}}, "pions"),
        ({"min_kinetic_energy": {"muons": -1}}, "muons"),
        ({"min_kinetic_energy": [1]}, "min_kinetic_energy"),
        ({"reference_distance": {"shower": 0}}, "shower"),
    ]:
        with pytest.raises(ground.GroundConfigError, match=match):
            GroundSettings.from_config({"ground": bad})
    with pytest.raises(ground.GroundConfigError, match="mapping"):
        GroundSettings.from_config({"ground": [1]})


# ------------------------------------------------------------- the dataset


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    folder = str(tmp_path_factory.mktemp("showers") / "e1e5_0")
    fake_history(folder, grow_cascade(np.random.default_rng(1), 4.0e3, 170_000.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        subshowers.run(folder, 100.0, leaves_only=True)
    array = detector.generate_at_ground(300, 6000.0, 300.0, 20.0, seed=0)
    source = DatasetSource(
        folder, output_path=os.path.join(folder, "subshowers_e100.0.h5"), name="e1e5_0"
    )
    samples = list(SubshowerSampler([source], array, seed=1).iter_epoch())
    return source, array, Responses.from_samples(samples, len(array))


def test_from_source(dataset):
    source, array, responses = dataset
    numbers = responses.values["subshower_number"]
    summaries = GroundSummaries.from_source(source, numbers)
    ground.check_aligned(summaries, responses)
    assert summaries.values["source"].tolist() == ["e1e5_0"] * len(responses)
    for name in ["start_pdg", "start_kinetic_energy", "start_time"]:
        np.testing.assert_allclose(summaries.values[name], responses.values[name])
    np.testing.assert_allclose(
        summaries.values["start_position"], responses.values["start_position"]
    )
    data = source.reader.data
    root = data.index[data["parent_label"] == -1][0]
    for i in range(len(summaries)):
        np.testing.assert_allclose(
            summaries.values["primary_position"][i], data.loc[root, ["x", "y", "z"]]
        )
        assert summaries.values["primary_time"][i] == data.loc[root, "time"]

    # each subshower's leaves, carried down by hand
    for i in [0, len(numbers) // 2, len(numbers) - 1]:
        start, leaves = source[int(numbers[i])]
        rows = data.loc[leaves]
        values = {name: column[i] for name, column in summaries.values.items()}
        expected = ground.summarise(
            *ground.leaves_to_ground(
                rows["pdg"].to_numpy(),
                rows[["x", "y", "z"]].to_numpy(),
                rows[["px", "py", "pz"]].to_numpy(),
                rows["kinetic_energy"].to_numpy(),
                rows["time"].to_numpy(),
            ),
            axes=GroundSummaries._axes_of(values),
        )
        for name in ground.HISTOGRAMS:
            np.testing.assert_allclose(
                summaries.histograms[name][i], expected[name], equal_nan=True
            )
    # these fake showers stop above the ground, so every subshower reaches it
    assert summaries.reaches_ground.all()

    # in another order, they aren't the same subshowers as the responses
    reordered = GroundSummaries.from_source(source, numbers[::-1])
    with pytest.raises(ValueError, match="different subshowers"):
        ground.check_aligned(reordered, responses)
    with pytest.raises(ValueError, match="responses"):
        ground.check_aligned(summaries.select(slice(1, None)), responses)


def test_shifted_responses_not_aligned(dataset):
    """
    Responses to subshowers the sampler shifted across the ground don't go
    with summaries of them where they were simulated, as the signal about
    each one's axis would be measured about the wrong place.
    """
    source, array, responses = dataset
    sampler = SubshowerSampler([source], array, seed=1, shift=True)
    shifted = Responses.from_samples(sampler.iter_epoch(), len(array))
    numbers = shifted.values["subshower_number"]
    # the same subshowers, in the same order
    np.testing.assert_array_equal(numbers, responses.values["subshower_number"])
    summaries = GroundSummaries.from_source(source, numbers)
    ground.check_aligned(summaries, responses)
    with pytest.raises(ValueError, match="shifted"):
        ground.check_aligned(summaries, shifted)


# ------------------------------------------------------------- detectors


def test_detector_areas():
    triangle = detector.Triangles(np.array([[[0.0, 0, 0], [4, 0, 0], [0, 3, 0]]]))
    assert ground.detector_areas(triangle).tolist() == [6.0]
    circles = detector.HorizontalCircles(np.zeros((2, 3)), 10)
    np.testing.assert_allclose(ground.detector_areas(circles), [100 * np.pi] * 2)


def test_signal_about_axes():
    centres = np.array([[0.0, 0, 0], [100, 0, 0], [0, 95, 0], [3000, 0, 0]])
    array = detector.HorizontalCircles(centres, 5.0)
    deposits = [np.array([1.0, 2.0, 0.0, 4.0]), np.array([1.0, 0, 0, 0])]
    samples = [
        SimpleNamespace(
            start_pdg=22,
            start_kinetic_energy=10.0,
            start_time=0.0,
            start_position=TOP,
            start_direction=DOWN,
            energy_per_detector=deposit,
        )
        for deposit in deposits
    ]
    responses = Responses.from_samples(samples, len(array))
    points = np.array([TOP, TOP])
    directions = np.array([DOWN, [0.0, 0, 0]])  # the second has no axis
    deposit, area, at_reference = ground.signal_about_axes(
        responses, array, points, directions, GroundSettings(), "subshower"
    )
    circle = 25 * np.pi
    for radius, energy in [(0, 1.0), (100, 2.0), (95, 0.0), (3000, 4.0)]:
        assert deposit[0, radius_bin(radius)] == energy
        assert area[0, radius_bin(radius)] == pytest.approx(circle)
    assert deposit[0].sum() == 7.0
    # 95 m and 100 m are both within the annulus about 100 m
    assert at_reference[0] == pytest.approx(2.0 / (2 * circle))
    assert np.all(np.isnan(deposit[1])) and np.isnan(at_reference[1])
    with pytest.raises(ValueError, match="detectors"):
        ground.signal_about_axes(
            responses, detector.generate_pierre_auger("sd750"), points, directions,
            GroundSettings(), "subshower",
        )  # fmt: skip
