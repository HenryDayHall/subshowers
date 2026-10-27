"""
Tests for subshowers.sampling.SubshowerSampler shifting subshowers across the
ground to augment the data, and for detector.spacing and detector.max_shift,
which say how far it shifts them for an array.
"""

import numpy as np
import pytest

from subshowers import detector, responses
from subshowers.sampling import DatasetSource, SubshowerSampler
from fake_data import COLUMNS, fake_history, grow_cascade

# a random array of large triangles, which the fake shower below falls on
TRIANGLES = detector.generate_at_ground(300, 6000.0, 300.0, 20.0, seed=0)
# and circles in the same places
CIRCLES = detector.HorizontalCircles(TRIANGLES.centers, 150.0)
# three small circles for the toy shower below: two 30 m apart on the x axis,
# and one on the y axis
TOY_ARRAY = detector.HorizontalCircles(
    np.array([[100.0, 0.0, 0.0], [130.0, 0.0, 0.0], [0.0, 100.0, 0.0]]), 10.0
)


# ------------------------------------------------------------------ helpers


def grid(spacing, triangular=False, angle=0.0, offset=(0.0, 0.0), size=9):
    """Centres of detectors on a square or triangular grid, turned by angle."""
    i, j = np.meshgrid(np.arange(size, dtype=float), np.arange(size, dtype=float))
    i, j = i.ravel(), j.ravel()
    if triangular:
        x, y = spacing * (i + j / 2), spacing * j * np.sqrt(3) / 2
    else:
        x, y = spacing * i, spacing * j
    cos, sin = np.cos(angle), np.sin(angle)
    x, y = cos * x - sin * y + offset[0], sin * x + cos * y + offset[1]
    return np.column_stack((x, y, np.zeros(len(x))))


def moved(array, shift):
    """The array, moved across the ground by shift."""
    if isinstance(array, detector.Triangles):
        return detector.Triangles(array.edges + shift)
    return detector.HorizontalCircles(array.centers + shift, array.radius)


def toy_history():
    """
    A primary making two particles below 100 GeV, so two subshowers, whose
    leaves come straight down from 2 km up: at x = 0 and 30 m, with 10 and
    20 GeV, below the first (50 GeV), and at x = 500 m below the second.
    """
    rows = {column: [] for column in COLUMNS}

    def add(parent, x, energy, z=2000.0):
        label = len(rows["label"])
        values = [0, 22, 0.0, 0.0, -1.0, x, 0.0, z, energy, 0.0, label, parent]
        for column, value in zip(COLUMNS, values):
            rows[column].append(value)
        return label

    primary = add(-1, 0.0, 1000.0, z=10_000.0)
    first = add(primary, 0.0, 50.0, z=5000.0)
    add(first, 0.0, 10.0)
    add(first, 30.0, 20.0)
    second = add(primary, 500.0, 70.0, z=5000.0)
    add(second, 500.0, 5.0)
    return {column: np.array(values) for column, values in rows.items()}


@pytest.fixture
def toy(tmp_path):
    folder = str(tmp_path / "toy")
    fake_history(folder, toy_history())
    return DatasetSource(folder, energy_cut_value=100.0, name="toy")


@pytest.fixture(scope="module")
def cascade(tmp_path_factory):
    """A fake shower, coming down near the middle of TRIANGLES."""
    folder = str(tmp_path_factory.mktemp("showers") / "e3e3_0")
    fake_history(folder, grow_cascade(np.random.default_rng(2), 3.0e3, 20_000.0))
    return DatasetSource(folder, energy_cut_value=100.0)


# ---------------------------------------------------------- the array's part


@pytest.mark.parametrize("triangular", [False, True], ids=["square", "triangular"])
def test_spacing_of_grids(triangular):
    circles = detector.HorizontalCircles(grid(1500.0, triangular), 10.0)
    assert detector.spacing(circles) == pytest.approx(1500.0)
    # wherever the grid is, however it's turned, and however high
    centres = grid(1500.0, triangular, angle=0.7, offset=(-3.0e4, 2.0e4))
    centres[:, 2] = np.random.default_rng(0).uniform(1300.0, 1600.0, len(centres))
    turned = detector.HorizontalCircles(centres, 10.0)
    assert detector.spacing(turned) == pytest.approx(1500.0)
    # detectors in the same place, like replaced stations, are counted once
    doubled = detector.HorizontalCircles(
        np.concatenate((circles.centers, circles.centers[:30])), 10.0
    )
    assert detector.spacing(doubled) == pytest.approx(1500.0)
    # and the spacing is how far subshowers are shifted
    assert detector.max_shift(circles) == detector.spacing(circles)


def test_spacing_of_random_arrays():
    array = detector.generate_at_ground(200, 3000.0, seed=1)
    spacing = detector.spacing(array)
    # of n detectors spread at random over a disc of radius r, half are
    # within r sqrt(ln 2 / n) of the next, but for those at the edge
    expected = 3000.0 * np.sqrt(np.log(2) / 200)
    assert 0.85 * expected < spacing < 1.2 * expected
    # an array 2.5 times the size has 2.5 times the spacing
    bigger = detector.Triangles(array.edges * 2.5)
    assert detector.spacing(bigger) == pytest.approx(2.5 * spacing)


@pytest.mark.parametrize(
    "for_array, expected",
    [(None, 1500.0), ("stop", 1500.0), ("sd1500", 1500.0), ("sd750", 750.0)],
)
def test_spacing_of_pierre_auger(for_array, expected):
    array = detector.generate_pierre_auger(for_array)
    assert detector.max_shift(array) == pytest.approx(expected, rel=0.02)


def test_spacing_needs_two_places():
    two = detector.HorizontalCircles(np.array([[0.0, 0.0, 0.0], [30.0, 40.0, 0.0]]), 1)
    assert detector.spacing(two) == pytest.approx(50.0)
    # one detector, or two at the same place but for their height
    lone = detector.HorizontalCircles(np.array([[0.0, 0.0, 0.0]]), 10.0)
    stacked = detector.HorizontalCircles(np.array([[5.0, 5, 0], [5.0, 5, 100]]), 10)
    for array in [lone, stacked]:
        with pytest.raises(ValueError, match="1 place"):
            detector.max_shift(array)


# ------------------------------------------------------------ the sampler


def test_shifted_samples(toy):
    sampler = SubshowerSampler(toy, TOY_ARRAY, seed=0)
    assert not sampler.shift and sampler.max_shift == 0
    start, leaves = toy[0]
    assert start == 1 and list(leaves) == [2, 3]

    # where it was simulated, the first subshower misses every detector
    sample = sampler.make_sample(0, 0)
    assert sample.n_hit_detectors == 0
    np.testing.assert_array_equal(sample.shift, [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(sample.start_position, [0.0, 0.0, 5000.0])

    # moved 100 m along x, its leaves fall on the two detectors there
    sample = sampler.make_sample(0, 0, shift=(100.0, 0.0))
    np.testing.assert_array_equal(sample.energy_per_detector, [10.0, 20.0, 0.0])
    assert sorted(sample.detected_leaf_indices) == [2, 3]
    np.testing.assert_array_equal(sample.shift, [100.0, 0.0, 0.0])
    np.testing.assert_array_equal(sample.start_position, [100.0, 0.0, 5000.0])
    np.testing.assert_array_equal(sample.start_direction, [0.0, 0.0, -1.0])
    assert sample.start_kinetic_energy == 50.0
    assert sample.total_leaf_energy == 30.0

    # 95 m along y, one leaf lands 5 m from the third detector's centre
    sample = sampler.make_sample(0, 0, shift=(0.0, 95.0, 0.0))
    np.testing.assert_array_equal(sample.energy_per_detector, [0.0, 0.0, 10.0])
    assert list(sample.detected_leaf_indices) == [2]
    # 89 m along x, both land 11 m from a centre, just outside
    assert sampler.make_sample(0, 0, shift=[89.0, 0.0]).n_hit_detectors == 0
    # while the history stays where it was
    assert toy.reader.data["x"].tolist() == [0.0, 0.0, 0.0, 30.0, 500.0, 500.0]

    # only across the ground
    for bad in [(0.0, 0.0, 5.0), (1.0, 2.0, 0.0, 4.0), (np.nan, 0.0), 3.0]:
        with pytest.raises(ValueError, match="dx, dy"):
            sampler.make_sample(0, 0, shift=bad)


def test_random_shifts(toy):
    sampler = SubshowerSampler(toy, TOY_ARRAY, seed=0, shift=True, max_shift=100.0)
    shifts = np.array([sampler.random_shift() for _ in range(20_000)])
    assert np.all(shifts[:, 2] == 0)
    distances = np.linalg.norm(shifts, axis=1)
    assert distances.max() <= 100.0
    # spread evenly over the disc: a quarter of them within half its
    # radius, 19% beyond nine tenths of it, and as many each way
    assert np.mean(distances < 50.0) == pytest.approx(0.25, abs=0.015)
    assert np.mean(distances > 90.0) == pytest.approx(0.19, abs=0.015)
    np.testing.assert_allclose(shifts[:, :2].mean(axis=0), 0.0, atol=1.5)
    quadrant = 2 * (shifts[:, 0] > 0) + (shifts[:, 1] > 0)
    np.testing.assert_allclose(np.bincount(quadrant) / len(shifts), 0.25, atol=0.015)

    # every sample is shifted, by a shift of its own
    samples = [sampler[0] for _ in range(20)]
    assert len({tuple(sample.shift) for sample in samples}) == 20
    for sample in samples:
        assert 0 < np.linalg.norm(sample.shift) <= 100.0
        np.testing.assert_allclose(sample.start_position, [0, 0, 5000] + sample.shift)


def test_max_shift_settings(toy):
    # by default, how far the array says
    sampler = SubshowerSampler(toy, TOY_ARRAY, shift=True)
    assert sampler.max_shift == detector.max_shift(TOY_ARRAY) == pytest.approx(30.0)
    # or as far as asked
    sampler = SubshowerSampler(toy, TOY_ARRAY, shift=True, max_shift=250)
    assert sampler.max_shift == 250.0
    sampler = SubshowerSampler(toy, TOY_ARRAY, shift=True, max_shift=0)
    assert not sampler.make_sample(0, 0).shift.any()
    # an array of one detector has no spacing, so needs max_shift
    lone = detector.HorizontalCircles(np.array([[0.0, 0.0, 0.0]]), 10.0)
    with pytest.raises(ValueError, match="no spacing"):
        SubshowerSampler(toy, lone, shift=True)
    assert SubshowerSampler(toy, lone, shift=True, max_shift=50).max_shift == 50.0


@pytest.mark.parametrize(
    "settings, match",
    [
        ({"max_shift": 100.0}, "shift=True"),
        ({"shift": True, "max_shift": -1.0}, "at least 0"),
        ({"shift": True, "max_shift": np.inf}, "at least 0"),
        ({"shift": True, "max_shift": "far"}, "at least 0"),
    ],
)
def test_bad_shift_settings(toy, settings, match):
    with pytest.raises(ValueError, match=match):
        SubshowerSampler(toy, TOY_ARRAY, **settings)


@pytest.mark.parametrize("array", [TRIANGLES, CIRCLES], ids=["triangles", "circles"])
def test_shifting_subshowers_is_moving_the_array_back(cascade, array):
    """
    What the array sees of a shifted subshower is what it would see of the
    subshower where it was simulated, were the array moved the other way.
    """
    sampler = SubshowerSampler(cascade, array, seed=4, shift=True)
    assert sampler.max_shift == pytest.approx(detector.spacing(TRIANGLES))
    reader = cascade.reader
    n_detected = 0
    for sample in sampler.iter_epoch():
        assert 0 < np.linalg.norm(sample.shift) <= sampler.max_shift
        start, leaves = cascade[sample.subshower_number]
        np.testing.assert_allclose(
            sample.energy_per_detector,
            responses.sample_energy(reader, leaves, moved(array, -sample.shift)),
        )
        np.testing.assert_allclose(
            sample.start_position - sample.shift,
            reader.data.loc[start, ["x", "y", "z"]].to_numpy(dtype=float),
        )
        n_detected += sample.n_hit_detectors > 0
    assert n_detected >= 10


def test_shifts_keep_the_order_and_change_each_epoch(cascade):
    plain = list(SubshowerSampler(cascade, TRIANGLES, seed=7).iter_epoch())
    sampler = SubshowerSampler(cascade, TRIANGLES, seed=7, shift=True)
    first = list(sampler.iter_epoch())
    # the same subshowers in the same order, with shifts or without
    numbers = [sample.subshower_number for sample in plain]
    assert [sample.subshower_number for sample in first] == numbers
    data = cascade.reader.data
    for sample in plain:
        assert not sample.shift.any()
        np.testing.assert_array_equal(
            sample.start_position,
            data.loc[sample.start_index, ["x", "y", "z"]].to_numpy(dtype=float),
        )
    # with the same seed, the same shifts
    again = SubshowerSampler(cascade, TRIANGLES, seed=7, shift=True).iter_epoch()
    np.testing.assert_array_equal(
        [sample.shift for sample in again], [sample.shift for sample in first]
    )
    # and every epoch, new ones
    shifts = {sample.subshower_number: sample.shift for sample in first}
    for sample in sampler.iter_epoch():
        assert not np.array_equal(sample.shift, shifts[sample.subshower_number])


def test_shifts_change_the_detectors_reached(cascade):
    sampler = SubshowerSampler(cascade, TRIANGLES, seed=5, shift=True)
    reached = {}  # {subshower: set of the detectors it reached, each epoch}
    for _ in range(6):
        for sample in sampler.iter_epoch(detected_only=True):
            assert sample.n_hit_detectors > 0
            hit = tuple(np.flatnonzero(sample.energy_per_detector))
            reached.setdefault(sample.subshower_number, set()).add(hit)
    assert len(reached) >= 10
    # most of the subshowers reached other detectors in other epochs
    varied = sum(len(hits) > 1 for hits in reached.values())
    assert varied > len(reached) / 2
