"""
Tests for records of ground observables, and for drawing them when there is
little to draw.
"""

from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from subshowers import detector, ground, observables  # noqa: E402
from subshowers import ground_observables as go  # noqa: E402
from subshowers.ground import GroundSummaries  # noqa: E402
from subshowers.responses import Responses  # noqa: E402

ARRAY = detector.HorizontalCircles(
    np.array([[0.0, 0, 0], [30, 0, 0], [0, 100, 0], [-500, 10, 0]]), 5.0
)
C = ground.SPEED_OF_LIGHT


def samples_of(particles, kinetic_energy=50.0, times=True, primary=True):
    """
    Subshowers starting 2 km up, given (pdg, x, y, kinetic energy) of the
    particles each puts on the ground, and depositing in detector 0 if any
    particle lands within 10 m of it.
    """
    samples = []
    for i, (pdg, x, y, energy) in enumerate(particles):
        x, y = np.asarray(x, float), np.asarray(y, float)
        deposit = np.zeros(len(ARRAY))
        deposit[0] = np.sum(np.asarray(energy)[np.hypot(x, y) < 10])
        sample = SimpleNamespace(
            start_pdg=22,
            start_kinetic_energy=kinetic_energy * (i + 1),
            start_time=0.0,
            start_position=np.array([0.0, 0.0, 2000.0]),
            start_direction=np.array([0.0, 0.0, -1.0]),
            ground_pdg=np.asarray(pdg),
            ground_x=x,
            ground_y=y,
            ground_kinetic_energy=np.asarray(energy, float),
            energy_per_detector=deposit,
        )
        if times:
            sample.ground_time = 2000 / C + np.hypot(x, y) * 1e-10
        if primary:
            sample.primary_position = np.array([10.0, 0.0, 100_000.0])
            sample.primary_direction = np.array([0.0, 0.0, -1.0])
            sample.primary_time = 0.0
        samples.append(sample)
    return samples


def record_of(particles, frame="subshower", signal=True, **kwargs):
    samples = samples_of(particles, **kwargs)
    summaries = GroundSummaries.from_samples(samples)
    if not signal:
        return go.record_of(summaries, frame)
    responses = Responses.from_samples(samples, len(ARRAY))
    record = observables.record_of(responses, ARRAY)
    record.update(go.entries_of(summaries, frame, responses, ARRAY))
    return record


NOTHING = ([], [], [], [])
SOME = ([11, -11, 22, 13], [0, 50, 120, 90], [0, 0, 5, 0], [0.01, 0.1, 0.02, 2.0])
MANY = (
    [11] * 30 + [22] * 30 + [13] * 10 + [211] * 3,
    np.linspace(-300, 300, 73),
    np.linspace(-80, 80, 73),
    np.geomspace(1e-3, 50, 73),
)
BELOW_CUTS = ([13, 22], [5, 5], [0, 0], [0.1, 1e-4])


def test_entries_and_records():
    samples = samples_of([SOME, MANY])
    summaries = GroundSummaries.from_samples(samples)
    responses = Responses.from_samples(samples, len(ARRAY))
    record = go.record_of(summaries, "shower", responses, ARRAY)
    for key in observables.conditioning_values + [
        "ground_number",
        "ground_lateral",
        "signal_area",
        "signal_at_reference",
    ]:
        assert len(record[key]) == 2, key
    assert record["ground_number"][0].tolist() == [2, 1, 1, 0]
    np.testing.assert_array_equal(
        record["ground_lateral"][1], summaries.histograms["lateral"][1, 1]
    )
    # pooling and splitting keep the ground entries
    pooled = observables.pool([record, record])
    assert len(pooled["ground_spectrum"]) == 4
    split = observables.split_record(pooled, ["a", "b", "a", "b"])
    assert len(split["a"]["signal_deposit"]) == 2
    with pytest.raises(ValueError, match="frame"):
        go.entries_of(summaries, "sideways")
    with pytest.raises(ValueError, match="array"):
        go.entries_of(summaries, "subshower", responses)


def all_figures(by_group, frame, reference=None):
    distance = ground.DEFAULT_REFERENCE_DISTANCE[frame]
    return [
        go.lateral_distributions(by_group, frame, reference, distance, "test"),
        go.energy_spectra(by_group, reference, "test"),
        go.particle_numbers(by_group, "test"),
        go.arrival_times(by_group, frame, "test"),
        go.readings_at_reference(by_group, frame, distance, "test"),
    ]


@pytest.mark.parametrize("frame", ground.FRAMES)
@pytest.mark.parametrize(
    "dataset, model",
    [
        ([NOTHING], [NOTHING]),  # nothing reaches the ground
        ([BELOW_CUTS] * 3, [NOTHING]),  # nothing is counted
        ([SOME], [SOME]),  # one subshower each
        ([SOME, MANY, NOTHING, MANY], [MANY, SOME]),
    ],
)
def test_figures_of_little_data(frame, dataset, model):
    by_group = {"dataset": record_of(dataset, frame), "model": record_of(model, frame)}
    # with ratio panels; the plotting script's tests draw them without
    for fig in all_figures(by_group, frame, reference="dataset"):
        fig.canvas.draw()
        plt.close(fig)


def test_figures_without_signal_times_or_primary():
    """A model giving only particles on the ground, without times or primary."""
    by_group = {
        "dataset": record_of([SOME, MANY], "shower"),
        "model": record_of(
            [MANY, SOME], "shower", signal=False, times=False, primary=False
        ),
    }
    for fig in all_figures(by_group, "shower", reference="model"):
        fig.canvas.draw()
        plt.close(fig)
    with pytest.raises(ValueError, match="reference"):
        go.energy_spectra(by_group, reference="CORSIKA 8")


def test_means():
    values = np.array([[1.0, np.nan, 4.0], [3.0, np.nan, np.nan]])
    mean, error, count = go._mean_and_error(values)
    np.testing.assert_allclose(mean, [2.0, np.nan, 4.0])
    assert count.tolist() == [2, 0, 1]
    assert error[0] == pytest.approx(1.0)  # standard deviation sqrt(2), over sqrt(2)
    assert np.isnan(error[1]) and np.isnan(error[2])
    centres = go._centres(ground.RADIUS_EDGES)
    assert centres[0] == pytest.approx(0.05)
    assert centres[1] == pytest.approx(np.sqrt(0.1 * 0.1 * 10**0.1))
