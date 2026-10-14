"""
Tests for grouping records of observables, and for plotting them when there
is little to plot.
"""

from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from subshowers import detector, observables  # noqa: E402
from subshowers.population_plotting import (  # noqa: E402
    GREY,
    OTHER,
    PALETTE,
    bin_edges,
    group_colours,
)
from subshowers.responses import Responses  # noqa: E402

ARRAY = detector.HorizontalCircles(
    np.array([[0.0, 0, 0], [30, 0, 0], [0, 40, 0], [-50, 10, 0]]), 5.0
)


def record_of(deposits, kinetic_energy=50.0):
    """A record of subshowers alike but for their deposits."""
    samples = [
        SimpleNamespace(
            start_pdg=22,
            start_kinetic_energy=kinetic_energy,
            start_time=0.0,
            start_position=np.array([0.0, 0.0, 2000.0]),
            start_direction=np.array([0.0, 0.0, -1.0]),
            energy_per_detector=np.asarray(deposit, dtype=float),
        )
        for deposit in deposits
    ]
    return observables.record_of(Responses.from_samples(samples, len(ARRAY)), ARRAY)


def test_group_colours():
    colours = group_colours(["cut 1000 GeV", OTHER, "cut 900 GeV", "cut 100 GeV"])
    assert list(colours) == ["cut 100 GeV", "cut 900 GeV", "cut 1000 GeV", OTHER]
    assert colours["cut 100 GeV"] == PALETTE[0]
    assert colours[OTHER] == GREY
    many = group_colours([f"group {i}" for i in range(10)])
    assert list(many.values()) == PALETTE + [GREY, GREY]
    # numbers work as names too, as gather_by_pdg makes them
    assert list(group_colours([22, -11, 11])) == [11, 22, -11]


def test_bin_edges():
    assert np.allclose(bin_edges([1, 100], log=True, n_edges=3), [1, 10, 100])
    assert np.allclose(bin_edges([0, 100], log=True, n_edges=3), [0, 50, 100])
    for values, log in [([5, 5], False), ([0, 0], False), ([5, 5], True), ([], True)]:
        edges = bin_edges(values, log)
        assert np.all(np.diff(edges) > 0)
        if values:
            assert edges[0] < values[0] < edges[-1]


def test_fold_split_pool():
    groups = ["a"] * 5 + ["b"] * 4 + ["c"] * 2 + ["d"] + [None]
    folded = observables.fold_small_groups(groups, max_groups=3)
    assert folded == ["a"] * 5 + ["b"] * 4 + [OTHER] * 3 + [None]
    assert observables.fold_small_groups(groups, max_groups=4) == groups

    record = record_of([[i, 0, 0, 0] for i in range(1, len(groups) + 1)])
    split = observables.split_record(record, folded)
    assert sorted(split) == sorted(["a", "b", OTHER])
    assert split["b"]["total_deposit_energy"] == [6.0, 7.0, 8.0, 9.0]
    assert observables.record_length(split[OTHER]) == 3
    pooled = observables.pool(split.values())
    assert observables.record_length(pooled) == len(groups) - 1


@pytest.mark.parametrize(
    "deposits",
    [
        [[0, 0, 0, 0]],  # one subshower, nothing detected
        [[0, 0, 0, 0]] * 5,  # nothing detected
        [[3, 0, 0, 0]] * 5,  # all the same
        [[3, 0, 0, 0]],  # just one detected
        [[3, 1, 0, 0], [0, 0, 0, 0], [1, 2, 4, 0], [0, 0, 0, 7]],
    ],
)
def test_plots_of_little_data(deposits):
    by_group = {"dataset": record_of(deposits), "model": record_of(deposits[:1], 70.0)}
    for function in [
        observables.observables_A,
        observables.observables_B,
        observables.observables_C,
        observables.distributions,
    ]:
        fig = function(by_group, title="test")
        fig.canvas.draw()
        plt.close(fig)
