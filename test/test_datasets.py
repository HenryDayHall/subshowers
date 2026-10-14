"""Tests for subshowers.datasets: which of a config's subshower sets exist."""

import warnings

import numpy as np
import pytest

from subshowers import datasets, subshowers
from fake_data import fake_history, grow_cascade


def make_set(folder, cut, leaves_only=True):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        subshowers.run(
            str(folder),
            cut,
            leaves_only=leaves_only,
            output_path=str(folder / f"subshowers_e{cut}.h5"),
        )


@pytest.fixture
def config(tmp_path):
    """
    e1e5_0 is made, with cut 100 ready, cut 500 made keeping every particle,
    cut 900 half written and cut 50 not made. e1e6_0 is still running,
    e1e7_0 finished without a history, and e1e8_0 isn't made.
    """
    storage = tmp_path / "data"
    showers = storage / "PDG22"
    made = showers / "e1e5_0"
    fake_history(str(made), grow_cascade(np.random.default_rng(1), 2.0e3, 170_000.0))
    make_set(made, 100)
    make_set(made, 500, leaves_only=False)
    (made / "subshowers_e900.h5").write_text("half written")
    running = showers / "e1e6_0"
    running.mkdir()
    (running / "summary.yaml").write_text("start time: 01/01/2026 10:00:00\n")
    no_history = showers / "e1e7_0"
    no_history.mkdir()
    (no_history / "summary.yaml").write_text(
        "start time: 01/01/2026 10:00:00\nend time: 01/01/2026 10:20:00\n"
    )
    return {
        "storage_path": f"{storage}/",
        "showers": {
            "folder": "PDG22",
            "subfolders": ["e1e5_0", "e1e6_0", "e1e7_0", "e1e8_0"],
        },
        "subshowers": {"energy_cuts": [100, 500, 900, 50], "leaves_only": True},
    }


def problems_of(checks):
    return {
        (check.subshower_set.shower.subfolder, check.subshower_set.energy_cut): (
            check.problem
        )
        for check in checks
    }


def test_check_subshower_sets(config):
    checks = datasets.check_subshower_sets(config)
    assert len(checks) == 16
    problems = problems_of(checks)
    assert problems[("e1e5_0", 100)] is None
    assert problems[("e1e5_0", 500)].startswith("made with other settings")
    assert "leaves only False" in problems[("e1e5_0", 500)]
    assert problems[("e1e5_0", 900)].startswith("unfinished")
    assert problems[("e1e5_0", 50)] == "not made yet"
    assert problems[("e1e6_0", 100)].startswith("the shower is unfinished")
    assert problems[("e1e7_0", 100)] == "the shower has no history/history.parquet"
    assert problems[("e1e8_0", 100)] == "the shower isn't made"

    ready = [check for check in checks if check.ready]
    assert [check.label for check in ready] == ["e1e5_0, energy cut 100 GeV"]
    source = ready[0].source()
    assert source.name == "e1e5_0"
    assert len(source) > 0


def test_queued_sets(config):
    sets = {
        (check.subshower_set.shower.subfolder, check.subshower_set.energy_cut): (
            check.subshower_set
        )
        for check in datasets.check_subshower_sets(config)
    }
    queued = {
        sets[("e1e5_0", 50)].output: "7_0",
        sets[("e1e8_0", 100)].shower.folder: "8_1",
    }
    problems = problems_of(datasets.check_subshower_sets(config, queued))
    assert problems[("e1e5_0", 50)] == "queued as job 7_0"
    assert problems[("e1e8_0", 100)] == "the shower is queued as job 8_1"
    assert problems[("e1e5_0", 100)] is None
