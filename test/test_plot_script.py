"""Tests for scripts/plot_observables_from_config.py, run on fake showers."""

import importlib.util
import os
import sys
import warnings

import numpy as np
import pytest
import yaml

from subshowers import subshowers
from subshowers.ground import GroundSummaries
from subshowers.responses import Responses
from fake_data import fake_history, grow_cascade

SCRIPT = os.path.join(
    os.path.dirname(__file__), "..", "scripts", "plot_observables_from_config.py"
)


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location("plot_observables", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def config_path(tmp_path):
    """
    Of three showers, e1e5_0 is split with cuts of 100 and 500 GeV, e1e6_0
    with 100 GeV only, and e1e7_0 isn't made.
    """
    storage = tmp_path / "data"
    for name, seed, energy, cuts in [
        ("e1e5_0", 1, 3.0e3, [100, 500]),
        ("e1e6_0", 2, 5.0e3, [100]),
    ]:
        folder = storage / "PDG22" / name
        fake_history(
            str(folder), grow_cascade(np.random.default_rng(seed), energy, 170_000.0)
        )
        for cut in cuts:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                subshowers.run(
                    str(folder),
                    cut,
                    leaves_only=True,
                    output_path=str(folder / f"subshowers_e{cut}.h5"),
                )
    config = {
        "run_name": "test",
        "storage_path": f"{storage}/",
        "showers": {"folder": "PDG22", "subfolders": ["e1e5_0", "e1e6_0", "e1e7_0"]},
        "subshowers": {"energy_cuts": [100, 500], "leaves_only": True},
        "detectors": {
            "generator": {
                "name": "generate_at_ground",
                "total": 300,
                "distance_from_center": 6000.0,
                "mean_width": 300.0,
                "std_width": 20.0,
                "seed": 0,
            }
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def run(script, monkeypatch, *args):
    monkeypatch.setattr(
        sys, "argv", ["plot_observables_from_config.py"] + list(map(str, args))
    )
    script.main()


def test_plots_what_exists(script, config_path, tmp_path, monkeypatch):
    # test_observables.py and test_ground_observables.py try every figure;
    # two of each are enough here
    monkeypatch.setattr(script, "FIGURES", [script.FIGURES[0], script.FIGURES[-1]])
    monkeypatch.setattr(
        script, "GROUND_FIGURES", [script.GROUND_FIGURES[0], script.GROUND_FIGURES[2]]
    )
    monkeypatch.setattr(script, "DPI", 30)
    output = tmp_path / "plots"
    run(script, monkeypatch, config_path, "--output-dir", output)
    report = (output / "report.txt").read_text()
    assert "3 of 6 ready" in report
    assert "e1e6_0, energy cut 500 GeV: not made yet" in report
    assert "e1e7_0, energy cut 100 GeV: the shower isn't made" in report
    assert "e1e7_0, energy cut 500 GeV: the shower isn't made" in report
    assert "Ground: z = 0 m, counting particles above e± 0.5 MeV" in report
    assert "readings at 100 m from the subshower axis" in report
    assert report.count("put particles on the ground; worked out in") == 3
    for name in [
        "e100/A_by_primary.png",
        "e100/A_by_pdg.png",
        "e100/distributions_by_primary_electrons_and_positrons.png",
        "e100/distributions_by_primary.png",
        "e100/examples.png",
        "e500/distributions_by_pdg.png",
        "all_cuts/distributions_by_cut.png",
        "e100/ground_lateral_subshower_axis_by_pdg.png",
        "e100/ground_numbers_by_primary_electrons_and_positrons.png",
        "e500/ground_lateral_subshower_axis_by_primary.png",
        "all_cuts/ground_lateral_subshower_axis_by_cut.png",
        "all_cuts/ground_numbers_by_cut.png",
    ]:
        assert (output / name).is_file(), name
    saved = sorted(path.name for path in (output / "responses").iterdir())
    assert saved == ["e1e5_0_e100.npz", "e1e5_0_e500.npz", "e1e6_0_e100.npz"]
    assert sorted(path.name for path in (output / "ground").iterdir()) == saved


def test_shower_frame(script, config_path, tmp_path, monkeypatch):
    monkeypatch.setattr(script, "FIGURES", [])
    monkeypatch.setattr(script, "plot_examples", lambda *args: None)
    monkeypatch.setattr(script, "DPI", 30)
    config = yaml.safe_load(config_path.read_text())
    config["subshowers"]["energy_cuts"] = [500]
    config["ground"] = {"reference_distance": {"shower": 1000}}
    config_path.write_text(yaml.safe_dump(config))
    output = tmp_path / "plots"
    run(script, monkeypatch, config_path, "--output-dir", output, "--frame", "shower")
    report = (output / "report.txt").read_text()
    assert "readings at 1000 m from the shower axis" in report
    written = sorted(path.name for path in (output / "e500").iterdir())
    assert written == [
        f"{name}_{suffix}.png"
        for name in [
            "ground_lateral_shower_axis",
            "ground_numbers",
            "ground_reference_shower_axis",
            "ground_spectra",
            "ground_timing_shower_axis",
        ]
        for suffix in ["by_pdg", "by_primary", "by_primary_electrons_and_positrons"]
    ]


def test_saved_responses_reused(script, config_path, tmp_path, monkeypatch, capsys):
    # only the responses matter here, not the plots
    monkeypatch.setattr(script, "plot_groups", lambda *args: None)
    monkeypatch.setattr(script, "plot_examples", lambda *args: None)
    output = tmp_path / "plots"
    arguments = [config_path, "--output-dir", output, "--max-subshowers", 5]

    run(script, monkeypatch, *arguments)
    printed = capsys.readouterr().out
    assert printed.count("measured in") == 3
    assert printed.count("ground; worked out in") == 3
    for path in (output / "responses").iterdir():
        assert len(Responses.load(str(path))) == 5
    for path in (output / "ground").iterdir():
        assert len(GroundSummaries.load(str(path))) == 5

    run(script, monkeypatch, *arguments)
    printed = capsys.readouterr().out
    assert printed.count("saved earlier") == 3
    assert printed.count("ground; as worked out before") == 3

    run(script, monkeypatch, *arguments, "--recompute")
    printed = capsys.readouterr().out
    assert printed.count("measured in") == 3
    assert printed.count("ground; worked out in") == 3

    # another array means measuring again, but the same subshowers are on
    # the ground
    config = yaml.safe_load(config_path.read_text())
    config["detectors"]["generator"]["seed"] = 1
    config_path.write_text(yaml.safe_dump(config))
    run(script, monkeypatch, *arguments)
    printed = capsys.readouterr().out
    assert printed.count("measured in") == 3
    assert printed.count("ground; as worked out before") == 3

    # another ground means working out what's on it again
    config["ground"] = {"height": 100.0}
    config_path.write_text(yaml.safe_dump(config))
    run(script, monkeypatch, *arguments)
    printed = capsys.readouterr().out
    assert printed.count("saved earlier") == 3
    assert printed.count("ground; worked out in") == 3

    # a random array without a seed is measured again every time
    del config["detectors"]["generator"]["seed"]
    config_path.write_text(yaml.safe_dump(config))
    for _ in range(2):
        run(script, monkeypatch, *arguments)
        printed = capsys.readouterr().out
        assert "has no seed" in printed
        assert printed.count("measured in") == 3


def test_nothing_to_plot(script, tmp_path, monkeypatch):
    config = {
        "storage_path": f"{tmp_path}/data/",
        "showers": {"folder": "PDG22", "subfolders": ["e1e5_0"]},
        "subshowers": {"energy_cuts": [100]},
        "detectors": {"generator": {"name": "generate_pierre_auger"}},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    output = tmp_path / "plots"
    with pytest.raises(SystemExit) as stopped:
        run(script, monkeypatch, config_path, "--output-dir", output)
    assert stopped.value.code == 1
    report = (output / "report.txt").read_text()
    assert "e1e5_0, energy cut 100 GeV: the shower isn't made" in report
    assert "nothing to plot" in report


def test_bad_detector(script, config_path, monkeypatch):
    config = yaml.safe_load(config_path.read_text())
    del config["detectors"]["generator"]["total"]
    config_path.write_text(yaml.safe_dump(config))
    with pytest.raises(SystemExit) as stopped:
        run(script, monkeypatch, config_path)
    assert "total" in str(stopped.value.code)


def test_ground_failure_keeps_other_plots(script, config_path, tmp_path, monkeypatch):
    monkeypatch.setattr(script, "FIGURES", [script.FIGURES[-1]])
    monkeypatch.setattr(script, "GROUND_FIGURES", [script.GROUND_FIGURES[1]])
    monkeypatch.setattr(script, "plot_examples", lambda *args: None)
    monkeypatch.setattr(script, "DPI", 30)
    summarise_ground = script.summarise_ground

    def failing(check, *args, **kwargs):
        if check.subshower_set.shower.subfolder == "e1e6_0":
            raise RuntimeError("broken")
        return summarise_ground(check, *args, **kwargs)

    monkeypatch.setattr(script, "summarise_ground", failing)
    output = tmp_path / "plots"
    run(script, monkeypatch, config_path, "--output-dir", output)
    report = (output / "report.txt").read_text()
    assert "couldn't be worked out, RuntimeError: broken" in report
    assert report.count("without what reaches the ground") == 1
    # e1e6_0 has only the 100 GeV cut, so the 500 GeV cut keeps its ground
    assert sorted(path.name for path in (output / "e100").iterdir()) == [
        "distributions_by_pdg.png",
        "distributions_by_primary.png",
        "distributions_by_primary_electrons_and_positrons.png",
    ]
    assert (output / "e500" / "ground_spectra_by_pdg.png").is_file()
    assert sorted(path.name for path in (output / "all_cuts").iterdir()) == [
        "distributions_by_cut.png"
    ]


def test_bad_ground(script, config_path, monkeypatch):
    config = yaml.safe_load(config_path.read_text())
    config["ground"] = {"reference_distance": {"subshower": -5}}
    config_path.write_text(yaml.safe_dump(config))
    with pytest.raises(SystemExit) as stopped:
        run(script, monkeypatch, config_path)
    assert "ground: reference_distance: subshower" in str(stopped.value.code)
