"""
Tests for subshowers.generation, with stand-in sbatch and squeue programs that
record their arguments and report a queue read from a file.
"""

import os
import re
import subprocess

import numpy as np
import pytest
import yaml

from subshowers import generation
from fake_data import fake_history, grow_cascade

FAKE_SBATCH = """#!/bin/bash
dir=$(dirname "$0")
job=$(cat "$dir/next_job" 2>/dev/null || echo 1000)
echo $((job + 1)) > "$dir/next_job"
printf '%s\\n' "$*" >> "$dir/sbatch_calls"
echo "$job"
"""

FAKE_SQUEUE = """#!/bin/bash
printf '%s\\n' "$*" >> "$(dirname "$0")/squeue_calls"
cat "$(dirname "$0")/queue" 2>/dev/null
true
"""


@pytest.fixture
def slurm(tmp_path, monkeypatch):
    """A folder of stand-in SLURM programs, put first on PATH."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, text in [("sbatch", FAKE_SBATCH), ("squeue", FAKE_SQUEUE)]:
        (bin_dir / name).write_text(text)
        (bin_dir / name).chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return bin_dir


def sbatch_calls(bin_dir):
    path = bin_dir / "sbatch_calls"
    return path.read_text().splitlines() if path.exists() else []


def set_queue(bin_dir, tasks):
    """Make squeue report these JOB_TASK strings as queued."""
    (bin_dir / "queue").write_text("".join(f"{t.replace('_', ' ')}\n" for t in tasks))


def finished_shower(folder, seed=1):
    fake_history(
        str(folder), grow_cascade(np.random.default_rng(seed), 4.0e4, 170_000.0)
    )
    (folder / "summary.yaml").write_text(
        "start time: 01/01/2026 10:00:00\nend time: 01/01/2026 10:20:00\n"
    )


@pytest.fixture
def config(tmp_path):
    """Three showers, of which only e1e5_0 is made."""
    storage = tmp_path / "eas" / "data"
    finished_shower(storage / "PDG22" / "e1e5_0")
    return {
        "storage_path": f"{storage}/",
        "showers": {"folder": "PDG22", "subfolders": ["e1e5_0", "e1e6_0", "e1e7_0"]},
        "subshowers": {"energy_cuts": [100, 500, 900], "leaves_only": True},
    }


def script_of(call):
    return call.split()[-1]


def time_limit_of(script):
    with open(script) as f:
        for line in f:
            if line.startswith("#SBATCH --time"):
                return line.split()[-1]


def test_wall_times():
    assert generation.shower_minutes(1e5) == pytest.approx(18.3)
    assert generation.shower_minutes(1e6) == pytest.approx(165)
    assert generation.shower_minutes(1e8) == pytest.approx(3500)
    energies = np.logspace(3, 9, 50)
    minutes = [generation.shower_minutes(e) for e in energies]
    assert all(np.diff(minutes) > 0)
    assert generation.wall_time(generation.shower_minutes(1e5)).limit == "0-01:00:00"
    assert generation.wall_time(generation.shower_minutes(1e6)).limit == "0-06:00:00"
    assert generation.wall_time(generation.shower_minutes(1e7)).limit == "0-12:00:00"
    long = generation.wall_time(generation.shower_minutes(1e9))
    assert long.limit == "7-00:00:00" and long.capped and long.too_long
    # under the limit, but not with the safety margin
    tight = generation.wall_time(5 * 24 * 60)
    assert tight.limit == "7-00:00:00" and tight.capped and not tight.too_long
    # subshower run time grows in proportion to the history length,
    # so even a billion particles fits the minimum time limit
    ratio = generation.subshower_minutes(1e10) / generation.subshower_minutes(1e9)
    assert 9 < ratio < 10
    assert generation.wall_time(generation.subshower_minutes(1e9)).limit == "0-01:00:00"


def test_make_shower(config):
    shower = generation.make_shower(config, "e1e5_0")
    assert (shower.pdg, shower.energy, shower.energy_text, shower.seed) == (
        22,
        1e5,
        "1e5",
        0,
    )
    shower = generation.make_shower(config, "e2.5e6_12")
    assert (shower.energy, shower.seed) == (2.5e6, 12)
    for name in ["e1e5", "example_10e4_photon_hist1", "ebig_0", "e1e5_0/extra"]:
        assert generation.make_shower(config, name).energy is None
    config["showers"]["folder"] = "examples"
    assert generation.make_shower(config, "e1e5_0").energy is None


@pytest.mark.parametrize(
    "change, message",
    [
        ({"energy_cuts": ["1e3"]}, "yaml reads 1e4 as text"),
        ({"energy_cuts": [100, -5]}, "isn't a positive number"),
        ({"energy_cuts": [True]}, "isn't a positive number"),
        ({"energy_cuts": []}, "must be a list"),
        ({"leaves_only": "true"}, "true or false"),
        ({"subfolders": "e1e5_0"}, "list of folder names"),
    ],
)
def test_bad_subshower_settings(config, change, message):
    config["subshowers"].update(change)
    with pytest.raises(generation.ConfigError, match=message):
        generation.wanted_subshower_sets(config)


def test_duplicate_subshower_sets(config):
    config["subshowers"]["energy_cuts"] = [100, 100]
    config["subshowers"]["subfolders"] = ["e1e5_0", "e1e5_0"]
    assert len(generation.wanted_subshower_sets(config)) == 1


def test_generate_everything(config, slurm):
    generation.generate(config)
    calls = sbatch_calls(slurm)
    assert len(calls) == 5
    showers_1e6, showers_1e7, sub_1e5, sub_1e6, sub_1e7 = calls

    # one shower job per energy, each with its own time limit
    assert time_limit_of(script_of(showers_1e6)) == "0-06:00:00"
    assert time_limit_of(script_of(showers_1e7)) == "0-12:00:00"
    assert "--dependency" not in showers_1e6

    # the made shower's subshowers start straight away,
    # the others wait for the job making their shower
    assert "--dependency" not in sub_1e5
    assert "--dependency=afterok:1000_0 --kill-on-invalid-dep=yes" in sub_1e6
    assert "--dependency=afterok:1001_0 --kill-on-invalid-dep=yes" in sub_1e7

    for call in calls:
        with open(script_of(call)) as f:
            # only the comment saying the REPLACE_WITH_ values are filled in
            assert set(re.findall(r"REPLACE_\w*", f.read())) <= {"REPLACE_WITH_"}
        subprocess.run(["bash", "-n", script_of(call)], check=True)

    outer_dir = generation.showers_folder(config)
    with open(script_of(showers_1e6)) as f:
        text = f.read()
    assert f"{outer_dir}/e1e6_0" in text
    assert (
        f"EAS_BASE_FOLDER={os.path.dirname(config['storage_path'].rstrip('/'))}" in text
    )
    with open(script_of(sub_1e5)) as f:
        text = f.read()
    assert "--array=0-2" in text and "LEAVES_ONLY=True" in text
    for cut in [100, 500, 900]:
        assert f"{outer_dir}/e1e5_0/subshowers_e{cut}.h5" in text

    log = generation.read_job_log(outer_dir)
    assert [record["job"] for record in log] == ["1000", "1001", "1002", "1003", "1004"]
    assert log[0]["outputs"] == [f"{outer_dir}/e1e6_0"]
    assert log[3]["outputs"][1] == f"{outer_dir}/e1e6_0/subshowers_e500.h5"


def test_nothing_submitted_twice(config, slurm, capsys):
    generation.generate(config)
    log = generation.read_job_log(generation.showers_folder(config))
    set_queue(slurm, [f"{r['job']}_{i}" for r in log for i in range(len(r["outputs"]))])
    generation.generate(config)
    assert len(sbatch_calls(slurm)) == 5
    out = capsys.readouterr().out
    assert "Skipping 2 shower(s) already queued" in out
    assert "Skipping 9 subshower set(s) already queued" in out
    assert "--user" in (slurm / "squeue_calls").read_text()


def test_jobs_that_left_the_queue_are_submitted_again(config, slurm):
    generation.generate(config)
    set_queue(slurm, [])  # the jobs failed before making anything
    generation.generate(config)
    assert len(sbatch_calls(slurm)) == 10


def test_subshowers_wait_for_queued_showers(config, slurm, capsys):
    generation.generate(config, make_subshowers=False)
    assert len(sbatch_calls(slurm)) == 2
    set_queue(slurm, ["1000_0", "1001_0"])
    generation.generate(config, make_showers=False)
    calls = sbatch_calls(slurm)
    assert len(calls) == 5
    assert "--dependency" not in calls[2]
    assert "--dependency=afterok:1000_0" in calls[3]
    assert "--dependency=afterok:1001_0" in calls[4]

    # once the showers have left the queue unmade, their subshowers can't be made
    set_queue(slurm, ["1002_0", "1002_1", "1002_2"])
    generation.generate(config, make_showers=False)
    assert len(sbatch_calls(slurm)) == 5
    assert "the shower isn't made or queued" in capsys.readouterr().out


def test_unfinished_shower_left_alone(config, slurm, capsys):
    folder = config["storage_path"] + "PDG22/e1e6_0"
    os.makedirs(folder)
    with open(os.path.join(folder, "summary.yaml"), "w") as f:
        f.write("start time: 01/01/2026 10:00:00\n")
    generation.generate(config)
    calls = sbatch_calls(slurm)
    assert not any("e1e6" in os.path.basename(script_of(call)) for call in calls)
    out = capsys.readouterr().out
    assert "e1e6_0: summary.yaml has no end time" in out
    assert "the shower is unfinished, as summary.yaml has no end time" in out


def test_too_long_not_submitted(config, slurm, capsys, monkeypatch):
    config["showers"]["subfolders"] = ["e1e5_0", "e1e9_0"]
    monkeypatch.setattr(generation, "history_length", lambda folder: 10**8)
    monkeypatch.setattr(generation, "SUBSHOWER_SECONDS_PER_PARTICLE", 0.01)
    generation.generate(config)
    assert sbatch_calls(slurm) == []
    out = capsys.readouterr().out
    assert "Not submitting e1e9_0: a shower at 1e+09 GeV" in out
    assert "Not submitting e1e5_0 cuts 100, 500, 900: with 1e+08 particles" in out


def test_dry_run(config, slurm, capsys):
    generation.generate(config, dry_run=True)
    assert sbatch_calls(slurm) == []
    outer_dir = generation.showers_folder(config)
    assert generation.read_job_log(outer_dir) == []
    scripts = os.listdir(os.path.join(outer_dir, generation.JOBS_FOLDER))
    assert len(scripts) == 5
    out = capsys.readouterr().out
    assert out.count("Would submit") == 5
    assert "showers_e1e6_0" in out  # a subshower job says which shower job it waits for


def test_squeue_failing(config, slurm, capsys):
    (slurm / "squeue").write_text("#!/bin/bash\necho 'squeue: error' >&2\nexit 1\n")
    generation.generate(config, make_subshowers=False)
    assert "squeue couldn't be run" in capsys.readouterr().out
    assert len(sbatch_calls(slurm)) == 2


def test_saved_settings(tmp_path):
    output = tmp_path / "subshowers_e100.h5"
    output.write_text("")
    assert generation.saved_settings(str(output)) is None
    settings = {key: 1 for key in generation.SAVED_KEYS}
    settings.update(metadata_energy_cut=100.0, metadata_leaves_only=True)
    (tmp_path / "subshowers_e100.yaml").write_text(yaml.safe_dump(settings))
    assert generation.saved_settings(str(output)) == (100.0, True)
    del settings["subshowers_leaves_only"]  # as if writing it was cut short
    (tmp_path / "subshowers_e100.yaml").write_text(yaml.safe_dump(settings))
    assert generation.saved_settings(str(output)) is None
