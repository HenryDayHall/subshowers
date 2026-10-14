"""
Plan and submit the SLURM jobs that make the data a config describes: showers
from CORSIKA (corsika_interactions/generate_shower_template.sh), and the
subshowers split from them (scripts/generate_subshowers_template.sh).

generate() compares what the config asks for with what's on disk and in the
queue, then submits array jobs for whatever is missing:

* one shower job per primary energy, so each gets a wall time to suit it;
* one subshower job per shower, covering all its energy cuts. If the shower
  is still being made, the job waits for it with --dependency=afterok, and is
  cancelled if the shower job fails.

Each submission is recorded in jobs/submitted_jobs.yaml in the showers folder.
With squeue, that tells later runs which outputs are already queued, so they
aren't submitted twice, and new subshower jobs can wait for queued showers.

Needs only the standard library and yaml. If pyarrow is available, it's used
to count the particles in finished histories, which sets subshower wall times.
"""

import argparse
import getpass
import math
import os
import re
import shlex
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass

import yaml

REPO_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
SHOWER_TEMPLATE = os.path.join(
    REPO_DIR, "corsika_interactions", "generate_shower_template.sh"
)
SUBSHOWER_TEMPLATE = os.path.join(
    REPO_DIR, "scripts", "generate_subshowers_template.sh"
)

# Job scripts and the record of submissions go in this folder,
# inside the showers folder
JOBS_FOLDER = "jobs"
JOB_LOG = "submitted_jobs.yaml"

DONE = "done"
MISSING = "missing"


class ConfigError(ValueError):
    """The config asks for something that can't be made."""


class SubmissionError(RuntimeError):
    """sbatch failed."""


# ------------------------------------------------------------------ wall time

# Run time of one shower, in minutes, at a few primary energies in GeV.
# 1e5 GeV is measured: PDG 22 took 18.3 minutes. The rest scale that by the
# timings noted in corsika_interactions/generate_shower.sh, where 1e6 GeV
# took 9 times as long as 1e5 GeV, but 1e7 GeV only 2.1 times as long as 1e6:
# c8_air_shower_with_history thins by default (--emthin 1e-6), and its
# maximum weight, 0.5e-6 times the energy in GeV, only passes 1 above 2e6 GeV.
# Outside the table, run time is taken to grow in proportion to energy, in
# line with the note that 1e8 GeV took more than two days.
SHOWER_MINUTES = {1e5: 18.3, 1e6: 165.0, 1e7: 350.0}

# Splitting a shower takes about this long per particle in its history, plus
# time to start up. subshowers.run took 7 seconds for 10 million particles
# (reading the history, building its tree, finding and saving subshowers),
# using about 320 bytes of memory per particle.
SUBSHOWER_SECONDS_PER_PARTICLE = 1e-6
SUBSHOWER_STARTUP_SECONDS = 60

# History length is taken to grow in proportion to energy, calibrated on any
# finished histories. With none to read, assume this many particles per GeV.
DEFAULT_PARTICLES_PER_GEV = 100

# Jobs ask for SAFETY_FACTOR times their estimated run time, rounded up to
# whole hours, and between MIN_MINUTES and MAX_MINUTES; 7 days is the longest
# the maxcpu partition allows.
SAFETY_FACTOR = 2
MIN_MINUTES = 60
MAX_MINUTES = 7 * 24 * 60


def shower_minutes(energy):
    """Estimated run time in minutes of one shower at this energy in GeV."""
    points = sorted(SHOWER_MINUTES.items())
    lowest_energy, lowest_minutes = points[0]
    highest_energy, highest_minutes = points[-1]
    if energy <= lowest_energy:
        return lowest_minutes * energy / lowest_energy
    if energy >= highest_energy:
        return highest_minutes * energy / highest_energy
    # straight lines between the table's points on a log-log plot
    for (energy_0, minutes_0), (energy_1, minutes_1) in zip(points, points[1:]):
        if energy <= energy_1:
            slope = math.log(minutes_1 / minutes_0) / math.log(energy_1 / energy_0)
            return minutes_0 * (energy / energy_0) ** slope


def subshower_minutes(n_particles):
    """Estimated run time in minutes of subshowers.run on a history this long."""
    seconds = SUBSHOWER_STARTUP_SECONDS + SUBSHOWER_SECONDS_PER_PARTICLE * n_particles
    return seconds / 60


@dataclass
class WallTime:
    """The --time to ask for, given an estimated run time."""

    estimate: float  # minutes
    limit: str  # D-HH:MM:SS
    capped: bool  # the safety margin was cut short by MAX_MINUTES
    too_long: bool  # the estimate alone is over MAX_MINUTES


def wall_time(estimated_minutes):
    wanted = max(MIN_MINUTES, 60 * math.ceil(SAFETY_FACTOR * estimated_minutes / 60))
    minutes = min(wanted, MAX_MINUTES)
    days, minutes = divmod(minutes, 24 * 60)
    hours, minutes = divmod(minutes, 60)
    return WallTime(
        estimate=estimated_minutes,
        limit=f"{days}-{hours:02d}:{minutes:02d}:00",
        capped=wanted > MAX_MINUTES,
        too_long=estimated_minutes > MAX_MINUTES,
    )


def describe_minutes(minutes):
    """Readable duration, e.g. 37 minutes, 5.5 hours, 3.2 days."""
    if minutes < 1.5:
        return "about a minute"
    if minutes < 90:
        return f"{minutes:.0f} minutes"
    if minutes < 48 * 60:
        return f"{minutes / 60:.1f} hours"
    return f"{minutes / (24 * 60):.1f} days"


# ----------------------------------------------------- what the config wants


@dataclass
class Shower:
    """
    A shower the config names. pdg, energy and seed are None unless the
    folders follow the PDG<pdg>/e<energy>_<seed> pattern that the shower
    template reads, which they must for the shower to be generated here.
    """

    subfolder: str
    folder: str  # absolute path
    pdg: int = None
    energy: float = None  # GeV
    energy_text: str = None  # as written in the folder name, e.g. 1e5
    seed: int = None


@dataclass
class SubshowerSet:
    """One shower split with one energy cut."""

    shower: Shower
    energy_cut: float  # GeV, as written in the config
    leaves_only: bool

    @property
    def output(self):
        # the name subshowers.run gives its output by default
        return os.path.join(self.shower.folder, f"subshowers_e{self.energy_cut}.h5")


def showers_folder(config):
    return os.path.abspath(
        os.path.join(config["storage_path"], config["showers"]["folder"])
    )


def eas_base_folder(config):
    """The folder holding CORSIKA's setup_env.sh and corsika-build/."""
    return os.path.dirname(os.path.abspath(config["storage_path"]))


def make_shower(config, subfolder):
    outer_dir = showers_folder(config)
    shower = Shower(subfolder, os.path.join(outer_dir, subfolder))
    pdg_match = re.fullmatch(r"PDG(-?\d+)", os.path.basename(outer_dir))
    name_match = re.fullmatch(r"e([^_/]+)_(\d+)", subfolder)
    if pdg_match and name_match:
        try:
            energy = float(name_match.group(1))
        except ValueError:
            return shower
        if math.isfinite(energy) and energy > 0:
            shower.pdg = int(pdg_match.group(1))
            shower.energy = energy
            shower.energy_text = name_match.group(1)
            shower.seed = int(name_match.group(2))
    return shower


def _list_of_names(value, where):
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ConfigError(f"{where} must be a list of folder names")
    return value


def wanted_showers(config):
    """The showers in the config's showers section."""
    subfolders = _list_of_names(config["showers"]["subfolders"], "showers: subfolders")
    return [make_shower(config, subfolder) for subfolder in dict.fromkeys(subfolders)]


def wanted_subshower_sets(config):
    """The subshower sets the config's subshowers section asks for."""
    section = config.get("subshowers")
    if not isinstance(section, dict):
        raise ConfigError("the config has no subshowers section")

    energy_cuts = section.get("energy_cuts")
    if not isinstance(energy_cuts, list) or not energy_cuts:
        raise ConfigError("subshowers: energy_cuts must be a list of energies in GeV")
    for energy_cut in energy_cuts:
        is_number = isinstance(energy_cut, (int, float)) and not isinstance(
            energy_cut, bool
        )
        if not (is_number and math.isfinite(energy_cut) and energy_cut > 0):
            hint = ""
            if isinstance(energy_cut, str):
                hint = " (yaml reads 1e4 as text, so write 10000 or 1.0e+4)"
            raise ConfigError(
                f"subshowers: energy cut {energy_cut!r} isn't a positive number"
                f" of GeV{hint}"
            )

    leaves_only = section.get("leaves_only", True)
    if not isinstance(leaves_only, bool):
        raise ConfigError(
            f"subshowers: leaves_only must be true or false, not {leaves_only!r}"
        )

    subfolders = _list_of_names(
        section.get("subfolders", config["showers"]["subfolders"]),
        "subshowers: subfolders",
    )
    sets = {}
    for subfolder in subfolders:
        shower = make_shower(config, subfolder)
        for energy_cut in energy_cuts:
            subshower_set = SubshowerSet(shower, energy_cut, leaves_only)
            sets.setdefault(subshower_set.output, subshower_set)
    return list(sets.values())


# --------------------------------------------------------------- what exists


def history_path(shower_folder):
    return os.path.join(shower_folder, "history", "history.parquet")


def shower_status(shower_folder):
    """
    DONE, MISSING, or why the shower is unfinished. CORSIKA writes an end time
    to summary.yaml once a shower is finished. A history without a
    summary.yaml (made some other way, e.g. test/fake_data.py) also counts.
    """
    if not os.path.exists(shower_folder):
        return MISSING
    summary_path = os.path.join(shower_folder, "summary.yaml")
    if os.path.isfile(summary_path):
        with open(summary_path, "r") as f:
            if any(line.startswith("end time: ") for line in f):
                return DONE
        return "summary.yaml has no end time"
    if os.path.isfile(history_path(shower_folder)):
        return DONE
    return "it has neither a summary.yaml nor a history"


def history_length(shower_folder):
    """Particles in a shower's history, from the parquet metadata, or None."""
    try:
        import pyarrow.parquet as pq

        return pq.ParquetFile(history_path(shower_folder)).metadata.num_rows
    except (ImportError, OSError, ValueError):
        return None


# Keys Output.save always writes to the .yaml next to its .h5
SAVED_KEYS = [
    "metadata_energy_cut",
    "metadata_leaves_only",
    "starts_folder",
    "starts_root",
    "subshowers_folder",
    "subshowers_leaves_only",
]


def saved_settings(output_path):
    """
    (energy cut, leaves only) of a finished subshower set, or None if there
    isn't one. Output.save writes the .h5 first and the .yaml last, so a set is
    finished once its .yaml holds every key. (File times can't be used, as
    loading a set opens the .h5 for writing.)
    """
    yaml_path = output_path[: -len(".h5")] + ".yaml"
    if not (os.path.isfile(output_path) and os.path.isfile(yaml_path)):
        return None
    try:
        with open(yaml_path, "r") as f:
            saved = yaml.safe_load(f)
    except yaml.YAMLError:
        return None
    if not isinstance(saved, dict) or not all(key in saved for key in SAVED_KEYS):
        return None
    return saved["metadata_energy_cut"], saved["metadata_leaves_only"]


# --------------------------------------------------------------------- queue


def queued_tasks():
    """
    This user's array tasks that are queued or running, as JOB_TASK strings,
    or None if squeue can't be run.
    """
    command = [
        "squeue", "--user", getpass.getuser(), "--noheader", "--array",
        "--format", "%F %K",
    ]  # fmt: skip
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    tasks = set()
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) == 2:
            tasks.add("_".join(fields))
    return tasks


def job_log_path(outer_dir):
    return os.path.join(outer_dir, JOBS_FOLDER, JOB_LOG)


def read_job_log(outer_dir):
    path = job_log_path(outer_dir)
    if not os.path.isfile(path):
        return []
    with open(path, "r") as f:
        records = yaml.safe_load(f)
    return records if isinstance(records, list) else []


def record_submission(outer_dir, job_id, script_path, outputs):
    """Add a job to the log; task i of the job makes outputs[i]."""
    records = read_job_log(outer_dir)
    records.append(
        {
            "job": job_id,
            "submitted": time.strftime("%Y-%m-%d %H:%M:%S"),
            "script": script_path,
            "outputs": list(outputs),
        }
    )
    path = job_log_path(outer_dir)
    with open(path + ".tmp", "w") as f:
        yaml.safe_dump(records, f, sort_keys=False)
    os.replace(path + ".tmp", path)


def queued_outputs(outer_dir):
    """
    {output path: JOB_TASK} for logged outputs whose job is still queued or
    running, or None if squeue can't be run.
    """
    tasks = queued_tasks()
    if tasks is None:
        return None
    found = {}
    for record in read_job_log(outer_dir):
        for task_number, output in enumerate(record.get("outputs", [])):
            task = f"{record['job']}_{task_number}"
            if task in tasks:
                found[output] = task
    return found


# ---------------------------------------------------------------- submitting


def write_job_script(template_path, replacements, script_path):
    with open(template_path, "r") as f:
        text = f.read()
    for placeholder, value in replacements.items():
        if placeholder not in text:
            raise ValueError(f"{placeholder} isn't in {template_path}")
        text = text.replace(placeholder, value)
    os.makedirs(os.path.dirname(script_path), exist_ok=True)
    with open(script_path, "w") as f:
        f.write(text)


def sbatch(script_path, wait_for=None):
    """
    Submit a job script and return its job ID. wait_for is a JOB_TASK that
    must finish successfully first; if it fails, this job is cancelled.
    """
    command = ["sbatch", "--parsable"]
    if wait_for is not None:
        command += [f"--dependency=afterok:{wait_for}", "--kill-on-invalid-dep=yes"]
    command.append(script_path)
    try:
        result = subprocess.run(command, capture_output=True, text=True)
    except FileNotFoundError:
        raise SubmissionError(
            "sbatch not found; submit from a SLURM login node, or use --dry-run"
        )
    if result.returncode != 0:
        raise SubmissionError(f"sbatch {script_path} failed:\n{result.stderr.strip()}")
    return result.stdout.strip().split(";")[0]


def _submit(outer_dir, script_path, outputs, wait_for, dry_run):
    """Submit and log a job, returning its ID; in a dry run, the script name."""
    if dry_run:
        return os.path.basename(script_path)[: -len(".sh")]
    job_id = sbatch(script_path, wait_for)
    record_submission(outer_dir, job_id, script_path, outputs)
    return job_id


def _print_list(heading, lines):
    if lines:
        print(heading)
        for line in lines:
            print(f"  {line}")


# -------------------------------------------------------------------- showers


def submit_showers(config, queued, stamp, dry_run=False):
    """
    Submit jobs for the config's showers that aren't made or queued, one
    array job per energy. Returns {shower folder: JOB_TASK making it}.
    """
    outer_dir = showers_folder(config)
    done, already_queued, unfinished, unnamed, to_make = [], [], [], [], {}
    for shower in wanted_showers(config):
        status = shower_status(shower.folder)
        if status == DONE:
            done.append(shower.subfolder)
        elif shower.folder in queued:
            already_queued.append(f"{shower.subfolder} (job {queued[shower.folder]})")
        elif status != MISSING:
            unfinished.append(f"{shower.subfolder}: {status}")
        elif shower.energy is None:
            unnamed.append(shower.subfolder)
        else:
            to_make.setdefault(shower.energy, []).append(shower)

    print(f"Showers in {outer_dir}")
    _print_list(f"Skipping {len(done)} shower(s) already made:", done)
    _print_list(
        f"Skipping {len(already_queued)} shower(s) already queued:", already_queued
    )
    _print_list(
        f"Skipping {len(unfinished)} unfinished shower(s), which may still be"
        " running or may have failed; delete the folder to make one again:",
        unfinished,
    )
    _print_list(
        f"Can't make {len(unnamed)} shower(s), as the folders don't follow"
        " PDG<pdg>/e<energy>_<seed>:",
        unnamed,
    )

    jobs = {}
    for energy, showers in sorted(to_make.items()):
        name = f"showers_e{showers[0].energy_text}"
        wall = wall_time(shower_minutes(energy))
        listed = ", ".join(shower.subfolder for shower in showers)
        if wall.too_long:
            print(
                f"Not submitting {listed}: a shower at {energy:g} GeV is estimated"
                f" to take {describe_minutes(wall.estimate)}, longer than the"
                f" {describe_minutes(MAX_MINUTES)} limit"
            )
            continue
        script = os.path.join(outer_dir, JOBS_FOLDER, f"{stamp}_{name}.sh")
        folders = [shower.folder for shower in showers]
        write_job_script(
            SHOWER_TEMPLATE,
            {
                "REPLACE_WITH_TIME_LIMIT": wall.limit,
                "REPLACE_WITH_ARRAY_RANGE": f"0-{len(showers) - 1}",
                "REPLACE_WITH_EAS_BASE_FOLDER": shlex.quote(eas_base_folder(config)),
                "REPLACE_THIS_LINE": os.linesep.join(map(shlex.quote, folders)),
            },
            script,
        )
        job_id = _submit(outer_dir, script, folders, None, dry_run)
        for task_number, folder in enumerate(folders):
            jobs[folder] = f"{job_id}_{task_number}"
        note = " (shortened to fit the limit)" if wall.capped else ""
        print(
            f"{'Would submit' if dry_run else 'Submitted'} job {job_id}:"
            f" {listed}, estimated {describe_minutes(wall.estimate)} each,"
            f" time limit {wall.limit}{note}"
        )
    if not to_make:
        print("No showers to submit.")
    return jobs


# ----------------------------------------------------------------- subshowers


def particle_counts(showers):
    """
    {shower folder: (particles, where the number comes from)}: read from
    finished histories, and for other showers scaled by energy from those.
    """
    counts = {}
    per_gev = []
    for shower in showers:
        n_particles = None
        if shower_status(shower.folder) == DONE:
            n_particles = history_length(shower.folder)
        if n_particles is not None:
            counts[shower.folder] = (n_particles, "in its history")
            if shower.energy is not None:
                per_gev.append(n_particles / shower.energy)
    if per_gev:
        particles_per_gev = statistics.median(per_gev)
        source = f"scaled by energy from {len(per_gev)} finished shower(s)"
    else:
        particles_per_gev = DEFAULT_PARTICLES_PER_GEV
        source = f"guessing {DEFAULT_PARTICLES_PER_GEV} per GeV"
    for shower in showers:
        if shower.folder not in counts and shower.energy is not None:
            counts[shower.folder] = (round(particles_per_gev * shower.energy), source)
    return counts


def submit_subshowers(config, queued, shower_jobs, stamp, dry_run=False):
    """
    Submit jobs for the config's subshower sets that aren't made or queued,
    one array job per shower. Sets of showers in shower_jobs ({folder:
    JOB_TASK}) wait for that task.
    """
    outer_dir = showers_folder(config)
    subshower_sets = wanted_subshower_sets(config)
    in_showers_section = set(config["showers"]["subfolders"])
    done, other_settings, already_queued, blocked = [], [], [], []
    to_make = {}  # shower folder -> [sets]
    for subshower_set in subshower_sets:
        shower = subshower_set.shower
        label = os.path.relpath(subshower_set.output, outer_dir)
        settings = saved_settings(subshower_set.output)
        status = shower_status(shower.folder)
        if settings == (subshower_set.energy_cut, subshower_set.leaves_only):
            done.append(label)
        elif settings is not None:
            other_settings.append(
                f"{label} (energy cut {settings[0]}, leaves only {settings[1]})"
            )
        elif subshower_set.output in queued:
            already_queued.append(f"{label} (job {queued[subshower_set.output]})")
        elif status == DONE and not os.path.isfile(history_path(shower.folder)):
            blocked.append(f"{label}: the shower has no history/history.parquet")
        elif status == DONE or shower.folder in shower_jobs:
            to_make.setdefault(shower.folder, []).append(subshower_set)
        elif status == MISSING and shower.subfolder not in in_showers_section:
            blocked.append(
                f"{label}: the shower isn't made, nor in the showers section"
            )
        elif status == MISSING:
            blocked.append(f"{label}: the shower isn't made or queued")
        else:
            blocked.append(f"{label}: the shower is unfinished, as {status}")

    print(f"Subshowers in {outer_dir}")
    _print_list(f"Skipping {len(done)} subshower set(s) already made:", done)
    _print_list(
        f"Skipping {len(other_settings)} subshower set(s) made with other"
        " settings; delete the .h5 and .yaml to make them again:",
        other_settings,
    )
    _print_list(
        f"Skipping {len(already_queued)} subshower set(s) already queued:",
        already_queued,
    )
    _print_list(f"Can't make {len(blocked)} subshower set(s) yet:", blocked)

    # every shower either section names helps calibrate history lengths
    showers = {shower.folder: shower for shower in wanted_showers(config)}
    for subshower_set in subshower_sets:
        showers.setdefault(subshower_set.shower.folder, subshower_set.shower)
    counts = particle_counts(showers.values())
    jobs = []
    for folder, sets in to_make.items():
        shower = showers[folder]
        cuts = ", ".join(str(subshower_set.energy_cut) for subshower_set in sets)
        if folder in counts:
            n_particles, source = counts[folder]
            wall = wall_time(subshower_minutes(n_particles))
            size = f"{n_particles:.3g} particles, {source}"
            estimate = f"estimated {describe_minutes(wall.estimate)} each"
        else:
            # an existing shower without an energy in its name, and no pyarrow
            wall = wall_time(12 * 60 / SAFETY_FACTOR)
            size = "size unknown"
            estimate = "no estimate"
        if wall.too_long:
            print(
                f"Not submitting {shower.subfolder} cuts {cuts}: with {size},"
                f" finding subshowers is estimated to take"
                f" {describe_minutes(wall.estimate)}, longer than the"
                f" {describe_minutes(MAX_MINUTES)} limit"
            )
            continue
        name = f"subshowers_{shower.subfolder.replace('/', '_')}"
        script = os.path.join(outer_dir, JOBS_FOLDER, f"{stamp}_{name}.sh")
        outputs = [subshower_set.output for subshower_set in sets]
        write_job_script(
            SUBSHOWER_TEMPLATE,
            {
                "REPLACE_WITH_TIME_LIMIT": wall.limit,
                "REPLACE_WITH_ARRAY_RANGE": f"0-{len(outputs) - 1}",
                "REPLACE_WITH_REPO_FOLDER": shlex.quote(REPO_DIR),
                "REPLACE_WITH_LEAVES_ONLY": str(sets[0].leaves_only),
                "REPLACE_WITH_OUTPUT_FILES": os.linesep.join(map(shlex.quote, outputs)),
            },
            script,
        )
        wait_for = shower_jobs.get(folder) if shower_status(folder) != DONE else None
        job_id = _submit(outer_dir, script, outputs, wait_for, dry_run)
        jobs.append(job_id)
        after = f", after job {wait_for}" if wait_for else ""
        note = " (shortened to fit the limit)" if wall.capped else ""
        print(
            f"{'Would submit' if dry_run else 'Submitted'} job {job_id}:"
            f" {shower.subfolder} cuts {cuts} ({size}), {estimate},"
            f" time limit {wall.limit}{note}{after}"
        )
    if not to_make:
        print("No subshowers to submit.")
    return jobs


# ---------------------------------------------------------------------- both


def generate(config, make_showers=True, make_subshowers=True, dry_run=False):
    """
    Submit jobs for whatever the config asks for that isn't made or queued.
    make_showers and make_subshowers choose the kinds of job; with dry_run the
    job scripts are written but not submitted.
    """
    outer_dir = showers_folder(config)
    if make_subshowers:
        wanted_subshower_sets(config)  # check the section before submitting anything
    queued = queued_outputs(outer_dir)
    if queued is None:
        print(
            "squeue couldn't be run, so jobs already in the queue can't be"
            " spotted, and their outputs may be submitted again."
        )
        queued = {}
    stamp = time.strftime("%Y%m%d-%H%M%S")

    # subshowers of showers still being made wait for the job making them,
    # whether it was queued earlier or is submitted now
    shower_jobs = dict(queued)
    if make_showers:
        shower_jobs.update(submit_showers(config, queued, stamp, dry_run))
    if make_subshowers:
        if make_showers:
            print()
        submit_subshowers(config, queued, shower_jobs, stamp, dry_run)
    if dry_run:
        print(f"Dry run: job scripts written to {os.path.join(outer_dir, JOBS_FOLDER)}")


def command_line(make_showers=True, make_subshowers=True, description=None):
    """Read a config named on the command line and run generate() on it."""
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("config_file")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="write the job scripts without submitting them",
    )
    args = parser.parse_args()
    print(f"Using config file: {args.config_file}")
    with open(args.config_file, "r") as f:
        config = yaml.safe_load(f)
    try:
        generate(config, make_showers, make_subshowers, args.dry_run)
    except (ConfigError, SubmissionError) as error:
        sys.exit(str(error))
