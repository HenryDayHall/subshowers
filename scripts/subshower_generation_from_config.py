"""
Submit a SLURM array job that splits showers into subshowers, as set out in
the subshowers section of a config. Each array task makes one set of
subshowers: one shower split with one energy cut.

    python scripts/subshower_generation_from_config.py configs/default.yaml

Sets already made with the same settings are skipped, as are showers that
haven't finished generating. With --dry-run the job script is written but
not submitted.
"""

import argparse
import math
import os
import shlex
import subprocess
import sys

import yaml


parser = argparse.ArgumentParser(
    description="Submit subshower generation jobs for the showers in a config."
)
parser.add_argument("config_file")
parser.add_argument(
    "--dry-run", action="store_true", help="write the job script without submitting it"
)
args = parser.parse_args()

config_file = args.config_file
print(f"Using config file: {config_file}")
with open(config_file, "r") as f:
    config = yaml.safe_load(f)

this_dir = os.path.dirname(os.path.realpath(__file__))
repo_dir = os.path.dirname(this_dir)
template_file_path = os.path.join(this_dir, "generate_subshowers_template.sh")
with open(template_file_path, "r") as f:
    template = f.read()


# Settings from the config
if not isinstance(config.get("subshowers"), dict):
    sys.exit(f"{config_file} has no subshowers section")
subshower_config = config["subshowers"]

energy_cuts = subshower_config.get("energy_cuts")
if not isinstance(energy_cuts, list) or not energy_cuts:
    sys.exit("subshowers: energy_cuts must be a list of energies in GeV")
for energy_cut in energy_cuts:
    is_number = isinstance(energy_cut, (int, float)) and not isinstance(energy_cut, bool)
    if not (is_number and math.isfinite(energy_cut) and energy_cut > 0):
        hint = ""
        if isinstance(energy_cut, str):
            hint = " (yaml reads 1e4 as text, so write 10000 or 1.0e+4)"
        sys.exit(
            f"subshowers: energy cut {energy_cut!r} isn't a positive number of GeV{hint}"
        )

leaves_only = subshower_config.get("leaves_only", True)
if not isinstance(leaves_only, bool):
    sys.exit(f"subshowers: leaves_only must be true or false, not {leaves_only!r}")

subfolders = subshower_config.get("subfolders", config["showers"]["subfolders"])
if not isinstance(subfolders, list) or not all(isinstance(s, str) for s in subfolders):
    sys.exit("subshowers: subfolders must be a list of folder names")


def output_file_name(energy_cut):
    """The name subshowers.run gives its output by default."""
    return f"subshowers_e{energy_cut}.h5"


def shower_problem(shower_folder):
    """Why a shower can't be split yet, or None if it can."""
    if not os.path.isfile(os.path.join(shower_folder, "history", "history.parquet")):
        return "no history/history.parquet"
    # CORSIKA's summary.yaml records an end time once the shower is finished.
    # Histories made some other way (e.g. test/fake_data.py) have no summary.
    summary_path = os.path.join(shower_folder, "summary.yaml")
    if os.path.isfile(summary_path):
        with open(summary_path, "r") as f:
            if not any(line.startswith("end time: ") for line in f):
                return "summary.yaml has no end time, so the shower may not be finished"
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
    (energy cut, leaves only) of a finished output, or None if there isn't one.
    Output.save writes the .h5 first and the .yaml last, so an output is
    finished once its .yaml holds every key. (File times can't be used, as
    loading an output opens the .h5 for writing.)
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


outer_dir = os.path.abspath(
    os.path.join(config["storage_path"], config["showers"]["folder"])
)

to_add = []
done = []
mismatched = []
not_ready = {}
seen = set()
for subfolder in subfolders:
    shower_folder = os.path.join(outer_dir, subfolder)
    problem = shower_problem(shower_folder)
    for energy_cut in energy_cuts:
        output_path = os.path.join(shower_folder, output_file_name(energy_cut))
        if output_path in seen:
            continue  # listed twice in the config
        seen.add(output_path)
        settings = saved_settings(output_path)
        if settings == (energy_cut, leaves_only):
            done.append(output_path)
        elif settings is not None:
            mismatched.append((output_path, settings))
        elif problem is not None:
            not_ready[subfolder] = problem
        else:
            to_add.append(output_path)

if done:
    print(f"Skipping {len(done)} subshower set(s) already made:")
    for output_path in done:
        print(f"  {os.path.relpath(output_path, outer_dir)}")

if mismatched:
    print(
        f"Skipping {len(mismatched)} subshower set(s) made with other settings;"
        " delete the .h5 and .yaml to remake them:"
    )
    for output_path, (saved_cut, saved_leaves_only) in mismatched:
        print(
            f"  {os.path.relpath(output_path, outer_dir)}"
            f" (energy cut {saved_cut}, leaves only {saved_leaves_only})"
        )

if not_ready:
    print(f"Skipping {len(not_ready)} shower(s) that can't be split yet:")
    for subfolder, problem in not_ready.items():
        print(f"  {subfolder}: {problem}")

if not to_add:
    print("Nothing to submit.")
    sys.exit(0)

n_tasks = len(to_add)
output_lines = os.linesep.join(shlex.quote(output_path) for output_path in to_add)

template = template.replace("REPLACE_WITH_REPO_FOLDER", shlex.quote(repo_dir))
template = template.replace("REPLACE_WITH_LEAVES_ONLY", str(leaves_only))
template = template.replace("REPLACE_WITH_ARRAY_RANGE", f"0-{n_tasks - 1}")
template = template.replace("REPLACE_WITH_OUTPUT_FILES", output_lines)

generated_script = os.path.join(outer_dir, "generate_subshowers.sh")
with open(generated_script, "w") as f:
    f.write(template)
print(
    f"Wrote {generated_script} with {n_tasks} subshower set(s), array 0-{n_tasks - 1}"
)
for output_path in to_add:
    print(f"  {os.path.relpath(output_path, outer_dir)}")

if args.dry_run:
    print("Dry run, not submitting.")
    sys.exit(0)

try:
    result = subprocess.run(
        ["sbatch", generated_script],
        capture_output=True,
        text=True,
    )
except FileNotFoundError:
    sys.exit("sbatch not found; submit from a SLURM login node, or use --dry-run")
if result.returncode != 0:
    print(f"sbatch failed:\n{result.stderr}")
    sys.exit(result.returncode)

print(result.stdout.strip())
job_id = result.stdout.strip().split()[-1]
print(f"Submitted array job {job_id}")
