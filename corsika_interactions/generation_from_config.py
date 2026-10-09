import yaml
import sys
import os
import subprocess


if not len(sys.argv) == 2:
    print(f"Usage: {sys.argv[0]} <config_file>")
    sys.exit(1)

config_file = sys.argv[1]
print(f"Using config file: {config_file}")
with open(config_file, "r") as f:
    config = yaml.safe_load(f)

this_dir = os.path.dirname(os.path.abspath(__file__))
template_file_path = os.path.join(this_dir, "generate_shower_template.sh")
with open(template_file_path, "r") as f:
    template = f.read()


def generation_complete(energy_folder):
    """A generation is complete once summary.yaml records an end time."""
    summary_path = os.path.join(energy_folder, "summary.yaml")
    if not os.path.isfile(summary_path):
        return False
    with open(summary_path, "r") as f:
        return any(line.startswith("end time: ") for line in f)


outer_dir = os.path.join(config["storage_path"], config["showers"]['folder'])
os.makedirs(outer_dir, exist_ok=True)

to_add = []
skipped = []
for subfolder in config["showers"]["subfolders"]:
    energy_folder = os.path.join(outer_dir, subfolder)
    if generation_complete(energy_folder):
        skipped.append(subfolder)
        continue
    to_add.append(energy_folder)

if skipped:
    print(f"Skipping {len(skipped)} already-completed shower(s):")
    for subfolder in skipped:
        print(f"  {subfolder}")

if not to_add:
    print("All showers already generated, nothing to submit.")
    sys.exit(0)

n_folders = len(to_add)
to_add = os.linesep.join(to_add)

template = template.replace("REPLACE_WITH_EAS_BASE_FOLDER", config["storage_path"])
template = template.replace("REPLACE_WITH_ARRAY_RANGE", f"0-{n_folders - 1}")
template = template.replace("REPLACE_THIS_LINE", to_add)

generated_script = os.path.join(outer_dir, "generate_showers.sh")
with open(generated_script, "w") as f:
    f.write(template)
print(f"Wrote {generated_script} with {n_folders} shower(s), array 0-{n_folders - 1}")

result = subprocess.run(
    ["sbatch", generated_script],
    capture_output=True,
    text=True,
)
if result.returncode != 0:
    print(f"sbatch failed:\n{result.stderr}")
    sys.exit(result.returncode)

print(result.stdout.strip())
job_id = result.stdout.strip().split()[-1]
print(f"Submitted array job {job_id}")

# 1. TODO want to check for existing showers, and skip all existing data, giving notice to the user DONE
# 2. TODO want to make a copy of the template, change the REPLACE_WITH_EAS_BASE_FOLDER to the path from the configs, compile all the subfolders fromt he configs and replace REPLACE_THIS_LINE with each of them DONE
# 3. TODO submit the job to slurm DONE
# 4. TODO cna we make a dagman to start the subshowers when this is done? Yes, ish. We can make --dependancy==afterok:{job_id}
