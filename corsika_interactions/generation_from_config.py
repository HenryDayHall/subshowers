import yaml
import sys
import os


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

outer_dir = os.path.join(config["storage_path"], config["showers"]['folder'])
os.makedirs(outer_dir, exist_ok=True)

to_add = []
for subfolder in config["showers"]["subfolders"]:
    to_add.append(os.path.join(outer_dir, subfolder))

to_add = os.linesep.join(to_add)

# TODO want to check for existing showers
