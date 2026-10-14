"""
Submit every SLURM job needed to make the data a config describes.

    python scripts/generate_data_from_config.py configs/default.yaml

Checks what's already made or queued, then submits
  * shower jobs for missing showers, one array job per energy;
  * subshower jobs for missing energy cuts, one array job per shower.
    Jobs for showers still being made wait for them with
    --dependency=afterok, and are cancelled if the shower job fails.
Each job gets a time limit to suit its shower's energy. With --dry-run the
job scripts are written but not submitted.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from subshowers import generation  # noqa: E402

if __name__ == "__main__":
    generation.command_line(description=__doc__)
