"""
Submit SLURM jobs that split showers into subshowers, as set out in the
subshowers section of a config: one array job per shower, with a task per
energy cut and a time limit to suit the size of the shower.

    python scripts/subshower_generation_from_config.py configs/default.yaml

Sets already made or queued are skipped. Jobs for showers that are still
queued wait for them; showers neither made nor queued are skipped, so use
scripts/generate_data_from_config.py to make those. With --dry-run the job
scripts are written but not submitted.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from subshowers import generation  # noqa: E402

if __name__ == "__main__":
    generation.command_line(make_showers=False, description=__doc__)
