"""
Submit SLURM jobs that generate the showers in a config's showers section,
one array job per primary energy, each with a time limit to suit the energy.

    python corsika_interactions/generation_from_config.py configs/default.yaml

Showers already made or queued are skipped. With --dry-run the job scripts
are written but not submitted. To make the subshowers as well, with jobs that
wait for these showers, use scripts/generate_data_from_config.py instead.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from subshowers import generation  # noqa: E402

if __name__ == "__main__":
    generation.command_line(make_subshowers=False, description=__doc__)
