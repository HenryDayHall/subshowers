#!/bin/bash
#SBATCH --time REPLACE_WITH_TIME_LIMIT
#SBATCH --nodes 1
#SBATCH --partition maxcpu
#SBATCH --job-name eas_gen_ss
#SBATCH --output /data/dust/user/%u/eas/joblogs/eas_gen_ss_%A_%a.out
#SBATCH --error /data/dust/user/%u/eas/joblogs/eas_gen_ss_%A_%a.err
#SBATCH --array=REPLACE_WITH_ARRAY_RANGE

# Splits one shower into subshowers, one array task per energy cut.
# subshowers/generation.py fills in the REPLACE_WITH_ values, with a time
# limit to suit the shower's size (see subshower_minutes there), and submits
# a copy, which waits for the shower's job if it's still being made.


start_time=$(date +%s)

REPO_FOLDER=REPLACE_WITH_REPO_FOLDER
LEAVES_ONLY=REPLACE_WITH_LEAVES_ONLY
CONDA_ENV=corsika
echo "Task ID is $SLURM_ARRAY_TASK_ID, REPO_FOLDER is $REPO_FOLDER"
set -euo pipefail

possible_output_files=(
REPLACE_WITH_OUTPUT_FILES
)

output_file=${possible_output_files[$SLURM_ARRAY_TASK_ID]}
# each output sits in the folder of the shower it splits,
# and is named subshowers_e<energy cut>.h5
shower_folder=${output_file%/*}
file_name=${output_file##*/}
energy_cut=${file_name#subshowers_e}
energy_cut=${energy_cut%.h5}


node_name=$(hostname -s)

echo "$output_file $node_name $energy_cut $LEAVES_ONLY"
command_record="${output_file%.h5}_commands.txt"

# subshowers.run can't write over an existing output, so the commands start
# by removing anything left by an earlier attempt at this output
commands=$(cat <<EOF
cd "${REPO_FOLDER}"
set +u
[[ \$(type -t conda) == function ]] || eval "\$("\${CONDA_EXE:-conda}" shell.bash hook)"
conda activate ${CONDA_ENV}
set -u

rm -f "${output_file}" "${output_file%.h5}.yaml"
python -u -c '
import sys
from subshowers.subshowers import run
folder, energy_cut, leaves_only, output_path = sys.argv[1:]
run(folder, float(energy_cut), leaves_only={"True": True, "False": False}[leaves_only],
    output_path=output_path, verbose=True)
' "${shower_folder}" "${energy_cut}" "${LEAVES_ONLY}" "${output_file}"
EOF
)
echo "${commands}" > "${command_record}"

set +e
eval "${commands}"
status=$?
set -e

end_time=$(date +%s)
duration=$((end_time-start_time))
echo "Job took $duration seconds" >> "${command_record}"
exit "$status"
