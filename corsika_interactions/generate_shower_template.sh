#!/bin/bash
#SBATCH --time 0-12:00:00
#SBATCH --nodes 1
#SBATCH --partition maxcpu
#SBATCH --job-name eas_gen_s
#SBATCH --output /data/dust/user/%u/eas/joblogs/eas_gen_s_%A_%a.out
#SBATCH --error /data/dust/user/%u/eas/joblogs/eas_gen_s_%A_%a.err
#SBATCH --array=0-2

set -euo pipefail

start_time=$(date +%s)

# if EAS_BASE_FOLDER is not set, pick a username based default
if [ -z "$EAS_BASE_FOLDER" ]; then
    EAS_BASE_FOLDER="/data/dust/user/$USER/eas"
fi
echo "Task ID is $SLURM_ARRAY_TASK_ID, EAS_BASE_FOLDER is $EAS_BASE_FOLDER"

possible_output_folders=(
REPLACE_THIS_LINE
)

output_folder=${possible_output_folders[$SLURM_ARRAY_TASK_ID]}
# trim a trailing slash if any
output_folder=${output_folder%/}
energy_dir_name=${output_folder##*/}
energy=${energy_dir_name#e}
energy=${energy%_*}
shower_number=${energy_dir_name#*_}

pdg_dir_name=${output_folder%/*}
pdg_dir_name=${pdg_dir_name##*/}
pdg=${pdg_dir_name#PDG}


node_name=$(hostname -s)

echo "$output_folder $node_name $pdg $energy $shower_number"
command_record="${output_folder}_commands.txt"

mkdir -p "${output_folder}"

commands=$(cat <<EOF
cd ${EAS_BASE_FOLDER}
source setup_env.sh
cd ${EAS_BASE_FOLDER}/corsika-build/applications

./c8_air_shower_with_history --pdg "${pdg}" -E "${energy}" -f "${output_folder}" --seed "${shower_number}"
EOF
)
echo "${commands}" > "${command_record}"

cd ${EAS_BASE_FOLDER}
source setup.sh
cd ${EAS_BASE_FOLDER}/corsika-build/applications

set +e
./c8_air_shower_with_history --pdg "${pdg}" -E "${energy}" -f "${output_folder}" --seed "${shower_number}"
status=$?
set -e

end_time=$(date +%s)
duration=$((end_time-start_time))
echo "Job took $duration seconds" >> "${command_record}"
exit $status
