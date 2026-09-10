#!/bin/bash
#SBATCH --time 0-12:00:00
#SBATCH --nodes 1
#SBATCH --partition maxcpu
#SBATCH --job-name eas_gen_s
#SBATCH --output /data/dust/user/dayhallh/eas/joblogs/eas_gen_s_%A_%a.out
#SBATCH --error /data/dust/user/dayhallh/eas/joblogs/eas_gen_s_%A_%a.err
#SBATCH --array=0-2

set -euo pipefail

start_time=$(date +%s)

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

#energy=1e5  # takes about 30 mins cpu
#energy=1e6  # takes 270 mins, or 4.5 hours, cpu
#energy=1e7  # takes 572 mins, or 9.5 hours, cpu
#energy=1e8  # takes more than 2 days, cpu > actual time unknown

node_name=$(hostname -s)

echo "$output_folder $node_name $pdg $energy $shower_number"
command_record="${output_folder}_commands.txt"

mkdir -p "${output_folder}"

commands=$(cat <<EOF
cd /data/dust/user/dayhallh/eas/
source setup.sh
cd /data/dust/user/dayhallh/eas/corsika-build/applications

./c8_air_shower_with_history --pdg "${pdg}" -E "${energy}" -f "${output_folder}" --seed "${shower_number}"
EOF
)
echo "${commands}" > "${command_record}"

cd /data/dust/user/dayhallh/eas/
source setup.sh
cd /data/dust/user/dayhallh/eas/corsika-build/applications

set +e
./c8_air_shower_with_history --pdg "${pdg}" -E "${energy}" -f "${output_folder}" --seed "${shower_number}"
status=$?
set -e

end_time=$(date +%s)
duration=$((end_time-start_time))
echo "Job took $duration seconds" >> "${command_record}"
exit $status
