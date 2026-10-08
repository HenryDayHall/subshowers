#!/bin/bash
#SBATCH --time 0-01:00:00
#SBATCH --nodes 1
#SBATCH --partition maxcpu
#SBATCH --job-name eas_gen_s
#SBATCH --output /data/dust/user/%u/eas/joblogs/eas_gen_s_%A_%a.out
#SBATCH --error /data/dust/user/%u/eas/joblogs/eas_gen_s_%A_%a.err
#SBATCH --array=0-2


start_time=$(date +%s)

# if SLURM_ARRAY_TASK_ID is not set, it will be 0
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    SLURM_ARRAY_TASK_ID=1
fi
# if EAS_BASE_FOLDER is not set, pick a username based default
if [ -z "$EAS_BASE_FOLDER" ]; then
    EAS_BASE_FOLDER="/data/dust/user/$USER/eas"
fi
echo "Task ID is $SLURM_ARRAY_TASK_ID, EAS_BASE_FOLDER is $EAS_BASE_FOLDER"
set -euo pipefail

#              15    30    45    60   75    90     105   120   135    150     165    180    195
#              electrons, photons, protons, iron
possible_pdgs=('11' '22' '2212' '1000260560')
total_showers_per_pdg=15
pdg_varient=$(($SLURM_ARRAY_TASK_ID / $total_showers_per_pdg))
pdg=${possible_pdgs[$pdg_varient]}
shower_number=$(($SLURM_ARRAY_TASK_ID - $pdg_varient * $total_showers_per_pdg))

energy=1e5  # takes about 30 mins cpu
#energy=1e6  # takes 270 mins, or 4.5 hours, cpu
#energy=1e7  # takes 572 mins, or 9.5 hours, cpu
#energy=1e8  # takes more than 2 days, cpu > actual time unknown

output_folder=${EAS_BASE_FOLDER}/data/PDG${pdg}/e${energy}_${shower_number}
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
source setup_env.sh
cd ${EAS_BASE_FOLDER}/corsika-build/applications

set +e
./c8_air_shower_with_history --pdg "${pdg}" -E "${energy}" -f "${output_folder}" --seed "${shower_number}"
status=$?
set -e

end_time=$(date +%s)
duration=$((end_time-start_time))
echo "Job took $duration seconds" >> "${command_record}"
