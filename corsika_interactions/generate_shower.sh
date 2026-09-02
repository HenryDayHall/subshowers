#!/bin/bash
#SBATCH --time 0-01:00:00
#SBATCH --nodes 1
#SBATCH --partition maxcpu
#SBATCH --job-name eas_gen_s
#SBATCH --output /data/dust/user/dayhallh/eas/joblogs/eas_gen_s_%j.out      # terminal output
#SBATCH --error /data/dust/user/dayhallh/eas/joblogs/eas_gen_s_%j.err       # error output
#SBATCH --array=2

start_time=`date +%s`

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

output_folder=/data/dust/user/dayhallh/eas/data/PDG${pdg}/e${energy}_${shower_number}
node_name=$(hostname -s)

echo $output_folder
command_record=${output_folder}_commands.txt

commands=\
"""
cd /data/dust/user/dayhallh/eas/
source setup.sh
cd /data/dust/user/dayhallh/eas/corsika-build/applications

./c8_air_shower_with_history --pdg ${pdg} -E ${energy} -f ${output_folder} --seed ${shower_number}
"""
echo "${commands}" > ${command_record}

cd /data/dust/user/dayhallh/eas/
source setup.sh
cd /data/dust/user/dayhallh/eas/corsika-build/applications

./c8_air_shower_with_history --pdg ${pdg} -E ${energy} -f ${output_folder} --seed ${shower_number}

end_time=`date +%s`
duration=$((end_time-start_time))
echo "Job took $duration seconds" >> ${command_record}
