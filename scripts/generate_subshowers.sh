#!/bin/bash
#SBATCH --time 01:20:00
#SBATCH --nodes 1
#SBATCH --partition maxcpu
#SBATCH --job-name subshowers
#SBATCH --output /data/dust/user/dayhallh/data/seperate_hadronic/joblogs/subshowers_%j.out      # terminal output
#SBATCH --error /data/dust/user/dayhallh/data/seperate_hadronic/joblogs/subshowers_%j.err
#SBATCH --array=3-26


# if this is run as a scrpt by sourceing it, set the SLURM_ARRAY_TASK_ID
if [ -z "$SLURM_ARRAY_TASK_ID" ]
then
    echo "SLURM_ARRAY_TASK_ID set to 0"
    SLURM_ARRAY_TASK_ID=0
fi

start_time=`date +%s`

outer_folder="/data/dust/user/dayhallh/eas/data/examples"
possible_subfolders=('example_10e4_photon_hist1' 'example_10e4_photon_hist2' 'example_10e4_photon_hist3' 'example_10e4_photon_hist4' 'example_10e8_photon_hist1')
total_batches_per_subfolder=2
subfolder_varient=$(($SLURM_ARRAY_TASK_ID / $total_batches_per_subfolder))
subfolder=${possible_subfolders[$subfolder_varient]}
echo "subfolder_varient: ${subfolder_varient} subfolder: ${subfolder}"
batch_number=$(($SLURM_ARRAY_TASK_ID - $subfolder_varient * $total_batches_per_subfolder))
folder=${outer_folder}/${subfolder}
possible_energies=('100' '500' '900')
energy=${possible_energies[$batch_number]}

command_record="$folder/generate_subshowers_e${energy}_command.txt"
node_name=$(hostname -s)


commands=\
"""
Commands used;
cd /home/dayhallh/eas/subshowers
source /etc/profile.d/modules.sh
module load maxwell cuda/12.6
module load maxwell mamba
. mamba-init
mamba activate corsika

python3 -c 'from subshowers.subshowers import run; run('${folder}', '${energy}')'
"""
echo "${commands}" >> ${command_record}

cd /home/dayhallh/eas/subshowers
source /etc/profile.d/modules.sh
module load maxwell cuda/12.6
module load maxwell mamba
. mamba-init
mamba activate corsika

python3 -c 'from subshowers.subshowers import run; run("'${folder}'", '${energy}')'

end_time=`date +%s`
duration=$((end_time-start_time))
echo "Job took $duration seconds" >> ${command_record}

