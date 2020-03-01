#!/bin/bash
#
#SBATCH --job-name=kilosort2
#SBATCH --array=1		## N_TASKS_TOTAL: SET THIS TO THE NUMBER OF INDEPENDENT JOBS,TYPICLLAY 100,SEE BELOW N_TASKS_TOTAL
#SBATCH --output=/home/camp/warnert/unit_scripts/dat_maker_output.txt	##
#SBATCH --error=/home/camp/warnert/unit_scripts/dat_maker_error_%a.txt
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=100G
#SBATCH --partition=gpu
ml matlab
ml ml CUDA/10.0.130

matlab -nodisplay -nosplash -nodesktop -r "run('/home/camp/warnert/working/Recordings/binary_pulses/200228/2020-02-28_19-56-29/kilosort2_master.m');"
