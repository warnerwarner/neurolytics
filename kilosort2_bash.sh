#!/bin/bash
#
#SBATCH --job-name=kilosort2
#SBATCH --array=1		## N_TASKS_TOTAL: SET THIS TO THE NUMBER OF INDEPENDENT JOBS,TYPICLLAY 100,SEE BELOW N_TASKS_TOTAL
#SBATCH --output=/home/camp/warnert/neurolytics/ks_output.txt	##
#SBATCH --error=/home/camp/warnert/neurolytics/ks_error.txt
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=100G
#SBATCH --partition=gpu
ml matlab
ml ml CUDA/10.0.130

matlab -nodisplay -nosplash -nodesktop -r "run('/home/camp/warnert/working/Recordings/binary_pulses/200303/2020-03-03_19-57-03/kilosort2_master.m');"
