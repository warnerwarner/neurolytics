#!/bin/bash
#
#SBATCH --job-name={job_name}
#SBATCH --output={output_loc}
#SBATCH --error={error_loc}
#SBATCH --ntasks={ntasks}
#SBATCH --array={array}
#SBATCH --time={time}
#SBATCH --mem={memory}
#SBATCH --partition={partition}

conda activate {enviroment}

python {file_name}