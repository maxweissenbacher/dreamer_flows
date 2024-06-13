#!/bin/bash
#PBS -l walltime=1:00:00
#PBS -l select=1:ncpus=4:mem=20gb:ngpus=1:gpu_type=RTX6000

# If run no CPU, remove 'ngpus' and 'gpu_type' from above 


cd $PBS_O_WORKDIR


# Cluster Environment Setup
module load anaconda3/personal
source activate RLf2
NUM_PORT=65

export LD_LIBRARY_PATH=~/anaconda3/envs/RLf2/lib/:$LD_LIBRARY_PATH

# Run the python code
python baseline_single_run.py 

echo "Launched training!"

exit 0
