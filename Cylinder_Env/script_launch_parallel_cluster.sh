#!/bin/bash
#PBS -l walltime=00:20:00
#PBS -l select=1:ncpus=70:mem=50gb
#######:ngpus=1:gpu_type=RTX6000

# If run no CPU, remove 'ngpus' and 'gpu_type' from above 


cd $PBS_O_WORKDIR


# Cluster Environment Setup
module load anaconda3/personal
source activate dreamer_cyl
NUM_PORT=65

export LD_LIBRARY_PATH=~/anaconda3/envs/dreamer_cyl/lib/:$LD_LIBRARY_PATH

# Run the python code
python launch_parallel_training.py -n $NUM_PORT -s testrun2

echo "Launched training!"

exit 0
