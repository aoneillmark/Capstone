#!/bin/sh

#SBATCH -n 2            # Request 2 cores
#SBATCH -t 0-00:01:00   # Request 1 minute (which provides a buffer for the 5-second expected runtime)
#SBATCH -p compute      # Use the "compute" partition
#SBATCH -J test_mpi     # Job name

# Activate the Conda environment
source ~/miniconda3/bin/activate Capstone_conda_env

# Run the MPI script using slurm # mpirun python test.py
srun -n 3 python test.py