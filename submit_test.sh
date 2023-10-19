#!/bin/sh

#SBATCH -n 2            # Request 2 cores
#SBATCH -t 0-00:05:00   # Request 1 minute (which provides a buffer for the 5-second expected runtime)
#SBATCH -p compute      # Use the "compute" partition
#SBATCH -J test_mpi     # Job name

# Activate the Conda environment
source ~/miniconda3/bin/activate Capstone_conda_env2
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Run the MPI script using slurm # mpirun python test.py
mpirun -np 2 python "Cathal Hogan code/VOTPP_convergence_runner.py"