#!/bin/sh

#SBATCH -n 128              # Request 32 cores
#SBATCH -t 0-12:00:00      # Request 12 hours
#SBATCH -p compute         # Use the "compute" partition
#SBATCH -J Convergence     # Job name
#SBATCH -o "Capstone/VOTPP folder/Slurm_convergence/slurm-%j.out"
#SBATCH -e "Capstone/VOTPP folder/Slurm_convergence/slurm-%j.err"

# Activate the Conda environment
source ~/miniconda3/bin/activate Capstone_conda_env2
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Run the MPI script using slurm 
mpirun -np 128 python "VOTPP folder/VOTPP_convergence_runner.py"
