#!/bin/sh

#SBATCH -n 16              # Request 16 cores
#SBATCH -t 0-00:60:00      # Request 60 minutes
#SBATCH -p compute         # Use the "compute" partition
#SBATCH -J Convergence     # Job name
#SBATCH -o Capstone/VOTPP folder/Slurm_convergence/slurm-%j.out
#SBATCH -e Capstone/VOTPP folder/Slurm_convergence/slurm-%j.err

# Activate the Conda environment
source ~/miniconda3/bin/activate Capstone_conda_env2
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Run the MPI script using slurm 
mpirun -np 16 python "VOTPP folder/VOTPP_convergence_runner.py"
