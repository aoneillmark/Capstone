#!/bin/sh

#SBATCH -n 1            # Request 1 cores
#SBATCH -t 0-00:5:00   # Request 10 seconds
#SBATCH -p compute      # Use the "compute" partition
#SBATCH -J Plotting     # Job name
#SBATCH -o "Capstone/VOTPP folder/Slurm_plotting/slurm-%j.out"
#SBATCH -e "Capstone/VOTPP folder/Slurm_plotting/slurm-%j.err"

# Activate the Conda environment
source ~/miniconda3/bin/activate Capstone_conda_env2
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Run the MPI script using slurm 
mpirun -np 1 python "VOTPP folder/VOTPP_results_plotter.py"