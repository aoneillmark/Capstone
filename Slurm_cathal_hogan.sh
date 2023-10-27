#!/bin/sh

#SBATCH -n 25              # Request 128 cores
#SBATCH -t 0-12:05:00      # Request 12 hours and 5 minutes
#SBATCH -p compute         # Use the "compute" partition
#SBATCH -J CombinedJob     # Job name
#SBATCH -o "Capstone/VOTPP folder/Slurm_combined/slurm-%j.out"
#SBATCH -e "Capstone/VOTPP folder/Slurm_combined/slurm-%j.err"

# Activate the Conda environment
source ~/miniconda3/bin/activate Capstone_conda_env2
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Run the MPI script for simulation using slurm 
mpirun -np 25 python "Cathal Hogan code/tbs n-n T2 MULTIPLE SPINS.py"

# Wait for the simulation to complete before starting plotting
wait

# Run the MPI script for plotting using slurm 
mpirun -np 1 python "VOTPP folder/results_plotter_temp.py"
