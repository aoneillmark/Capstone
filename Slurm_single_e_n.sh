#!/bin/sh

#SBATCH -n 128              # Request 128 cores
#SBATCH -t 0-12:00:00      # Request 1 hours and 0 minutes
#SBATCH -p compute         # Use the "compute" partition
#SBATCH -J CombinedJob     # Job name
#SBATCH -o "Capstone/VOTPP folder/Slurm_combined/slurm-%j.out"
#SBATCH -e "Capstone/VOTPP folder/Slurm_combined/slurm-%j.err"

# Activate the Conda environment
source ~/miniconda3/bin/activate Capstone_conda_env2
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Run the MPI script for simulation using slurm 
mpirun -np 128 python "VOTPP folder/VOTPP_convergence_runner_e_n.py"

# Wait for the simulation to complete before starting plotting
wait

# Run the MPI script for plotting using slurm 
mpirun -np 1 python "VOTPP folder/VOTPP_results_and_fit_plotter.py"
