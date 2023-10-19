#!/bin/sh

#SBATCH -n 1            # Request 4 cores
#SBATCH -t 0-00:10:00   # Request 1 minute (which provides a buffer for the 5-second expected runtime)
#SBATCH -p compute      # Use the "compute" partition
#SBATCH -J test_mpi     # Job name

# Activate the Conda environment
source ~/miniconda3/bin/activate Capstone_conda_env2
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Run the MPI script using slurm # mpirun python test.py
# mpirun -np 16 python "VOTPP folder/VOTPP_convergence_runner.py"
mpirun -np 1 python "VOTPP folder/Results/VOTPP_results_plotter.py"
# mpirun -np 4 python "test.py"