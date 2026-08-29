#!/usr/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=0-02
#SBATCH -J max_run

module load cuda/12 cudnn

source /mnt/home/amoran/GW_pop_cosmo_inference/.venv/bin/activate
unset LD_LIBRARY_PATH

cd /mnt/home/amoran/GW_pop_cosmo_inference/scripts/

python injection_daignoistic.py ../runs/mixture-semi_o1_o2-real_o3_o4a_o4b-cartesian_spins_20260410130052UTC-clipped.hdf