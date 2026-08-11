#!/usr/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=h100_pcie:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0-08
#SBATCH -J inf_nbump
#SBATCH -o /mnt/home/misi/src/GW_pop_cosmo_inference/scripts/inf_narrowbump_%j.log

# Step 2 of 2: infer the mass-function parameters from the narrow-bump mock.
# MUST run after call_rwt_narrowbump.sh has produced
# runs/endO5_narrowbump/{PE_narrowbump.h5,sel_narrowbump.h5}.  Chain it with:
#
#   sbatch --dependency=afterok:<rwt_jobid> call_inf_narrowbump.sh
#
# Cosmology and mpisndot pinned to truth (priors/gwtc5_massonly.prior), so this
# runs on the fast 1-D mass table with 15 free parameters.  The comparable
# cosmology-pinned run (endO5_evo, mpisndot free) took 59 min on 2x H100;
# pinning mpisndot as well should make this faster.  8 h is a generous cap.
#
# max_tree_depth/dense_mass left at the defaults (7 / False): the pinned
# geometry is well conditioned (endO5_evo2 reached bulk ESS 118 at depth 7).
# If the output shows depth saturation with low ESS, add
#   max_tree_depth = 10
#   dense_mass = true
# to run_configs/mock_O5_narrowbump.ini and resubmit.

cd /mnt/home/misi/src/GW_pop_cosmo_inference/scripts/

srun /mnt/home/misi/src/GW_pop_cosmo_inference/.venv/bin/python run_inf.py \
    --config run_configs/mock_O5_narrowbump.ini
