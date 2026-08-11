#!/usr/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=h100_pcie:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1-00
#SBATCH -J fullcosmo_evo3
#SBATCH -o /mnt/home/misi/src/GW_pop_cosmo_inference/scripts/fullcosmo_evo3_%j.log

# Rerun of endO5_fullcosmo_evo2 with dense_mass=true and max_tree_depth=10.
# evo2 recovered the truth (17/18 params in 95%, 0 divergences) but mixed
# poorly: every iteration saturated depth 7, bulk ESS 3-18, r-hat up to 1.9.
# Worst case here is 8x the leapfrog steps of evo2 (2h12m), i.e. ~18h;
# dense mass should let trajectories U-turn well before depth 10.

cd /mnt/home/misi/src/GW_pop_cosmo_inference/scripts/

srun /mnt/home/misi/src/GW_pop_cosmo_inference/.venv/bin/python run_inf.py \
    --config run_configs/mock_O5_fullcosmo_evo3.ini
