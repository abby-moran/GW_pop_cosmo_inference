#!/usr/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=h100_pcie:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0-16
#SBATCH -J fullcosmo_evo
#SBATCH -o /mnt/home/misi/src/GW_pop_cosmo_inference/scripts/fullcosmo_evo_%j.log

# Full-freedom inference: h, Omh2, w AND mpisndot all sampled, on the
# endO5_val2 mock data (truth: w = -1, mpisndot = 0).  First end-to-end
# exercise of the 2-D mass-function table (commit 7fc7a6e) with a fully
# traced cosmology.  Expected ~2-4 s/sample at tree depth 7; 16 h is a
# generous cap for 1800+1800 samples x 2 chains.

cd /mnt/home/misi/src/GW_pop_cosmo_inference/scripts/

srun /mnt/home/misi/src/GW_pop_cosmo_inference/.venv/bin/python run_inf.py \
    --config run_configs/mock_O5_fullcosmo_evo.ini
