#!/usr/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=h100_pcie:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0-16
#SBATCH -J evo_inf
#SBATCH -o /mnt/home/misi/src/GW_pop_cosmo_inference/scripts/evo_inf_%j.log

# Cosmology pinned to truth, mpisndot free.  Companion to fullcosmo_evo
# (both free) and fullcosmo (cosmo free, mpisndot=0).

cd /mnt/home/misi/src/GW_pop_cosmo_inference/scripts/

srun /mnt/home/misi/src/GW_pop_cosmo_inference/.venv/bin/python run_inf.py \
    --config run_configs/mock_O5_evo.ini
