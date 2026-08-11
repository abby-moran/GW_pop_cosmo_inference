#!/usr/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=h100_pcie:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0-08
#SBATCH -J val_inf
#SBATCH -o /mnt/home/misi/src/GW_pop_cosmo_inference/scripts/val_inf_%j.log

cd /mnt/home/misi/src/GW_pop_cosmo_inference/scripts/

srun /mnt/home/misi/src/GW_pop_cosmo_inference/.venv/bin/python run_inf.py \
    --config run_configs/mock_O5_noevo_val.ini
