#!/usr/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=h100_pcie:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0-04
#SBATCH -J val_true
#SBATCH -o /mnt/home/misi/src/GW_pop_cosmo_inference/scripts/val_true_%j.log

cd /mnt/home/misi/src/GW_pop_cosmo_inference/scripts/

srun /mnt/home/misi/src/GW_pop_cosmo_inference/.venv/bin/python run_inf_true.py \
    --config run_configs/mock_O5_noevo_val.ini --nobs 9000 --nmcmc 1000 --nchain 2
