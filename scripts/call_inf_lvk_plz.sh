#!/usr/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=h100_pcie:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH -J rG5_lvk_plz
#SBATCH -o /mnt/home/misi/src/GW_pop_cosmo_inference/scripts/realGWTC5_lvk_plz_%j.log

# Real GWTC-5 catalog (259 events), LVK 'PowerLaw + 2 Peaks' mass model with a
# PURE POWER-LAW redshift evolution dN/dV/dt ~ (1+z)^lam (run_inf_lvk.py),
# cosmology fixed.  Identical data/sampler settings to call_inf_lvk_control.sh,
# so that pair differs ONLY in the redshift sector (no kappa/dkappa, no zp).
# 4 chains on 4 GPUs.

cd /mnt/home/misi/src/GW_pop_cosmo_inference/scripts/

srun /mnt/home/misi/src/GW_pop_cosmo_inference/.venv/bin/python run_inf_lvk.py \
    --config run_configs/realGWTC5_lvk_plz.ini
