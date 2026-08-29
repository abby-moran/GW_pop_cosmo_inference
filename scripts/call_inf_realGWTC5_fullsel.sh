#!/usr/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=h100_pcie:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH -J realGWTC5_fullsel
#SBATCH -o /mnt/home/misi/src/GW_pop_cosmo_inference/scripts/realGWTC5_fullsel_%j.log

# Real GWTC-5 catalog, no-evolution model, cosmology fixed.
# Mimics abbys_runs/GWTC5_gc_reparam_noevo but with the FULL selection set
# (all 488839 rows of sel_GWTC5.h5), a prior widened where Abby's posterior
# railed (real_dat_noevo_fullsel.prior), and 4 chains on 4 GPUs.

cd /mnt/home/misi/src/GW_pop_cosmo_inference/scripts/

srun /mnt/home/misi/src/GW_pop_cosmo_inference/.venv/bin/python run_inf.py \
    --config run_configs/realGWTC5_noevo_fullsel.ini
