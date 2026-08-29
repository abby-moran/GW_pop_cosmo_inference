#!/usr/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=h100_pcie:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH -J realGWTC5_fullsel2
#SBATCH -o /mnt/home/misi/src/GW_pop_cosmo_inference/scripts/realGWTC5_fullsel2_%j.log

# Real GWTC-5 catalog, no-evolution model, cosmology fixed.
# Same as call_inf_realGWTC5_fullsel.sh but with the CORRECTED selection set
# (sel_GWTC5_fixed.h5, 1,433,314 rows, fix-selection-normalization branch):
# full FAR/SNR detection mask, true-m2 cut, spin-marginalization factor, and
# no spurious pdraw rescale.  Selection fix is the only change vs the old run.

cd /mnt/home/misi/src/GW_pop_cosmo_inference/scripts/

srun /mnt/home/misi/src/GW_pop_cosmo_inference/.venv/bin/python run_inf.py \
    --config run_configs/realGWTC5_noevo_fullsel2.ini
