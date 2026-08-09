#!/usr/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=h100_pcie:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0-08
#SBATCH -J inf_bbump
#SBATCH -o /mnt/home/misi/src/GW_pop_cosmo_inference/scripts/inf_broadbump_%j.log

# Control for call_inf_narrowbump.sh: same inference on the ORIGINAL
# endO5_val2 data (truth msigma_low = 4), same gwtc5_massonly.prior with
# cosmology and mpisndot pinned.  The two runs differ only in the true bump
# width, so `a` recovery can be compared directly.
#
# No dependency: the data already exists (symlinked from endO5_val2), so this
# can run immediately and in parallel with the narrow-bump chain.
#
# Reads runs/endO5_val2/ read-only through symlinks; writes only to
# runs/endO5_broadbump/.  Does not touch any in-flight run.

cd /mnt/home/misi/src/GW_pop_cosmo_inference/scripts/

srun /mnt/home/misi/src/GW_pop_cosmo_inference/.venv/bin/python run_inf.py \
    --config run_configs/mock_O5_broadbump.ini
