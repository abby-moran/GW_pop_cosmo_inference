#!/usr/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH -J rG5_lvk_plz_plots
#SBATCH -o /mnt/home/misi/src/GW_pop_cosmo_inference/scripts/realGWTC5_lvk_plz_plots_%j.log

# Post-processing plots for the finished realGWTC5_lvk_plz inference (LVK
# 'PowerLaw + 2 Peaks' mass model with pure power-law redshift evolution,
# output ../runs/realGWTC5_lvk_plz/realGWTC5_lvk_plz.nc): per-chain trace
# plots and marginal PPDs with the official LVK GWTC-5.0 Default BBH overlay.
# Small CPU-only job; chained automatically by ./submit_run.sh.

set -e

# CPU node: keep JAX off CUDA (plot_ppd --marginal initializes JAX).
export JAX_PLATFORMS=cpu

cd /mnt/home/misi/src/GW_pop_cosmo_inference/scripts/

PYTHON=/mnt/home/misi/src/GW_pop_cosmo_inference/.venv/bin/python

srun $PYTHON plot_trace.py --run realGWTC5_lvk_plz
srun $PYTHON plot_ppd.py --run realGWTC5_lvk_plz --no_truth --lvk
