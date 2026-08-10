#!/usr/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=h100_pcie:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1-00
#SBATCH -J fullcosmo_evo4
#SBATCH -o /mnt/home/misi/src/GW_pop_cosmo_inference/scripts/fullcosmo_evo4_%j.log

# Parallel validation of the scatter-free VJP optimization
# (notes/2026-08-09-scatter-free-vjp.md) against the still-running
# endO5_fullcosmo_evo3 baseline.  Distinct job name, log, run_dir, and
# NetCDF output; Slurm allocates a fresh 2x H100 set.  Do not touch
# runs/endO5_fullcosmo_evo3/.
#
# Bench said ~19% faster grads with free mpisndot (37.5 -> 30.3 ms); at
# evo3's ~16.5 s/it that projects to ~13-14 h wall time.  Keep 24 h.

cd /mnt/home/misi/src/GW_pop_cosmo_inference/scripts/

srun /mnt/home/misi/src/GW_pop_cosmo_inference/.venv/bin/python run_inf.py \
    --config run_configs/mock_O5_fullcosmo_evo4.ini
