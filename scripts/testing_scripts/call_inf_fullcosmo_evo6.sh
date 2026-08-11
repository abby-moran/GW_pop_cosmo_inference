#!/usr/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=h100_pcie:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1-00
#SBATCH -J fullcosmo_evo6
#SBATCH -o /mnt/home/misi/src/GW_pop_cosmo_inference/scripts/fullcosmo_evo6_%j.log

# Twin of fullcosmo_evo5 with log_fpeak instead of log_flow
# (notes/2026-08-09-log-fpeak-parametrization.md).  Same pivot prior,
# data, and sampler knobs; distinct job/log/run_dir so evo4/evo5 are
# untouched.

cd /mnt/home/misi/src/GW_pop_cosmo_inference/scripts/

srun /mnt/home/misi/src/GW_pop_cosmo_inference/.venv/bin/python run_inf.py \
    --config run_configs/mock_O5_fullcosmo_evo6.ini
