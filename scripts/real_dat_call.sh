#!/usr/bin/bash
#SBATCH --mem=16G
#SBATCH --time=0-20

module load cuda/12 cudnn

source /mnt/home/amoran/GW_pop_cosmo_inference/.venv/bin/activate
unset LD_LIBRARY_PATH

cd /mnt/home/amoran/GW_pop_cosmo_inference/scripts/

srun python real_dat_run.py --config /mnt/home/amoran/GW_pop_cosmo_inference/scripts/run_configs/confg_realTC5_cosmo.ini
