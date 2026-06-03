#!/usr/bin/bash
#SBATCH -p gpu
#SBATCH --gpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=0-20

module load cuda/12 cudnn

source /mnt/home/amoran/GW_pop_cosmo_inference/.venv/bin/activate
unset LD_LIBRARY_PATH

cd /mnt/home/amoran/GW_pop_cosmo_inference/scripts/

srun python reweight_inj.py --config /mnt/home/amoran/GW_pop_cosmo_inference/scripts/run_configs/confg_mocko5_evo_errnj_mmin4.ini