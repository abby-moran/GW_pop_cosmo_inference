#!/usr/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=2
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --time=2-00
#SBATCH --constraint=h100
#SBATCH -J nb_nocosmo
#SBATCH --exclude=workergpu067
#SBATCH -o /mnt/home/amoran/GW_pop_cosmo_inference/scripts/logs/nb_nocosmo_run_inf%j.log

set -euo pipefail

module load cuda/12 cudnn
source /mnt/home/amoran/GW_pop_cosmo_inference/.venv/bin/activate
unset LD_LIBRARY_PATH

cd /mnt/home/amoran/GW_pop_cosmo_inference/scripts/

srun python run_inf.py --config /mnt/home/amoran/GW_pop_cosmo_inference/scripts/run_configs/CE_ne.ini