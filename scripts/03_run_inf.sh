#!/usr/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=4
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --time=1-00
#SBATCH --constraint=h100
#SBATCH -J ne_nocosmo
#SBATCH --exclude=workergpu067
#SBATCH -o /mnt/home/amoran/GW_pop_cosmo_inference/scripts/logs/ne_nocosmo_run_inf%j.log

set -euo pipefail

module load cuda/12 cudnn
source /mnt/home/amoran/GW_pop_cosmo_inference/.venv/bin/activate
unset LD_LIBRARY_PATH

cd /mnt/home/amoran/GW_pop_cosmo_inference/scripts/

srun python run_inf.py --config /mnt/home/amoran/GW_pop_cosmo_inference/scripts/run_configs/mock_O5_ne.ini