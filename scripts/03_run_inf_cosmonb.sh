#!/usr/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=4
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --time=0-10
#SBATCH --constraint=h100
#SBATCH -J nb_cosmo
#SBATCH --exclude=workergpu067
#SBATCH --exclude=workergpu220

set -euo pipefail

module load cuda/12 cudnn
source /mnt/home/amoran/GW_pop_cosmo_inference/.venv/bin/activate
unset LD_LIBRARY_PATH

cd /mnt/home/amoran/GW_pop_cosmo_inference/scripts/

srun python run_inf.py --config /mnt/home/amoran/GW_pop_cosmo_inference/scripts/run_configs/mock_O5_ne_nb_cosmo.ini