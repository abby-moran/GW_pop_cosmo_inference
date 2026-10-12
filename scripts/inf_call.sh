#!/usr/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --time=0-06
#SBATCH -J asharp
#SBATCH --constraint=h100
#SBATCH --exclude=workergpu067

module load cuda/12 cudnn

source /mnt/home/amoran/GW_pop_cosmo_inference/.venv/bin/activate
unset LD_LIBRARY_PATH

cd /mnt/home/amoran/GW_pop_cosmo_inference/scripts/

srun python run_inf.py --config /mnt/home/amoran/GW_pop_cosmo_inference/scripts/run_configs/CE_ne.ini
