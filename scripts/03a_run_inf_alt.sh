#!/usr/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=h100_pcie:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1-00
#SBATCH -J evo7_run_inf_alt
#SBATCH --exclude=workergpu067
#SBATCH -o /mnt/home/amoran/GW_pop_cosmo_inference/scripts/logs/evo7_03_run_inf_%j.log

set -euo pipefail

module load cuda/12 cudnn
source /mnt/home/amoran/GW_pop_cosmo_inference/.venv/bin/activate
unset LD_LIBRARY_PATH

cd /mnt/home/amoran/GW_pop_cosmo_inference/scripts/

srun python run_inf.py --config /mnt/home/amoran/GW_pop_cosmo_inference/scripts/run_configs/mock_O5_fullcosmo_evo7_alt.ini