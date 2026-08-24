#!/usr/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --time=0-06
#SBATCH -J evo7_reweight
#SBATCH -o /mnt/home/amoran/GW_pop_cosmo_inference/scripts/logs/O5_fullcosmo_reweight_%j.log

set -euo pipefail

module load cuda/12 cudnn
source /mnt/home/amoran/GW_pop_cosmo_inference/.venv/bin/activate
unset LD_LIBRARY_PATH

cd /mnt/home/amoran/GW_pop_cosmo_inference/scripts/

srun python reweight_res.py --config /mnt/home/amoran/GW_pop_cosmo_inference/scripts/run_configs/mock_O5_ne.ini