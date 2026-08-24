#!/usr/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=0-20
#SBATCH -J evo7_gen_inj
#SBATCH -o /mnt/home/amoran/GW_pop_cosmo_inference/scripts/logs/O5_fullcosmo_gen_inj_%j.log
 
set -euo pipefail
 
module load cuda/12 cudnn
source /mnt/home/amoran/GW_pop_cosmo_inference/.venv/bin/activate
unset LD_LIBRARY_PATH
 
cd /mnt/home/amoran/GW_pop_cosmo_inference/scripts/
 
srun python gen_inj.py --config /mnt/home/amoran/GW_pop_cosmo_inference/scripts/run_configs/mock_O5_ne.ini
 