#!/usr/bin/bash
#SBATCH --mem=16G
#SBATCH --time=0-20
#SBATCH -J CE_gen_inj
 
set -euo pipefail
 
module load cuda/12 cudnn
source /mnt/home/amoran/GW_pop_cosmo_inference/.venv/bin/activate
unset LD_LIBRARY_PATH
 
cd /mnt/home/amoran/GW_pop_cosmo_inference/scripts/
 
srun python gen_inj.py --config /mnt/home/amoran/GW_pop_cosmo_inference/scripts/run_configs/CE_ne_fullcosmo.ini
 