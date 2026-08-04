#!/usr/bin/bash
#SBATCH -p gpu
#SBATCH --gpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --time=0-02

module load cuda/12 cudnn

source /mnt/home/amoran/GW_pop_cosmo_inference/.venv/bin/activate
unset LD_LIBRARY_PATH

cd /mnt/home/amoran/GW_pop_cosmo_inference/scripts/

srun python reweight_res.py --config /mnt/home/amoran/GW_pop_cosmo_inference/scripts/run_configs/mock_O5_noevo_nb.ini