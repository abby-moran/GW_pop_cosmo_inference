#!/usr/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=h100_pcie:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1-00
#SBATCH -J fullcosmo_evo7


module load cuda/12 cudnn

source /mnt/home/amoran/GW_pop_cosmo_inference/.venv/bin/activate
unset LD_LIBRARY_PATH

cd /mnt/home/amoran/GW_pop_cosmo_inference/scripts/

srun python run_inf.py --config /mnt/home/amoran/GW_pop_cosmo_inference/scripts/run_configs/mock_O5_fullcosmo_evo7.ini
