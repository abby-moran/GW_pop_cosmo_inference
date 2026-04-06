#!/usr/bin/bash
#SBATCH -p gpu
#SBATCH --gpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=0-20

module load cuda/12 cudnn

source ~/miniforge3/bin/activate GW_all
unset LD_LIBRARY_PATH

cd /mnt/home/amoran/GW_pop_cosmo_inference/src/

srun python gen_inj.py
