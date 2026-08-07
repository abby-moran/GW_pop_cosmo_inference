#!/usr/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=h100_pcie:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0-01
#SBATCH -J f32acc_scale
#SBATCH -o /mnt/home/misi/src/GW_pop_cosmo_inference/scripts/f32acc_scale_%j.log

# Production-scale float32 accuracy probe of the merged recentering +
# selection-pdraw-scale implementation (module im_fast_merged = merge commit
# a0d74a4), truth point, 4 reduction-order permutations + float64 reference.
# Single GPU: the two precision legs run sequentially as subprocesses.

cd /mnt/home/misi/src/GW_pop_cosmo_inference/scripts/

export XLA_PYTHON_CLIENT_PREALLOCATE=false

srun /mnt/home/misi/src/GW_pop_cosmo_inference/.venv/bin/python \
    test_float32_accuracy.py --module im_fast_merged --recenter --point truth
