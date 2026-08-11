#!/usr/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=h100_pcie:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0-04
#SBATCH -J reparam_pivot
#SBATCH -o /mnt/home/misi/src/GW_pop_reparam/scripts/reparam_pivot_%j.log

# Geometry bench: pivot_tight variant on the endO5_val2 mock.
# 1x H100 (bench uses num_chains=1). Isolated from evo4
# (different job/node/log; read-only PE/sel paths).

cd /mnt/home/misi/src/GW_pop_reparam/scripts/

srun /mnt/home/misi/src/GW_pop_cosmo_inference/.venv/bin/python -u bench_reparam.py \
    --real --nobs 2000 --zpivot 0.75 \
    --steps 150 --max_tree_depth 10 \
    --variants pivot_tight --seed 2
