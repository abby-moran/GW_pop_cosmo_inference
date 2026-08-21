#!/usr/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=h100_pcie:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=2-00
#SBATCH -J nbump_s2sel_f64
#SBATCH -o /mnt/home/misi/src/GW_pop_cosmo_inference/scripts/narrowbump_s2sel_f64_%j.log

# Float64 twin of endO5_narrowbump_s2sel.  Same PE/sel/prior/sampler;
# JAX_ENABLE_X64 must be set before jax is imported (run_inf.py imports
# jax at module load).  XLA_PYTHON_CLIENT_PREALLOCATE=false avoids the
# spurious OOM the float32-accuracy audit hit on the float64 leg.
# See notes/2026-08-07-float32-accuracy-audit.md.

cd /mnt/home/misi/src/GW_pop_cosmo_inference/scripts/

export JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false

srun /mnt/home/misi/src/GW_pop_cosmo_inference/.venv/bin/python run_inf.py \
    --config run_configs/mock_O5_narrowbump_s2sel_f64.ini
