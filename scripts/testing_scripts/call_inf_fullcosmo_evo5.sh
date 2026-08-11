#!/usr/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=h100_pcie:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1-00
#SBATCH -J fullcosmo_evo5
#SBATCH -o /mnt/home/misi/src/GW_pop_cosmo_inference/scripts/fullcosmo_evo5_%j.log

# Full-scale validation of the pivoted mass-scale prior
# (notes/2026-08-09-pivot-reparam.md) against evo3/evo4.  Same data,
# sampler knobs, and draw count as evo4; only the prior changes
# (mpisn -> mpisn_ref + zpivot=0.75).  Own run_dir so evo4 is untouched.
#
# Geometry bench said ~2.3x fewer grads/ESS on the 2000-event subset;
# wall time should be similar to or shorter than evo4's ~13-14 h.  Keep 24 h.

cd /mnt/home/misi/src/GW_pop_cosmo_inference/scripts/

srun /mnt/home/misi/src/GW_pop_cosmo_inference/.venv/bin/python run_inf.py \
    --config run_configs/mock_O5_fullcosmo_evo5.ini
