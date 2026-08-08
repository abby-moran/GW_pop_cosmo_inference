#!/usr/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=h100_pcie:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0-16
#SBATCH -J evo_inf2
#SBATCH -o /mnt/home/misi/src/GW_pop_cosmo_inference/scripts/evo_inf2_%j.log

# Rerun of endO5_evo (cosmology pinned to truth, mpisndot free) after the
# tabulated-selection consistency fix.  The original run walked every shape
# parameter onto its prior wall because the event samples used the 2-D mass
# table while the selection set used the direct evaluation, leaving +125 nats
# of uncancelled interpolation bias in the numerator of the hierarchical
# ratio.  See notes/2026-08-08-tabulated-selection-consistency.md.
#
# Same data (symlinks into runs/endO5_val2), same prior, same seed as
# runs/endO5_evo, so the two posteriors are directly comparable.
# Expect this to be somewhat FASTER than the 59 min original: the selection
# set is now a bilinear lerp instead of the full direct evaluation.

cd /mnt/home/misi/src/GW_pop_cosmo_inference/scripts/

srun /mnt/home/misi/src/GW_pop_cosmo_inference/.venv/bin/python run_inf.py \
    --config run_configs/mock_O5_evo2.ini
