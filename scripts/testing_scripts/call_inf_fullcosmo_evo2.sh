#!/usr/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=h100_pcie:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0-16
#SBATCH -J fullcosmo_evo2
#SBATCH -o /mnt/home/misi/src/GW_pop_cosmo_inference/scripts/fullcosmo_evo2_%j.log

# Rerun of endO5_fullcosmo_evo (h, Omh2, w AND mpisndot free) after the
# tabulated-selection consistency fix -- see
# notes/2026-08-08-tabulated-selection-consistency.md.  The original hit the
# same mpisndot floor mode as endO5_evo, plus h -> 0.41 (its floor).
#
# Same data, prior and seed as runs/endO5_fullcosmo_evo, so the posteriors are
# directly comparable.

cd /mnt/home/misi/src/GW_pop_cosmo_inference/scripts/

srun /mnt/home/misi/src/GW_pop_cosmo_inference/.venv/bin/python run_inf.py \
    --config run_configs/mock_O5_fullcosmo_evo2.ini
