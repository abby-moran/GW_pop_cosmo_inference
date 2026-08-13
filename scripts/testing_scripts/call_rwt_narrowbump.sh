#!/usr/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0-03
#SBATCH -J rwt_nbump
#SBATCH -o /mnt/home/misi/src/GW_pop_cosmo_inference/scripts/rwt_narrowbump_%j.log

# Step 1 of 2: reweight the endO5_val2 injection pool to the narrow-bump truth
# population (msigma_low = 1 instead of 4).  Produces obs/PE/selection files in
# runs/endO5_narrowbump/.  The injection pool itself is population independent
# and is symlinked from ceph, so gen_inj.py does NOT need rerunning.
#
# val2's equivalent step took ~25 min; 3 h is a generous cap.
# Writes ~1.4 GB into runs/endO5_narrowbump/ (home filesystem, quota'd).
#
# Does not touch runs/endO5_fullcosmo_evo4/ or any other in-flight run.

cd /mnt/home/misi/src/GW_pop_cosmo_inference/scripts/

srun /mnt/home/misi/src/GW_pop_cosmo_inference/.venv/bin/python reweight_res.py \
    --config run_configs/mock_O5_narrowbump.ini
