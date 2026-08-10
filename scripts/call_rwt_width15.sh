#!/usr/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0-03
#SBATCH -J rwt_width15
#SBATCH -o /mnt/home/misi/src/GW_pop_cosmo_inference/scripts/rwt_width15_%j.log

# Reweight the enlarged endO5_w15pool injection file to the msigma_low=1.5
# truth.  Chain after gen_inj:
#   sbatch --dependency=afterok:<gen_jobid> call_rwt_width15.sh
# Retargets endO5_fullcosmo_evo7's inj symlink at the new pool and clears
# any partial obs/sel/PE from a cancelled earlier attempt.  Does not
# touch endO5_val2 or in-flight evo4/5/6.

set -euo pipefail

POOL_INJ="${HOME}/ceph/GW_pop_cosmo_inference/endO5_w15pool/inj_mockGW5_SNR0.h5"
EVO7="/mnt/home/misi/src/GW_pop_cosmo_inference/runs/endO5_fullcosmo_evo7"

if [[ ! -f "$POOL_INJ" ]]; then
    echo "ERROR: enlarged pool not found at $POOL_INJ" >&2
    exit 1
fi

mkdir -p "$EVO7"
rm -f "$EVO7/obs_width15.h5" "$EVO7/sel_width15.h5" "$EVO7/PE_width15.h5"
ln -sfn "$POOL_INJ" "$EVO7/inj_mockGW5_SNR0.h5"
ls -la "$EVO7"

cd /mnt/home/misi/src/GW_pop_cosmo_inference/scripts/

srun /mnt/home/misi/src/GW_pop_cosmo_inference/.venv/bin/python reweight_res.py \
    --config run_configs/mock_O5_fullcosmo_evo7.ini
