#!/usr/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=0-02
#SBATCH -J diagnose_peakO5

module load cuda/12 cudnn

source /mnt/home/amoran/GW_pop_cosmo_inference/.venv/bin/activate
unset LD_LIBRARY_PATH

cd /mnt/home/amoran/GW_pop_cosmo_inference/scripts/

python diagnose_run.py --nc "/mnt/home/amoran/GW_pop_cosmo_inference/runs/end_05_0821/O5_ne_cosmo.nc" \
  --run "/mnt/home/amoran/GW_pop_cosmo_inference/runs/end_05_0821" \
  --prior "/mnt/home/amoran/GW_pop_cosmo_inference/scripts/priors/O5_noevo_cosmo.prior" \
  --pop_config "/mnt/home/amoran/GW_pop_cosmo_inference/scripts/pop_configs/mock_O5_ne_nb.txt"
