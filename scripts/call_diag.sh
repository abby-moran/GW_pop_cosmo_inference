#!/usr/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=0-02
#SBATCH -J full_cosmo_fun

module load cuda/12 cudnn

source /mnt/home/amoran/GW_pop_cosmo_inference/.venv/bin/activate
unset LD_LIBRARY_PATH

cd /mnt/home/amoran/GW_pop_cosmo_inference/scripts/

python diagnose_run.py --nc "/mnt/home/amoran/GW_pop_cosmo_inference/runs/CE_98/CE_full_cosmo_ne.nc" \
  --run "/mnt/home/amoran/GW_pop_cosmo_inference/runs/CE_98" \
  --prior "/mnt/home/amoran/GW_pop_cosmo_inference/runs/priors/CE_full_cosmo_ne.prior" \
  --pop_config "/mnt/home/amoran/GW_pop_cosmo_inference/scripts/pop_configs/CE_peak_q.txt"