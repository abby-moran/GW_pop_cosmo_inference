#!/usr/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=0-02
#SBATCH -J max_run

module load cuda/12 cudnn

source /mnt/home/amoran/GW_pop_cosmo_inference/.venv/bin/activate
unset LD_LIBRARY_PATH

cd /mnt/home/amoran/GW_pop_cosmo_inference/scripts/

python diagnose_run.py --nc "/mnt/home/amoran/GW_pop_cosmo_inference/runs/endO5_fullcosmo_evo8/max_cosmo7_halfsel.nc" \
  --run "/mnt/home/amoran/GW_pop_cosmo_inference/runs/endO5_fullcosmo_evo8" \
  --prior "/mnt/home/amoran/GW_pop_cosmo_inference/runs/priors_old/gwtc5_fullcosmo_evo_pivot_fpeak_w15.prior" \
  --pop_config "/mnt/home/amoran/GW_pop_cosmo_inference/scripts/pop_configs/mock_O5_width15.txt"