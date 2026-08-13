#!/usr/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=0-02
#SBATCH -J diagnose_peakO5

module load cuda/12 cudnn

source /mnt/home/amoran/GW_pop_cosmo_inference/.venv/bin/activate
unset LD_LIBRARY_PATH

cd /mnt/home/amoran/GW_pop_cosmo_inference/scripts/

python diagnose_run.py --nc "../runs/end_05_0811/O5_ne_gc.nc" \
  --run "end_05_0811" \
  --prior "priors/O5_noevo.prior" \
  --pop_config "pop_configs/mock_O5_nb_ne.txt"