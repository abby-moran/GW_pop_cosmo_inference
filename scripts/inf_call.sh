#!/usr/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --time=0-02

module load cuda/12 cudnn

source /mnt/home/amoran/GW_pop_cosmo_inference/.venv/bin/activate
unset LD_LIBRARY_PATH

cd /mnt/home/amoran/GW_pop_cosmo_inference/scripts/

#srun python run_inf.py --config /mnt/home/amoran/GW_pop_cosmo_inference/scripts/run_configs/mock_O5_noevo_cosmo_nb.ini
# quick sanity check with fake data, no files needed
srun python profile_mod.py --synthetic --nobs 9000 --nsamp 4000 --ndraw 500000

# real run, against your actual config/data
srun python profile_mod.py --config run_configs/mock_O5_noevo.ini

# plus a full XLA-op-level trace you can open in chrome://tracing or perfetto
surn python profile_mod.py --config run_configs/mock_O5_noevo.ini --trace