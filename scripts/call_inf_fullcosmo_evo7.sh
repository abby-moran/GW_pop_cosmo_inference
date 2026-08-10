#!/usr/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=h100_pcie:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1-00
#SBATCH -J fullcosmo_evo7
#SBATCH -o /mnt/home/misi/src/GW_pop_cosmo_inference/scripts/fullcosmo_evo7_%j.log

# Fullcosmo inference on the msigma_low=1.5 mock.  Chain after reweight:
#   sbatch --dependency=afterok:<rwt_jobid> call_inf_fullcosmo_evo7.sh
# Pivot + log_fpeak prior, dense_mass, depth 10; evt_end=8000 (selection
# budget at width 1.5).  Distinct from evo4/5/6.

cd /mnt/home/misi/src/GW_pop_cosmo_inference/scripts/

srun /mnt/home/misi/src/GW_pop_cosmo_inference/.venv/bin/python run_inf.py \
    --config run_configs/mock_O5_fullcosmo_evo7.ini
