#!/usr/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0-02
#SBATCH -J gen_w15pool
#SBATCH -o /mnt/home/misi/src/GW_pop_cosmo_inference/scripts/gen_w15pool_%j.log

# Build a ~1.7x larger clone of the endO5_val2 injection proposal
# (num_loops=70) on ceph under endO5_w15pool/.  val2's 40-loop run took
# ~17 min; 70 loops ~30 min.  Does not touch endO5_val2.

cd /mnt/home/misi/src/GW_pop_cosmo_inference/scripts/

srun /mnt/home/misi/src/GW_pop_cosmo_inference/.venv/bin/python gen_inj.py \
    --config run_configs/mock_O5_w15pool.ini
