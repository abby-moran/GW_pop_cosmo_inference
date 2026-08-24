#!/usr/bin/bash
# Submits gen_inj -> reweight_res -> run_inf as a dependency chain.
# Each stage only starts if the previous one exits 0 (afterok).
# If any stage fails, the rest are automatically cancelled by Slurm.
set -euo pipefail
 
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p /mnt/home/amoran/GW_pop_cosmo_inference/scripts/logs
 
jid1=$(sbatch --parsable "${SCRIPT_DIR}/03_run_inf.sh")
echo "submitted gen_inj:     ${jid1}"

jid2=$(sbatch --parsable "${SCRIPT_DIR}/03a_run_inf_nb.sh")
echo "submitted gen_inj:     ${jid2}"
 
echo ""
echo "Chain submitted. Track with:"
echo "  squeue -u \$USER -j ${jid1},${jid2}"
echo ""
echo "Logs will land in /mnt/home/amoran/GW_pop_cosmo_inference/scripts/logs/"