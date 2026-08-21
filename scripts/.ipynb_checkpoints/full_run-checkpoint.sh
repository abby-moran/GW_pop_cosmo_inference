#!/usr/bin/bash
# Submits gen_inj -> reweight_res -> run_inf as a dependency chain.
# Each stage only starts if the previous one exits 0 (afterok).
# If any stage fails, the rest are automatically cancelled by Slurm.
set -euo pipefail
 
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p /mnt/home/amoran/GW_pop_cosmo_inference/scripts/logs
 
jid1=$(sbatch --parsable "${SCRIPT_DIR}/01_gen_inj.sh")
echo "submitted gen_inj:     ${jid1}"
 
jid2=$(sbatch --parsable --dependency=afterok:${jid1} "${SCRIPT_DIR}/02_reweight.sh")
echo "submitted reweight:    ${jid2}  (after ${jid1})"
 
jid3=$(sbatch --parsable --dependency=afterok:${jid2} "${SCRIPT_DIR}/03_run_inf.sh")
echo "submitted run_inf:     ${jid3}  (after ${jid2})"

jid3=$(sbatch --parsable --dependency=afterok:${jid2} "${SCRIPT_DIR}/03a_run_inf_alt.sh")
echo "submitted run_inf_alt:     ${jid3}  (after ${jid2})"
 
echo ""
echo "Chain submitted. Track with:"
echo "  squeue -u \$USER -j ${jid1},${jid2},${jid3}"
echo ""
echo "Logs will land in /mnt/home/amoran/GW_pop_cosmo_inference/scripts/logs/"