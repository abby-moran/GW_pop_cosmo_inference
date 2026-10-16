#!/usr/bin/bash
# Submits gen_inj -> reweight_res -> run_inf as a dependency chain.
# Each stage only starts if the previous one exits 0 (afterok).
# If any stage fails, the rest are automatically cancelled by Slurm.
set -euo pipefail
 
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
 
#jid0=$(sbatch --parsable "${SCRIPT_DIR}/call_SNR_grid.sh")
#echo "submitted gen_inj:     ${jid0}"

jid1=$(sbatch --parsable "${SCRIPT_DIR}/03_run_inf_cosmo_2evo.sh")
echo "submitted Cosmo_2_Evo:     ${jid1}"

jid2=$(sbatch --parsable  "${SCRIPT_DIR}/inf_call.sh")
echo "submitted Cosmo:     ${jid2}"

jid3=$(sbatch --parsable  "${SCRIPT_DIR}/03_run_inf_cosmonb.sh")
echo "submitted Cosmo_no_bump:     ${jid3}"

jid4=$(sbatch --parsable  "${SCRIPT_DIR}/call_inf.sh")
echo "submitted Full_cosmo:     ${jid4}"

#jid2=$(sbatch --parsable "${SCRIPT_DIR}/02a_reweight.sh")
#echo "submitted gen_inj:     ${jid2}"

#jid3=$(sbatch --parsable --dependency=afterok:7227184 "${SCRIPT_DIR}/03a_run_inf_nb.sh")
#echo "submitted gen_inj:     ${jid3}"

#jid4=$(sbatch --parsable "${SCRIPT_DIR}/03a_run_inf_nb.sh")
#echo "submitted gen_inj:     ${jid4}"
 
echo ""
echo "Chain submitted."
echo ""
