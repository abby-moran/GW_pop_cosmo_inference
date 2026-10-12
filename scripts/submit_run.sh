#!/bin/bash
# Submit an inference job and chain its postprocessing (plots) job on it.
#
#   ./submit_run.sh call_inf_<name>.sh [call_plots_<name>.sh]
#
# The plots script defaults to the call_inf_ -> call_plots_ naming convention.
# Unsets the (expired) SBATCH/SLURM reservation variables from the login env.
set -euo pipefail

inf_script=${1:?usage: $0 call_inf_<name>.sh [call_plots_<name>.sh]}
plots_script=${2:-${inf_script/call_inf_/call_plots_}}

if [[ ! -f $inf_script ]]; then
    echo "error: $inf_script not found" >&2
    exit 1
fi

jobid=$(env -u SBATCH_RESERVATION -u SLURM_RESERVATION sbatch --parsable "$inf_script")
echo "inference: $jobid ($inf_script)"

if [[ -f $plots_script ]]; then
    plotid=$(env -u SBATCH_RESERVATION -u SLURM_RESERVATION sbatch --parsable \
        --dependency=afterok:"$jobid" "$plots_script")
    echo "plots:     $plotid (afterok:$jobid, $plots_script)"
else
    echo "no plots script ($plots_script not found); submitted inference only"
fi
