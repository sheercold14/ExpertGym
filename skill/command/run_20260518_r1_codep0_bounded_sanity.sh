#!/usr/bin/env bash
set -euo pipefail

cd /mnt/cache/wuruixiao/users/lsc/Agent/ExpertGym

export RUN_NAME="${RUN_NAME:-r1_codep0_layer28_z001_bounded_codeonly_sanity_20260518}"
export NUM_ITERS="${NUM_ITERS:-4}"
export COEFF_BOUND_BY_EXPERT="${COEFF_BOUND_BY_EXPERT:-reasoning=0.0:0.003}"
export MAX_COEFF_DELTA_BY_EXPERT="${MAX_COEFF_DELTA_BY_EXPERT:-reasoning=0.002}"

bash skill/command/run_20260518_r1_codep0_sanity.sh
