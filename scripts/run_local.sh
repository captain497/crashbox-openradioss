#!/bin/bash
# run_local.sh <dir> <name> [threads] : starter + engine of <dir>/<name>_000{0,1}.rad with the local build
cd "$1" || exit 1
/home/user/or_bin/starter.sh -i "$2_0000.rad" -np 1 > st.log 2>&1
source /home/user/or_bin/env.sh
T0=$(date +%s.%N); $OPENRADIOSS_PATH/exec/engine_linux64_gf -i "$2_0001.rad" -nt ${3:-2} > eng.log 2>&1; echo "$(date +%s.%N) - $T0" | bc > runtime_s.txt
grep -q "NORMAL TERMINATION" "$2_0001.out" && echo "$1 OK" || echo "$1 FAILED"
