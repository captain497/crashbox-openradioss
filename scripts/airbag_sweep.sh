#!/bin/bash
# airbag_sweep.sh <pulse.csv> <tag> <outroot> : belt-only + cushion at several initial gauge pressures, run locally
P=$1; TAG=$2; ROOT=$3; S=$(dirname "$0")
run() { python3 $S/gen_airbag.py --name $1 --pulse $P --out $ROOT/$1 "${@:2}" >/dev/null && $S/run_local.sh $ROOT/$1 $1 2 >/dev/null \
        && python3 $S/post_airbag.py $ROOT/$1 --case $1 >/dev/null && python3 $S/anim_airbag.py $ROOT/$1 --case $1 --label "$LABEL" && echo "$1 done"; }
LABEL="$TAG, belt only" run AB_${TAG}_BELT --no_bag 1 &
for p0 in 10 20 30 45 60; do LABEL="$TAG, airbag p0 = $p0 kPa" run AB_${TAG}_P$p0 --p0_kPa $p0 & done
wait
