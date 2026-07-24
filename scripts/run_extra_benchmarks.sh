#!/usr/bin/env bash
# Champion ProCon on Uni-Medical (BMAD, pixel-mask subsets) and BTAD.
# Runs categories strictly sequentially to avoid GPU contention.
# Usage: scripts/run_extra_benchmarks.sh
set -uo pipefail

cd "$(dirname "$0")/.."
CONDA_BASE="$(conda info --base)"
PY="$CONDA_BASE/envs/ad_env/bin/python"
RECIPE="p3_drop4_3689"

run_one() {
    local ds="$1" cat="$2"
    local out="runs_procon/extra_benchmarks_b1/${ds}/${cat}"
    if [[ -f "${out}/results_seed0.json" ]]; then
        echo "== skip ${ds}/${cat} (exists) =="
        return
    fi
    echo "== ${ds}/${cat} =="
    PYTHONPATH="$PWD" "$PY" run_procon.py \
        --dataset "$ds" --recipe "$RECIPE" --num_banks 1 --topmean_ratio 0.005 \
        --category "$cat" --output "$out"
}

# Uni-Medical: three subsets that ship pixel masks.
for c in brain liver retina_resc; do run_one uni_medical "$c"; done
# BTAD: three product categories.
for c in 01 02 03; do run_one btad "$c"; done

echo "== extra benchmarks done -> runs_procon/extra_benchmarks_b1/ =="
