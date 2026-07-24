#!/usr/bin/env bash
# Reproduce the ProCon champion on the full MVTec-AD and VisA benchmarks.
# Champion recipe: p3_drop4_3689 (layer pool {4,5,7,10}, mean fusion, top-mean 0.005,
# B=5 banks, 1% coreset). All 8 metrics are written per category.
set -u
cd "$(dirname "$0")/.."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ad_env

RECIPE="p3_drop4_3689"
ROOT="runs_procon/champion"

for DS in mvtec visa; do
  OUT="${ROOT}/${DS}"
  mkdir -p "$OUT"
  echo "=== ${DS} champion $(date) ==="
  python run_procon.py \
    --dataset "$DS" \
    --recipe "$RECIPE" \
    --output "$OUT"
done

echo "=== done -> ${ROOT}/{mvtec,visa}/results_seed0.json ==="
