#!/usr/bin/env bash
# Real-IAD (single-view) champion {-3,-6,-8,-9} at the 1% coreset budget,
# all 30 categories. Per-category isolated processes (avoids the cross-category
# memory accumulation that crashed the large-coreset MVTec/VisA run). Already-
# completed categories are skipped, so the script is resumable.
#
# 1% only: Real-IAD has only ~1.2k train images per category, so a 1% coreset is
# already rich; 5% would 2-3x the runtime for no expected gain on this much
# larger benchmark (~5k images/category x 30 categories).
set -u
cd "$(dirname "$0")/.."
CONDA_BASE="${CONDA_BASE:-$(conda info --base 2>/dev/null)}"
[ -z "${CONDA_BASE:-}" ] && CONDA_BASE="$HOME/miniconda3"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate ad_env
PROJ="$(pwd)"
# Absolute env python: each category is launched inside its own transient
# systemd-user scope (see below), where the interactive conda shell hooks are
# not available, so we invoke the environment interpreter directly.
PY="$CONDA_BASE/envs/ad_env/bin/python"

ROOT="runs_consensuscore/realiad"
LOG="runs_consensuscore/realiad_run.log"
mkdir -p "$ROOT"
RECIPE="p3_drop4_3689"   # champion {-3,-6,-8,-9}, 1% budget
SUB="r01"
CATS="audiojack bottle_cap button_battery end_cap eraser fire_hood mint mounts pcb phone_battery plastic_nut plastic_plug porcelain_doll regulator rolled_strip_base sim_card_set switch tape terminalblock toothbrush toy toy_brick transistor1 u_block usb usb_adaptor vcpill wooden_beads woodstick zipper"

echo "=== Real-IAD champion 1% started $(date) ===" | tee "$LOG"
for CAT in $CATS; do
  OUT="${ROOT}/${SUB}/${CAT}"
  [ -f "${OUT}/results_seed0.json" ] && { echo "skip $CAT (done)"; continue; }
  mkdir -p "$OUT"
  # Run each category in its OWN transient systemd-user scope, OUTSIDE the
  # VS Code snap.code cgroup scope. The metric stage of large Real-IAD
  # categories briefly forces page-cache reclaim; when run inside the editor's
  # cgroup that memory-pressure (PSI) signal makes systemd-oomd kill the whole
  # editor scope (and our job with it) even though >70 GB RAM is free. An
  # isolated scope insulates the editor and lets the job run to completion.
  UNIT="realiad_${SUB}_${CAT}"
  systemctl --user reset-failed "$UNIT" 2>/dev/null || true
  systemd-run --user --unit="$UNIT" --wait --collect \
    --working-directory="$PROJ" \
    bash -c "$PY -u run_consensuscore.py --dataset realiad --category '$CAT' \
      --recipe '$RECIPE' --bank_vectorized --output '$OUT'" >>"$LOG" 2>&1
  echo ">>> ${CAT} done $(date)" | tee -a "$LOG"
done
echo "=== Real-IAD champion 1% done $(date) ===" | tee -a "$LOG"

