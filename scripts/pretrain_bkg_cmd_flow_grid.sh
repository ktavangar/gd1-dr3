#!/bin/bash
#
# Run scripts/pretrain_bkg_cmd_flow.py over a small grid of
# (flow_layers, nn_width, nn_depth) to compare background CMD flow
# architectures.
#
# Usage:
#   bash scripts/pretrain_bkg_cmd_flow_grid.sh
#
# Can be run from anywhere -- it cd's to the repo root itself. Edit the
# *_values arrays below to change the grid. Other pretrain_bkg_cmd_flow.py
# args (data file, epochs, etc.) can be overridden via environment
# variables, e.g.:
#
#   MAX_EPOCHS=50 N_PARALLEL=2 bash scripts/pretrain_bkg_cmd_flow_grid.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${REPO_ROOT}" || exit 1

# Activate the Python virtual environment
source stream-env/bin/activate

# --- Grid definition ---------------------------------------------------
flow_layers_values=(4 8 16 32)
nn_width_values=(32 50 100)
nn_depth_values=(1 2)

# --- Fixed args (override via env vars if needed) -----------------------
DATA_FILE="${DATA_FILE:-/Users/Tavangar/Work/gd1-dr3/data/GD-1_all_data_2026.fits}"
BKG_TABLE_PICKLE="${BKG_TABLE_PICKLE:-/Users/Tavangar/Work/gd1-dr3/data/post_cats_data_flow_version.pkl}"
MAX_EPOCHS="${MAX_EPOCHS:-200}"
MAX_PATIENCE="${MAX_PATIENCE:-10}"
BATCH_SIZE="${BATCH_SIZE:-256}"
SEED="${SEED:-42}"
N_PARALLEL="${N_PARALLEL:-4}"

OUT_DIR="data/flow_tests/grid"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

# --- Build the grid of combinations -------------------------------------
combinations=()
for flow_layers in "${flow_layers_values[@]}"; do
  for nn_width in "${nn_width_values[@]}"; do
    for nn_depth in "${nn_depth_values[@]}"; do
      combinations+=("${flow_layers} ${nn_width} ${nn_depth}")
    done
  done
done

echo "Running ${#combinations[@]} flow configurations" \
     "(${#flow_layers_values[@]} flow_layers x ${#nn_width_values[@]} nn_width x ${#nn_depth_values[@]} nn_depth)," \
     "${N_PARALLEL} at a time..."
echo "Outputs -> ${OUT_DIR}, logs -> ${LOG_DIR}"

# --- Run one grid point ---------------------------------------------------
run_one() {
  local args=($1)
  local flow_layers="${args[0]}"
  local nn_width="${args[1]}"
  local nn_depth="${args[2]}"
  local tag="L${flow_layers}_W${nn_width}_D${nn_depth}"
  local out="${OUT_DIR}/bkg_cmd_flow_${tag}"
  local log_file="${LOG_DIR}/${tag}.log"

  echo "[${tag}] starting -- log: ${log_file}"
  python scripts/pretrain_bkg_cmd_flow.py \
    --data-file "${DATA_FILE}" \
    --bkg-table-pickle "${BKG_TABLE_PICKLE}" \
    --flow-layers "${flow_layers}" \
    --nn-width "${nn_width}" \
    --nn-depth "${nn_depth}" \
    --max-epochs "${MAX_EPOCHS}" \
    --max-patience "${MAX_PATIENCE}" \
    --batch-size "${BATCH_SIZE}" \
    --seed "${SEED}" \
    --out "${out}" \
    > "${log_file}" 2>&1
  status=$?
  if [ ${status} -eq 0 ]; then
    echo "[${tag}] done"
  else
    echo "[${tag}] FAILED (exit ${status}) -- see ${log_file}"
  fi
  return ${status}
}
export -f run_one
export OUT_DIR LOG_DIR DATA_FILE BKG_TABLE_PICKLE MAX_EPOCHS MAX_PATIENCE BATCH_SIZE SEED

# --- Run the grid, in parallel -------------------------------------------
printf "%s\n" "${combinations[@]}" | xargs -n 3 -P "${N_PARALLEL}" -I {} bash -c 'run_one "{}"'

deactivate

# --- Summary, sorted by final validation loss (best first) ---------------
echo
echo "=== Summary (sorted by final val loss; lower is better) ==="
printf "%-20s %s\n" "config" "final_val_loss"
for log in "${LOG_DIR}"/*.log; do
  tag="$(basename "${log}" .log)"
  val_loss="$(grep "final val loss" "${log}" | awk '{print $NF}')"
  echo "${tag} ${val_loss:-FAILED}"
done | sort -k2 -g | awk '{printf "%-20s %s\n", $1, $2}'

echo
echo "All done. Flow weights/metadata/diagnostic plots in ${OUT_DIR}, full logs in ${LOG_DIR}."
