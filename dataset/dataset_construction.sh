#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

: "${BLENDED_DIR:?Set BLENDED_DIR}"
: "${LAION_DIR:?Set LAION_DIR}"
: "${CAPTION_DIR:?Set CAPTION_DIR}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR}"

CAPTION_META="${CAPTION_META:-${CAPTION_DIR}/captions.jsonl}"
NUM_SAMPLES="${NUM_SAMPLES:-500000}"
START_INDEX="${START_INDEX:-0}"
SEED="${SEED:-42}"
MAX_BASE_SAMPLES="${MAX_BASE_SAMPLES:-18000}"
MIN_DONOR_SAMPLES="${MIN_DONOR_SAMPLES:-1}"
MAX_DONOR_SAMPLES="${MAX_DONOR_SAMPLES:-4}"
MIN_LAYERS_PER_DONOR="${MIN_LAYERS_PER_DONOR:-0}"
MAX_LAYERS_PER_DONOR="${MAX_LAYERS_PER_DONOR:-2}"
MIN_LAYERS_TO_REMOVE="${MIN_LAYERS_TO_REMOVE:-1}"
MAX_LAYERS_TO_REMOVE="${MAX_LAYERS_TO_REMOVE:-4}"
ADDED_LAYER_MIN_SIZE="${ADDED_LAYER_MIN_SIZE:-0.9}"
ADDED_LAYER_MAX_SIZE="${ADDED_LAYER_MAX_SIZE:-1.1}"
LAION_PROB="${LAION_PROB:-0.60}"
CAPTION_PROB="${CAPTION_PROB:-0.35}"
LAION_MIN_SIZE="${LAION_MIN_SIZE:-0.3}"
LAION_MAX_SIZE="${LAION_MAX_SIZE:-0.4}"
CAPTION_MIN_SIZE="${CAPTION_MIN_SIZE:-0.6}"
CAPTION_MAX_SIZE="${CAPTION_MAX_SIZE:-0.8}"
ALPHAVAE_MIN_LAYERS="${ALPHAVAE_MIN_LAYERS:-0}"
ALPHAVAE_MAX_LAYERS="${ALPHAVAE_MAX_LAYERS:-0}"
ALPHAVAE_MIN_SIZE="${ALPHAVAE_MIN_SIZE:-0.25}"
ALPHAVAE_MAX_SIZE="${ALPHAVAE_MAX_SIZE:-0.40}"
NUM_WORKERS="${NUM_WORKERS:-64}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"

cmd=(
  "$PYTHON_BIN" "$SCRIPT_DIR/scaleup_dataset.py"
  --blended_dir "$BLENDED_DIR"
  --laion_dir "$LAION_DIR"
  --caption_dir "$CAPTION_DIR"
  --caption_meta "$CAPTION_META"
  --output_dir "$OUTPUT_DIR"
  --num_samples "$NUM_SAMPLES"
  --start_index "$START_INDEX"
  --seed "$SEED"
  --min_donor_samples "$MIN_DONOR_SAMPLES"
  --max_donor_samples "$MAX_DONOR_SAMPLES"
  --min_layers_per_donor "$MIN_LAYERS_PER_DONOR"
  --max_layers_per_donor "$MAX_LAYERS_PER_DONOR"
  --added_layer_min_size "$ADDED_LAYER_MIN_SIZE"
  --added_layer_max_size "$ADDED_LAYER_MAX_SIZE"
  --laion_prob "$LAION_PROB"
  --caption_prob "$CAPTION_PROB"
  --laion_min_size "$LAION_MIN_SIZE"
  --laion_max_size "$LAION_MAX_SIZE"
  --caption_min_size "$CAPTION_MIN_SIZE"
  --caption_max_size "$CAPTION_MAX_SIZE"
  --min_layers_to_remove "$MIN_LAYERS_TO_REMOVE"
  --max_layers_to_remove "$MAX_LAYERS_TO_REMOVE"
  --alphavae_min_layers "$ALPHAVAE_MIN_LAYERS"
  --alphavae_max_layers "$ALPHAVAE_MAX_LAYERS"
  --alphavae_min_size "$ALPHAVAE_MIN_SIZE"
  --alphavae_max_size "$ALPHAVAE_MAX_SIZE"
  --max_base_samples "$MAX_BASE_SAMPLES"
  --num_workers "$NUM_WORKERS"
)

if [[ -n "${ALPHAVAE_DIR:-}" ]]; then
  cmd+=(--alphavae_dir "$ALPHAVAE_DIR")
fi

if [[ -n "${ALPHAVAE_PROMPTS:-}" ]]; then
  cmd+=(--alphavae_prompts "$ALPHAVAE_PROMPTS")
fi

if [[ "$SKIP_EXISTING" == "1" ]]; then
  cmd+=(--skip_existing)
fi

"${cmd[@]}"
