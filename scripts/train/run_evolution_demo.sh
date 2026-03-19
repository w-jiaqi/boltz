#!/bin/bash
# End-to-end demo: cache features for 2 complexes, then train evolution head.
#
# Usage (inside apptainer with boltz installed):
#   bash scripts/train/run_evolution_demo.sh \
#       --checkpoint /boltz_cache/boltz2_conf.ckpt \
#       --cache /boltz_cache \
#       --workdir /output/evolution_demo
#
# Or with environment variables:
#   CHECKPOINT=/boltz_cache/boltz2_conf.ckpt \
#   BOLTZ_CACHE=/boltz_cache \
#   WORKDIR=/output/evolution_demo \
#   bash scripts/train/run_evolution_demo.sh

set -e

# Parse args or use env vars
CHECKPOINT="${CHECKPOINT:-}"
BOLTZ_CACHE_DIR="${BOLTZ_CACHE:-}"
WORKDIR="${WORKDIR:-./evolution_demo}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint) CHECKPOINT="$2"; shift 2;;
        --cache) BOLTZ_CACHE_DIR="$2"; shift 2;;
        --workdir) WORKDIR="$2"; shift 2;;
        *) echo "Unknown option: $1"; exit 1;;
    esac
done

if [ -z "$CHECKPOINT" ] || [ -z "$BOLTZ_CACHE_DIR" ]; then
    echo "Usage: bash run_evolution_demo.sh --checkpoint /path/to/boltz2_conf.ckpt --cache /path/to/boltz_cache [--workdir /path/to/workdir]"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "Evolution Head Training Demo"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT"
echo "Cache:      $BOLTZ_CACHE_DIR"
echo "Workdir:    $WORKDIR"
echo ""

# ---- Step 1: Create dummy input structures ----
echo "[Step 1/4] Creating dummy input YAML files..."

STRUCTURES_DIR="$WORKDIR/structures"
mkdir -p "$STRUCTURES_DIR"

cat > "$STRUCTURES_DIR/complex_human.yaml" << 'EOF'
version: 1
sequences:
  - protein:
      id: A
      sequence: MAHHHHHHVAVDAVSFTLLQDQLQSVLDTLSEREAGVVRLRFGLTDGQPRTLDEIGQVYGVTRERIRQIESKTMSKLRHPSRSQVLRDYLDGSSGSGTPEERLLRAIFGEKA
  - protein:
      id: B
      sequence: MRYAFAAEATTCNAFWRNVDMTVTALYEVPLGVCTQDPDRWTTTPDDEAKTLCRACPRRWLCARDAVESAGAEGLWAGVVIPESGRARAFALGQLRSLAERNGYPVRDHRVSAQSA
EOF

cat > "$STRUCTURES_DIR/complex_mouse.yaml" << 'EOF'
version: 1
sequences:
  - protein:
      id: A
      sequence: MAHHHHHHVAVDAVSFTLLQDQLQSVLDTLSEREAGVVRLRFGLTDGQPRTLDEIGQVYGVTRERIRQIESKTMSKLRHPSRSQVLRDYLDGSSGSGTPEERLLRAIFGEKA
  - protein:
      id: B
      sequence: MRYAFAAEATTCNAFWRNVDMTVTALYEVPLGVCTQDPDRWTTTPDDEAKTLCRACPRRWLCARDAVESAGAEGLWAGVVIPESGRARAFALGQLRSLAERNGYPVREHHVSAQSA
EOF

echo "  Created: complex_human.yaml, complex_mouse.yaml"

# ---- Step 2: Cache trunk representations ----
echo ""
echo "[Step 2/4] Caching trunk representations (this runs Boltz2 inference)..."

CACHE_DIR="$WORKDIR/cache"
python "$SCRIPT_DIR/cache_evolution_features.py" \
    --data "$STRUCTURES_DIR" \
    --output "$CACHE_DIR" \
    --checkpoint "$CHECKPOINT" \
    --cache "$BOLTZ_CACHE_DIR" \
    --use_msa_server \
    --no_kernels \
    --recycling_steps 1 \
    --sampling_steps 20 \
    --diffusion_samples 1

echo ""
echo "  Cached files:"
ls -lh "$CACHE_DIR"/*.pt 2>/dev/null || echo "  WARNING: No .pt files found!"

# ---- Step 3: Create pairs CSV ----
echo ""
echo "[Step 3/4] Creating training pairs CSV..."

PAIRS_CSV="$WORKDIR/train_pairs.csv"
cat > "$PAIRS_CSV" << EOF
preferred,dispreferred,dist_preferred,dist_dispreferred
complex_human,complex_mouse,0.0,1.0
complex_mouse,complex_human,1.0,0.0
EOF

echo "  Created: $PAIRS_CSV"

# ---- Step 4: Train evolution head ----
echo ""
echo "[Step 4/4] Training evolution head (3 steps, debug mode)..."

TRAIN_CONFIG="$WORKDIR/train_config.yaml"
cat > "$TRAIN_CONFIG" << EOF
cache_dir: $CACHE_DIR
train_pairs_csv: $PAIRS_CSV
val_pairs_csv: null
output: $WORKDIR/training_output

evolution_model_args:
  token_s: 384
  token_z: 128
  pairformer_args:
    num_blocks: 2
    dropout: 0.0
    pairwise_head_width: 32
    pairwise_num_heads: 4
    activation_checkpointing: false
  head_hidden_dim: 128
  num_dist_bins: 64
  max_dist: 22
  use_interface_mask: false

training:
  lr: 1.0e-3
  adam_beta_1: 0.9
  adam_beta_2: 0.95
  adam_eps: 1.0e-8
  weight_decay: 0.0
  bt_weight: 1.0
  margin_weight: 0.5
  bt_temperature: 1.0
  margin_alpha: 1.0

trainer:
  accelerator: cpu
  devices: 1
  precision: 32
  max_steps: 3

num_workers: 0
save_top_k: 1
debug: true
EOF

python "$SCRIPT_DIR/train_evolution.py" "$TRAIN_CONFIG"

echo ""
echo "=========================================="
echo "Demo complete!"
echo "=========================================="
echo ""
echo "Cached features: $CACHE_DIR/"
echo "Training output: $WORKDIR/training_output/"
echo ""
echo "To train for real, edit $TRAIN_CONFIG:"
echo "  - Set trainer.accelerator: gpu"
echo "  - Set trainer.max_epochs: 100"
echo "  - Set trainer.accumulate_grad_batches: 16"
echo "  - Add more complexes and pairs"
echo "  - Optionally enable wandb logging"
