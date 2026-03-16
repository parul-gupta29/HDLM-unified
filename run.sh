#!/usr/bin/env bash
# Run commands for HDLM experiments.
# All paths and hyperparameters live in configs/<dataset>.yaml.
# CLI flags below override anything in the config.
# Run all commands from the project root:  bash run.sh
# Run a single section by sourcing just those lines.

set -e   # exit on first error
CONFIG="configs/gsm8k.yaml"

# ═══════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════

# ─────────────────────────────────────────
# Option A: Unified model (recommended)
#   Single backbone, joint LM + hierarchy loss.
#   Saves to: checkpoints/gsm8k/unified_lora/
# ─────────────────────────────────────────
python scripts/train_unified.py --config $CONFIG

# ─────────────────────────────────────────
# Option B: Two-stage (legacy)
#   Stage 1 — hierarchy predictor
#   Stage 2 — generator LoRA
# ─────────────────────────────────────────
# python scripts/train_hierarchy.py  --config $CONFIG
# python scripts/train_generator.py  --config $CONFIG


# ═══════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════

# ─────────────────────────────────────────
# 1. Unified model — greedy (1 sample)
# ─────────────────────────────────────────
python scripts/eval_unified.py \
    --config $CONFIG \
    --output results/unified_results.json

# ─────────────────────────────────────────
# 2. Unified model — best-of-4
# ─────────────────────────────────────────
python scripts/eval_unified.py \
    --config      $CONFIG \
    --num_samples 4 \
    --output      results/unified_results_4sample.json

# ─────────────────────────────────────────
# 3. Baseline — LLaDA-Instruct, no fine-tuning
# ─────────────────────────────────────────
python scripts/eval_baseline.py \
    --config $CONFIG \
    --output results/baseline_base.json

# ─────────────────────────────────────────
# 4. Baseline — LLaDA-Instruct, no fine-tuning (best-of-4)
# ─────────────────────────────────────────
python scripts/eval_baseline.py \
    --config      $CONFIG \
    --num_samples 4 \
    --output      results/baseline_base_4sample.json
