#!/usr/bin/env bash

set -euo pipefail

cd "$HOME/work/cs336-profile/assignments/assignment2-systems"

PROFILE_DIR="$HOME/var/cs336-profile/profiles/small_b2_s512"
REPORT_BASE="$PROFILE_DIR/profile"
mkdir -p "$PROFILE_DIR"
rm -f "$REPORT_BASE.nsys-rep" "$REPORT_BASE.sqlite" "$PROFILE_DIR/result.json"

PYTHONPATH="$HOME/work/cs336-profile/assignments/assignment1-basics:$PWD" \
CUDA_VISIBLE_DEVICES=0 \
PYTHONDONTWRITEBYTECODE=1 \
nsys profile \
  --force-overwrite=true \
  --output="$REPORT_BASE" \
  -- "$HOME/.venvs/cs336-profile-cu126/bin/python" benchmark.py \
    --model-size small --mode full --device cuda --dtype float32 \
    --batch-size 2 --context-length 512 \
    --warmup-steps 5 --measurement-steps 1 --seed 0 \
    --output-json "$PROFILE_DIR/result.json"

printf 'profile_report=%s\n' "$REPORT_BASE.nsys-rep"
