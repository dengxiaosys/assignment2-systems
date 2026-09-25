#!/usr/bin/env bash

set -euo pipefail

cd "$HOME/work/cs336-profile/assignments/assignment2-systems"

PYTHONPATH="$HOME/work/cs336-profile/assignments/assignment1-basics:$PWD" \
CUDA_VISIBLE_DEVICES=0 \
"$HOME/.venvs/cs336-profile-cu126/bin/python" benchmark.py \
  --model-size small --mode full --device cuda --dtype float32 \
  --batch-size 2 --context-length 512 \
  --warmup-steps 5 --measurement-steps 10 --seed 0 \
  --output-json "$HOME/var/cs336-profile/results/small_b2_s512_full.json"
