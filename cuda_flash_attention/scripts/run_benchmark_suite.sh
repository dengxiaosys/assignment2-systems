#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
PYTHON=${FA_PYTHON:-"$HOME/.venvs/cs336-profile-cu126/bin/python"}
OUTPUT_ROOT=${1:-"$HOME/var/cs336-fa/results/baseline"}

mkdir -p "$OUTPUT_ROOT"

printf 'project_root=%s\n' "$PROJECT_ROOT"
printf 'output_root=%s\n' "$OUTPUT_ROOT"
nvidia-smi \
  --query-gpu=name,driver_version,temperature.gpu,pstate,clocks.sm,clocks.mem,memory.used,memory.free \
  --format=csv,noheader

for mode in noncausal causal; do
    causal_args=()
    if [[ "$mode" == "causal" ]]; then
        causal_args+=(--causal)
    fi

    for sequence_length in 128 256 512 1024 2048; do
        if ((sequence_length <= 512)); then
            warmup=20
            iterations=100
        elif ((sequence_length == 1024)); then
            warmup=10
            iterations=50
        else
            warmup=5
            iterations=20
        fi

        case_name="n${sequence_length}_d64_${mode}"
        printf 'case_started=%s\n' "$case_name"

        "$PROJECT_ROOT/build/fa_benchmark" \
          --nq "$sequence_length" \
          --nk "$sequence_length" \
          --head-dim 64 \
          "${causal_args[@]}" \
          --impl both \
          --warmup "$warmup" \
          --iterations "$iterations" \
          --no-verify \
          | tee "$OUTPUT_ROOT/cpp_${case_name}.log"

        if [[ "${FA_SKIP_PYTORCH:-0}" != "1" ]]; then
            PYTHONDONTWRITEBYTECODE=1 \
            PYTHONWARNINGS="ignore::UserWarning" \
            "$PYTHON" "$PROJECT_ROOT/app/pytorch_benchmark.py" \
              --nq "$sequence_length" \
              --nk "$sequence_length" \
              --head-dim 64 \
              "${causal_args[@]}" \
              --warmup "$warmup" \
              --iterations "$iterations" \
              --output-json "$OUTPUT_ROOT/pytorch_${case_name}.json"
        fi

        printf 'case_completed=%s\n' "$case_name"
    done
done

nvidia-smi \
  --query-gpu=name,driver_version,temperature.gpu,pstate,clocks.sm,clocks.mem,memory.used,memory.free \
  --format=csv,noheader
printf 'suite_status=passed\n'
