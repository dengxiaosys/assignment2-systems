#!/usr/bin/env bash

set -Eeuo pipefail
IFS=$'\n\t'

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
ASSIGNMENT2_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd -P)
ASSIGNMENTS_ROOT=$(cd -- "$ASSIGNMENT2_ROOT/.." && pwd -P)
ASSIGNMENT1_ROOT="$ASSIGNMENTS_ROOT/assignment1-basics"

REMOTE="${CS336_SYNC_REMOTE:-cuda-via-a}"
REMOTE_ROOT="${CS336_SYNC_REMOTE_ROOT:-/home/dengxiao/work/cs336-profile/assignments}"
MODE="dry-run"
ASSUME_YES=false
TEMPORARY_ROOT=""

SSH_OPTIONS=(
  -o BatchMode=yes
  -o ConnectTimeout=10
)

RSYNC_OPTIONS=(
  --archive
  --no-owner
  --no-group
  --compress
  --checksum
  --delay-updates
  --delete
  --delete-delay
  --delete-excluded
  --omit-dir-times
  --prune-empty-dirs
  --safe-links
  --itemize-changes
  --human-readable
  --out-format=%i\ %n%L
  --timeout=30
  -e "ssh -o BatchMode=yes -o ConnectTimeout=10"
)

ASSIGNMENT1_FILTERS=(
  --exclude='**/__pycache__/***'
  --exclude='**/*.pyc'
  --include='/cs336_basics/***'
  --exclude='*'
)

ASSIGNMENT2_FILTERS=(
  --exclude='**/__pycache__/***'
  --exclude='**/*.pyc'
  --include='/cs336_systems/***'
  --include='/benchmark.py'
  --include='/remote_profile/'
  --include='/remote_profile/run_small_benchmark.sh'
  --exclude='*'
)

usage() {
  cat <<'EOF'
Usage:
  ./remote_sync/sync_to_cuda.sh [--dry-run]
  ./remote_sync/sync_to_cuda.sh --apply [--yes]

Options:
  --dry-run           Preview changes without modifying the remote host (default).
  --apply             Preview, confirm, apply, then verify a second dry-run is empty.
  --yes               Skip the confirmation prompt; only valid with --apply.
  --remote HOST       SSH host alias (default: cuda-via-a).
  --remote-root PATH  Remote assignments directory. It must end in
                      /work/cs336-profile/assignments.
  -h, --help          Show this help.

Environment overrides:
  CS336_SYNC_REMOTE
  CS336_SYNC_REMOTE_ROOT

Only these files are synchronized:
  assignment1-basics/cs336_basics/**
  assignment2-systems/cs336_systems/**
  assignment2-systems/benchmark.py
  assignment2-systems/remote_profile/run_small_benchmark.sh

The script never installs dependencies or runs profiling commands.
EOF
}

log() {
  printf '%s\n' "$*"
}

die() {
  printf 'error=%s\n' "$*" >&2
  exit 1
}

cleanup() {
  if [[ -n "$TEMPORARY_ROOT" && -d "$TEMPORARY_ROOT" ]]; then
    rm -rf -- "$TEMPORARY_ROOT"
  fi
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

parse_args() {
  while (($# > 0)); do
    case "$1" in
      --dry-run)
        MODE="dry-run"
        shift
        ;;
      --apply)
        MODE="apply"
        shift
        ;;
      --yes)
        ASSUME_YES=true
        shift
        ;;
      --remote)
        (($# >= 2)) || die "--remote requires a value"
        REMOTE=$2
        shift 2
        ;;
      --remote-root)
        (($# >= 2)) || die "--remote-root requires a value"
        REMOTE_ROOT=${2%/}
        shift 2
        ;;
      -h | --help)
        usage
        exit 0
        ;;
      *)
        die "unknown argument: $1"
        ;;
    esac
  done

  if [[ "$ASSUME_YES" == true && "$MODE" != "apply" ]]; then
    die "--yes is only valid with --apply"
  fi

  REMOTE_ROOT=${REMOTE_ROOT%/}
}

validate_configuration() {
  require_command flock
  require_command rsync
  require_command ssh

  [[ -d "$ASSIGNMENT1_ROOT/cs336_basics" ]] || die "missing source directory: $ASSIGNMENT1_ROOT/cs336_basics"
  [[ -d "$ASSIGNMENT2_ROOT/cs336_systems" ]] || die "missing source directory: $ASSIGNMENT2_ROOT/cs336_systems"
  [[ -f "$ASSIGNMENT2_ROOT/benchmark.py" ]] || die "missing source file: $ASSIGNMENT2_ROOT/benchmark.py"

  [[ "$REMOTE" =~ ^[A-Za-z0-9._@-]+$ ]] || die "unsafe remote host value: $REMOTE"
  [[ "$REMOTE_ROOT" =~ ^/[A-Za-z0-9._/-]+$ ]] || die "remote root must be a simple absolute path"
  [[ "$REMOTE_ROOT" == */work/cs336-profile/assignments ]] || {
    die "remote root must end in /work/cs336-profile/assignments"
  }
}

acquire_lock() {
  local lock_file="${XDG_RUNTIME_DIR:-/tmp}/cs336-profile-sync-${UID}.lock"
  exec 9>"$lock_file"
  flock -n 9 || die "another sync process is already running"
}

check_remote() {
  ssh "${SSH_OPTIONS[@]}" "$REMOTE" \
    'command -v rsync >/dev/null 2>&1' ||
    die "cannot reach remote host or remote rsync is unavailable"
}

remote_directories_exist() {
  ssh "${SSH_OPTIONS[@]}" "$REMOTE" \
    "test -d '$REMOTE_ROOT/assignment1-basics' &&
     test -d '$REMOTE_ROOT/assignment2-systems'"
}

sync_assignment1() {
  local destination=$1
  local dry_run=$2
  local options=("${RSYNC_OPTIONS[@]}")

  if [[ "$dry_run" == true ]]; then
    options+=(--dry-run)
  fi

  rsync "${options[@]}" "${ASSIGNMENT1_FILTERS[@]}" \
    "$ASSIGNMENT1_ROOT/" "$destination"
}

sync_assignment2() {
  local destination=$1
  local dry_run=$2
  local options=("${RSYNC_OPTIONS[@]}")

  if [[ "$dry_run" == true ]]; then
    options+=(--dry-run)
  fi

  rsync "${options[@]}" "${ASSIGNMENT2_FILTERS[@]}" \
    "$ASSIGNMENT2_ROOT/" "$destination"
}

preview_remote_changes() {
  log "preview=assignment1-basics"
  sync_assignment1 "$REMOTE:$REMOTE_ROOT/assignment1-basics/" true
  log "preview=assignment2-systems"
  sync_assignment2 "$REMOTE:$REMOTE_ROOT/assignment2-systems/" true
}

preview_first_deployment() {
  TEMPORARY_ROOT=$(mktemp -d)

  mkdir -p "$TEMPORARY_ROOT/assignment1-basics"
  mkdir -p "$TEMPORARY_ROOT/assignment2-systems"

  log "remote_destination=missing"
  log "preview=assignment1-basics_against_empty_destination"
  sync_assignment1 "$TEMPORARY_ROOT/assignment1-basics/" true
  log "preview=assignment2-systems_against_empty_destination"
  sync_assignment2 "$TEMPORARY_ROOT/assignment2-systems/" true
  log "next_action=create_remote_directories_in_tmux_then_run_--apply"

  rm -rf -- "$TEMPORARY_ROOT"
  TEMPORARY_ROOT=""
}

confirm_apply() {
  if [[ "$ASSUME_YES" == true ]]; then
    return
  fi
  [[ -t 0 ]] || die "interactive confirmation unavailable; use --apply --yes"

  local reply
  read -r -p "Apply the changes shown above? [y/N] " reply
  [[ "$reply" == "y" || "$reply" == "Y" ]] || {
    log "result=cancelled"
    exit 0
  }
}

apply_changes() {
  log "apply=assignment1-basics"
  sync_assignment1 "$REMOTE:$REMOTE_ROOT/assignment1-basics/" false
  log "apply=assignment2-systems"
  sync_assignment2 "$REMOTE:$REMOTE_ROOT/assignment2-systems/" false
}

verify_idempotency() {
  local remaining_changes
  remaining_changes=$(
    {
      sync_assignment1 "$REMOTE:$REMOTE_ROOT/assignment1-basics/" true
      sync_assignment2 "$REMOTE:$REMOTE_ROOT/assignment2-systems/" true
    }
  )

  if [[ -n "$remaining_changes" ]]; then
    printf 'idempotency_check=failed\n%s\n' "$remaining_changes" >&2
    exit 1
  fi
  log "idempotency_check=passed"
}

main() {
  trap cleanup EXIT
  trap 'exit 129' HUP
  trap 'exit 130' INT
  trap 'exit 143' TERM

  parse_args "$@"
  validate_configuration
  acquire_lock

  log "mode=$MODE"
  log "remote=$REMOTE"
  log "remote_root=$REMOTE_ROOT"
  log "assignment1_source=$ASSIGNMENT1_ROOT"
  log "assignment2_source=$ASSIGNMENT2_ROOT"

  check_remote

  if [[ "$MODE" == "dry-run" ]]; then
    if remote_directories_exist; then
      preview_remote_changes
    else
      preview_first_deployment
    fi
    log "result=dry_run_complete"
    return
  fi

  remote_directories_exist || {
    die "remote destinations are missing; create them inside the cs336 tmux session first"
  }
  preview_remote_changes
  confirm_apply
  apply_changes
  verify_idempotency
  log "result=sync_complete"
}

main "$@"
