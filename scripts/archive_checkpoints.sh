#!/bin/bash
#
# Archive DreamerV3 `latest.pt` checkpoints during long training runs.
#
# Purpose:
# - NM512's `external/dreamerv3-torch/dreamer.py` only writes one checkpoint
#   file, `latest.pt`, inside the training logdir.
# - That file is overwritten every `eval_every` steps.
# - This helper script monitors the logdir and copies each refreshed
#   `latest.pt` into an archive directory as `checkpoint_00001.pt`,
#   `checkpoint_00002.pt`, and so on, so intermediate training snapshots are
#   preserved for DriftDetect Month 3 dynamics analysis.
#
# Important:
# - Start this script AFTER the DreamerV3 training process has begun.
# - On Linux / RunPod, it will also try to read the latest training step from
#   `metrics.jsonl` and include it in the console log message when available.
#
# Usage:
#   chmod +x scripts/archive_checkpoints.sh
#
#   Terminal 1 (start training):
#     cd external/dreamerv3-torch
#     python dreamer.py \
#       --configs dmc_proprio \
#       --task dmc_cheetah_run \
#       --logdir ../../results/training_dynamics/cheetah/ \
#       --steps 500000 \
#       --batch_size 32 \
#       --envs 8 \
#       --eval_every 5000
#
#   Terminal 2 (start archiving):
#     bash scripts/archive_checkpoints.sh results/training_dynamics/cheetah/
#
# Arguments:
#   $1  Logdir to monitor.
#       Default: results/training_dynamics/cheetah/ (resolved from repo root)
#   $2  Archive subdirectory name or path.
#       Default: checkpoints_archive (inside the logdir)
#
# Environment variables:
#   POLL_INTERVAL     Seconds between checks. Default: 30
#   MAX_CHECKPOINTS   Stop after archiving this many new checkpoints.
#                     Default: 110

set -u

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)

DEFAULT_LOGDIR="$REPO_ROOT/results/training_dynamics/cheetah"
LOGDIR_INPUT="${1:-$DEFAULT_LOGDIR}"
ARCHIVE_INPUT="${2:-checkpoints_archive}"
POLL_INTERVAL="${POLL_INTERVAL:-30}"
MAX_CHECKPOINTS="${MAX_CHECKPOINTS:-110}"

STOP_REQUESTED=0
SESSION_ARCHIVED=0

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

log() {
  printf '[%s] %s\n' "$(timestamp)" "$*"
}

resolve_path() {
  case "$1" in
    /*)
      printf '%s\n' "$1"
      ;;
    *)
      printf '%s\n' "$REPO_ROOT/$1"
      ;;
  esac
}

get_mtime() {
  if stat -c %Y "$1" >/dev/null 2>&1; then
    stat -c %Y "$1"
  else
    stat -f %m "$1"
  fi
}

get_size_bytes() {
  if stat -c %s "$1" >/dev/null 2>&1; then
    stat -c %s "$1"
  else
    stat -f %z "$1"
  fi
}

format_size_mb() {
  awk -v bytes="$1" 'BEGIN { printf "%.0fMB", bytes / (1024 * 1024) }'
}

detect_dreamer_running() {
  ps -eo pid=,args= \
    | grep '[p]ython' \
    | grep '[d]reamer' \
    | grep -v '[a]rchive_checkpoints.sh' >/dev/null 2>&1
}

infer_latest_step() {
  metrics_file="$LOGDIR/metrics.jsonl"
  if [ ! -f "$metrics_file" ]; then
    return 1
  fi

  last_line=$(tail -n 1 "$metrics_file" 2>/dev/null)
  if [ -z "$last_line" ]; then
    return 1
  fi

  step=$(printf '%s\n' "$last_line" | sed -n 's/.*"step"[[:space:]]*:[[:space:]]*\([0-9][0-9]*\).*/\1/p')
  if [ -n "$step" ]; then
    printf '%s\n' "$step"
    return 0
  fi
  return 1
}

summarize_exit() {
  end_epoch=$(date +%s)
  elapsed_seconds=$((end_epoch - START_EPOCH))
  elapsed_h=$((elapsed_seconds / 3600))
  elapsed_m=$(((elapsed_seconds % 3600) / 60))
  elapsed_s=$((elapsed_seconds % 60))

  if [ -d "$ARCHIVE_DIR" ]; then
    total_usage=$(du -sh "$ARCHIVE_DIR" 2>/dev/null | awk '{print $1}')
    total_files=$(find "$ARCHIVE_DIR" -maxdepth 1 -type f -name 'checkpoint_*.pt' | wc -l | awk '{print $1}')
  else
    total_usage="0B"
    total_files=0
  fi

  log "Archive summary: session_archived=$SESSION_ARCHIVED total_archived=$total_files disk_usage=$total_usage elapsed=${elapsed_h}h${elapsed_m}m${elapsed_s}s"
}

handle_interrupt() {
  STOP_REQUESTED=1
  log "Interrupt received. Finishing current iteration and exiting gracefully."
}

trap 'handle_interrupt' INT TERM

LOGDIR=$(resolve_path "$LOGDIR_INPUT")
case "$ARCHIVE_INPUT" in
  /*)
    ARCHIVE_DIR="$ARCHIVE_INPUT"
    ;;
  *)
    ARCHIVE_DIR="$LOGDIR/$ARCHIVE_INPUT"
    ;;
esac

LATEST_PT="$LOGDIR/latest.pt"
mkdir -p "$ARCHIVE_DIR"

COUNTER=0
LAST_ARCHIVE_FILE=""

for existing in "$ARCHIVE_DIR"/checkpoint_*.pt; do
  [ -e "$existing" ] || continue
  base_name=$(basename "$existing")
  number=$(printf '%s\n' "$base_name" | sed -n 's/^checkpoint_\([0-9][0-9][0-9][0-9][0-9]\)\.pt$/\1/p')
  [ -n "$number" ] || continue
  number=$((10#$number))
  if [ "$number" -gt "$COUNTER" ]; then
    COUNTER="$number"
    LAST_ARCHIVE_FILE="$existing"
  fi
done

if [ -n "$LAST_ARCHIVE_FILE" ] && [ -f "$LAST_ARCHIVE_FILE" ]; then
  LAST_ARCHIVED_MTIME=$(get_mtime "$LAST_ARCHIVE_FILE")
else
  LAST_ARCHIVED_MTIME=0
fi

START_EPOCH=$(date +%s)

log "Monitoring logdir: $LOGDIR"
log "Archive directory: $ARCHIVE_DIR"
log "Poll interval: ${POLL_INTERVAL}s | Max new checkpoints this session: $MAX_CHECKPOINTS"

while [ "$STOP_REQUESTED" -eq 0 ]; do
  if [ -f "$LATEST_PT" ]; then
    latest_mtime=$(get_mtime "$LATEST_PT")
    if [ "$latest_mtime" -gt "$LAST_ARCHIVED_MTIME" ]; then
      COUNTER=$((COUNTER + 1))
      archive_name=$(printf 'checkpoint_%05d.pt' "$COUNTER")
      archive_path="$ARCHIVE_DIR/$archive_name"

      cp "$LATEST_PT" "$archive_path"

      LAST_ARCHIVED_MTIME="$latest_mtime"
      LAST_ARCHIVE_FILE="$archive_path"
      SESSION_ARCHIVED=$((SESSION_ARCHIVED + 1))

      checkpoint_bytes=$(get_size_bytes "$archive_path")
      checkpoint_size=$(format_size_mb "$checkpoint_bytes")

      if step=$(infer_latest_step); then
        log "Archived $archive_name ($checkpoint_size, step=$step)"
      else
        log "Archived $archive_name ($checkpoint_size)"
      fi

      if [ "$SESSION_ARCHIVED" -ge "$MAX_CHECKPOINTS" ]; then
        log "Reached MAX_CHECKPOINTS=$MAX_CHECKPOINTS. Exiting."
        break
      fi
    fi
  fi

  if ! detect_dreamer_running; then
    log "No active python process with 'dreamer' in its command was found. Exiting."
    break
  fi

  sleep "$POLL_INTERVAL"
done

summarize_exit
