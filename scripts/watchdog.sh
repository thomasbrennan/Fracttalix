#!/usr/bin/env bash
# Fracttalix Coordinator Watchdog
#
# Runs OUTSIDE Claude Code. Monitors the checkpoint state file and
# relaunches the coordinator if it goes stale (indicating instance death).
#
# Usage:
#   ./scripts/watchdog.sh [--stale-threshold 300] [--max-restarts 5]
#
# How it works:
#   1. Watches .checkpoint/state.json modification time
#   2. If no update for STALE_THRESHOLD seconds, assumes coordinator death
#   3. Launches a new Claude Code instance with recovery instructions
#   4. The new instance reads state.json and picks up where the dead one left off
#
# Requirements:
#   - claude CLI installed and authenticated
#   - .checkpoint/state.json exists (coordinator must have initialized)
#
# Session S56 — 2026-03-13

set -euo pipefail

STALE_THRESHOLD=${1:-300}  # seconds before declaring death (default: 5 min)
MAX_RESTARTS=${2:-5}
RESTART_COUNT=0
CHECKPOINT_FILE=".checkpoint/state.json"
WATCHDOG_LOG=".checkpoint/watchdog.log"

log() {
    local msg="[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $1"
    echo "$msg"
    echo "$msg" >> "$WATCHDOG_LOG"
}

get_file_age() {
    if [[ ! -f "$CHECKPOINT_FILE" ]]; then
        echo "999999"
        return
    fi
    local mod_time
    mod_time=$(stat -c %Y "$CHECKPOINT_FILE" 2>/dev/null || stat -f %m "$CHECKPOINT_FILE" 2>/dev/null)
    local now
    now=$(date +%s)
    echo $(( now - mod_time ))
}

build_recovery_prompt() {
    cat <<'PROMPT'
You are a RECOVERY COORDINATOR for the Fracttalix project.

A previous coordinator instance has died (hit its ~200K token limit).
Your job is to resume operations seamlessly.

IMMEDIATE ACTIONS:
1. Run: python scripts/checkpoint.py status
2. Read the output carefully — it shows all tasks, their status, and progress
3. Run: python scripts/checkpoint.py validate
4. Assess what was in progress when the previous instance died
5. Resume or reassign incomplete tasks

RULES:
- Do NOT re-initialize the checkpoint (the state from the dead instance is valuable)
- Do NOT redo completed tasks
- Resume in_progress tasks from their last checkpoint
- Reassign any tasks that were assigned to workers that may also have died
- If the objective is complete, run: python scripts/checkpoint.py handoff --to NEXT_SESSION --context "..."

Read .checkpoint/PROTOCOL.md if you need the full protocol reference.
Read docs/claude-bootstrap.md for full project context.
PROMPT
}

# --- Main loop ---

log "Watchdog started. Stale threshold: ${STALE_THRESHOLD}s, max restarts: ${MAX_RESTARTS}"

if [[ ! -f "$CHECKPOINT_FILE" ]]; then
    log "ERROR: No checkpoint file found at $CHECKPOINT_FILE"
    log "Start a coordinator first: python scripts/checkpoint.py init --session SXX --role coordinator --objective '...'"
    exit 1
fi

log "Monitoring $CHECKPOINT_FILE..."

while true; do
    sleep 30  # Check every 30 seconds

    age=$(get_file_age)

    if (( age > STALE_THRESHOLD )); then
        log "ALERT: Checkpoint stale for ${age}s (threshold: ${STALE_THRESHOLD}s)"
        log "Coordinator presumed dead."

        RESTART_COUNT=$((RESTART_COUNT + 1))

        if (( RESTART_COUNT > MAX_RESTARTS )); then
            log "FATAL: Max restarts ($MAX_RESTARTS) exceeded. Manual intervention required."
            exit 1
        fi

        log "Launching recovery coordinator (restart #${RESTART_COUNT}/${MAX_RESTARTS})..."

        # Launch new Claude Code instance with recovery prompt
        RECOVERY_PROMPT=$(build_recovery_prompt)

        # Using claude CLI in non-interactive mode
        # The new instance will read state.json and resume
        claude --print --dangerously-skip-permissions "$RECOVERY_PROMPT" 2>&1 | tee -a "$WATCHDOG_LOG" &

        log "Recovery coordinator launched (PID: $!)"
        log "Waiting for new coordinator to start updating checkpoint..."

        # Wait for the new instance to update the checkpoint file
        sleep 60
    fi
done
