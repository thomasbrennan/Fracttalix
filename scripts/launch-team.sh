#!/usr/bin/env bash
# Fracttalix Team Launcher
#
# Launches the orchestrator inside a persistent tmux session that survives
# terminal disconnects, SSH drops, and laptop sleep/wake cycles.
#
# Usage:
#   ./scripts/launch-team.sh --session S57 --objective "Build feature X" [options]
#   ./scripts/launch-team.sh --attach          # Reattach to running session
#   ./scripts/launch-team.sh --status          # Check status without attaching
#   ./scripts/launch-team.sh --stop            # Stop the orchestrator
#   ./scripts/launch-team.sh --logs            # Tail the watchdog log
#
# The tmux session persists until explicitly stopped or the machine reboots.
# You can disconnect, close your terminal, and reconnect later.
#
# Session S56 — 2026-03-13

set -euo pipefail

TMUX_SESSION="fracttalix-team"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ORCHESTRATOR="$REPO_ROOT/scripts/orchestrator.py"
CHECKPOINT_DIR="$REPO_ROOT/.checkpoint"
LOG_FILE="$CHECKPOINT_DIR/watchdog.log"

# Defaults
SESSION=""
OBJECTIVE=""
TEAM_SIZE=4
STALE_SECONDS=300
MAX_GENERATIONS=10

usage() {
    cat <<EOF
Fracttalix Team Launcher — persistent orchestrator via tmux

Start a new team:
  $0 --session S57 --objective "Build feature X" [--team-size 4] [--stale 300] [--max-gen 10]

Manage running team:
  $0 --attach          Reattach to the tmux session (Ctrl-B D to detach again)
  $0 --status          Show orchestrator + checkpoint status
  $0 --stop            Stop the orchestrator gracefully
  $0 --logs            Tail the watchdog log live
  $0 --kill            Force-kill the tmux session
EOF
}

start_team() {
    # Check if already running
    if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
        echo "ERROR: Team session '$TMUX_SESSION' is already running."
        echo "Use --attach to reconnect, --stop to stop, or --kill to force-kill."
        exit 1
    fi

    if [[ -z "$SESSION" || -z "$OBJECTIVE" ]]; then
        echo "ERROR: --session and --objective are required."
        usage
        exit 1
    fi

    echo "Starting Fracttalix Team Orchestrator..."
    echo "  Session:        $SESSION"
    echo "  Objective:      $OBJECTIVE"
    echo "  Team size:      $TEAM_SIZE"
    echo "  Stale timeout:  ${STALE_SECONDS}s"
    echo "  Max generations: $MAX_GENERATIONS"
    echo "  tmux session:   $TMUX_SESSION"
    echo ""

    # Create tmux session in detached mode
    tmux new-session -d -s "$TMUX_SESSION" -c "$REPO_ROOT" \
        "python '$ORCHESTRATOR' start \
            --session '$SESSION' \
            --objective '$OBJECTIVE' \
            --team-size $TEAM_SIZE \
            --stale-seconds $STALE_SECONDS \
            --max-generations $MAX_GENERATIONS \
        2>&1 | tee -a '$LOG_FILE'; \
        echo ''; \
        echo '=== Orchestrator exited. Press Enter to close this tmux pane. ==='; \
        read"

    echo "Orchestrator launched in tmux session '$TMUX_SESSION'."
    echo ""
    echo "Commands:"
    echo "  Attach:    $0 --attach     (then Ctrl-B D to detach)"
    echo "  Status:    $0 --status"
    echo "  Logs:      $0 --logs"
    echo "  Stop:      $0 --stop"
}

attach_session() {
    if ! tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
        echo "No active team session found."
        exit 1
    fi
    echo "Attaching to '$TMUX_SESSION' (Ctrl-B D to detach)..."
    tmux attach-session -t "$TMUX_SESSION"
}

show_status() {
    echo "=== tmux Session ==="
    if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
        echo "  tmux session '$TMUX_SESSION': RUNNING"
    else
        echo "  tmux session '$TMUX_SESSION': NOT RUNNING"
    fi
    echo ""

    echo "=== Orchestrator ==="
    if [[ -f "$CHECKPOINT_DIR/orchestrator.json" ]]; then
        python "$ORCHESTRATOR" status
    else
        echo "  No orchestrator state found."
    fi
    echo ""

    echo "=== Checkpoint ==="
    if [[ -f "$CHECKPOINT_DIR/state.json" ]]; then
        python "$REPO_ROOT/scripts/checkpoint.py" status
    else
        echo "  No checkpoint state found."
    fi
}

stop_orchestrator() {
    echo "Stopping orchestrator..."
    python "$ORCHESTRATOR" stop 2>/dev/null || true

    # Give it a moment to clean up
    sleep 2

    if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
        echo "Closing tmux session..."
        tmux kill-session -t "$TMUX_SESSION" 2>/dev/null || true
    fi
    echo "Stopped."
}

tail_logs() {
    if [[ ! -f "$LOG_FILE" ]]; then
        echo "No log file found at $LOG_FILE"
        exit 1
    fi
    echo "Tailing $LOG_FILE (Ctrl-C to stop)..."
    tail -f "$LOG_FILE"
}

kill_session() {
    if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
        tmux kill-session -t "$TMUX_SESSION"
        echo "Killed tmux session '$TMUX_SESSION'."
    else
        echo "No session to kill."
    fi
}

# --- Parse arguments ---

if [[ $# -eq 0 ]]; then
    usage
    exit 0
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --session)     SESSION="$2"; shift 2 ;;
        --objective)   OBJECTIVE="$2"; shift 2 ;;
        --team-size)   TEAM_SIZE="$2"; shift 2 ;;
        --stale)       STALE_SECONDS="$2"; shift 2 ;;
        --max-gen)     MAX_GENERATIONS="$2"; shift 2 ;;
        --attach)      attach_session; exit 0 ;;
        --status)      show_status; exit 0 ;;
        --stop)        stop_orchestrator; exit 0 ;;
        --logs)        tail_logs; exit 0 ;;
        --kill)        kill_session; exit 0 ;;
        --help|-h)     usage; exit 0 ;;
        *)             echo "Unknown option: $1"; usage; exit 1 ;;
    esac
done

start_team
