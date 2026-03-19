#!/usr/bin/env bash
# Helper: kill any process listening on a given port.
# Usage: source _kill_port.sh
#        kill_port <port>

kill_port() {
    local port="$1"
    local pid
    pid=$(lsof -ti :"$port" 2>/dev/null) || true
    if [[ -n "$pid" ]]; then
        echo "  Killing stale process $pid on port $port"
        kill "$pid" 2>/dev/null || true
        sleep 1
    fi
}
