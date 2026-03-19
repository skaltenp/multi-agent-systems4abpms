#!/usr/bin/env bash
# Helper: wait until an HTTP endpoint responds (up to TIMEOUT seconds).
# Usage: source _wait_for_server.sh
#        wait_for_server <url> [timeout_seconds]

wait_for_server() {
    local url="$1"
    local timeout="${2:-60}"
    local start=$SECONDS
    echo -n "  Waiting for $url "
    while ! curl --max-time 2 -sf -o /dev/null "$url" 2>/dev/null; do
        if (( SECONDS - start >= timeout )); then
            echo " TIMEOUT"
            echo "ERROR: Server at $url did not become ready within ${timeout}s" >&2
            return 1
        fi
        echo -n "."
        sleep 2
    done
    echo " ready"
}
