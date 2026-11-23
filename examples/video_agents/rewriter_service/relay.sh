#!/bin/bash
# Public entry point: forwards the login node's 0.0.0.0:$PORT to the gateway on the node running the service job.
# Compute-node ports are filtered from outside the cluster, the login node's are not (checked from the arena VM on
# 2026-11-13). This is a plain TCP pipe; the gateway authenticates every call except /health. backlog=1024 because
# socat's default (5) overflowed when 135 clients connected at once and some saw "connection reset".
#   relay.sh start [node] | stop | status        (restart it after the service job moves to another node)
SVC=/home/jeff/rl-benchmarks/rl-rewriter-s300-service
PORT=${PORT:-18400}
PIDFILE=$SVC/logs/relay.pid
running() { [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; }
case "${1:-status}" in
  start)
    running && { echo "relay already running (pid $(cat "$PIDFILE"))"; exit 0; }
    node=${2:-$(squeue -u "$USER" -n rl-rewriter-s300-svc -h -t R -o %N | head -1)}
    [ -n "$node" ] || { echo "no running rl-rewriter-s300-svc job"; exit 1; }
    target=$(scontrol show node "$node" | grep -oP "NodeAddr=\K\S+")
    nohup setsid socat TCP-LISTEN:"$PORT",fork,reuseaddr,backlog=1024,bind=0.0.0.0 TCP:"$target":"$PORT" \
      >> "$SVC/logs/relay.log" 2>&1 < /dev/null &
    echo $! > "$PIDFILE"
    echo "[$(date -Is)] relay $(hostname -s):$PORT -> $node ($target):$PORT pid $!" | tee -a "$SVC/logs/relay.log"
    ;;
  stop)
    running && kill "$(cat "$PIDFILE")" && echo "relay stopped" || echo "relay not running"
    rm -f "$PIDFILE"
    ;;
  status)
    running && echo "relay running (pid $(cat "$PIDFILE")): $(tail -1 "$SVC/logs/relay.log")" || echo "relay not running"
    ;;
esac
