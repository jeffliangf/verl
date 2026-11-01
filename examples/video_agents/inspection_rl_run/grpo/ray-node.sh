#!/usr/bin/env bash
set -euo pipefail
source /home/jeff/rl-benchmarks/inspection-v1-1-group8-16h100-20261101/grpo/cluster-env.sh
source "$RL_RUN_DIR/shutdown-sync.sh"
rl_phase=training
rl_node=$(hostname)
rl_node_ip=$(ip -4 -o addr show dev ens10f0np0 scope global | awk 'NR==1 {split($4,a,"/"); print a[1]}')
rl_ray_port=$((30000 + SLURM_JOB_ID % 10000))
export RAY_ADDRESS="$RL_HEAD_IP:$rl_ray_port"
export RL_MODEL_PATH="$RL_LOCAL_MODEL"
rl_ray_dir="/tmp/jeff-video-agents-ray-$SLURM_JOB_ID"
rl_control="$RL_RUN_DIR/control-$SLURM_JOB_ID"
mkdir -p "$rl_ray_dir" "$rl_control"
rl_ssh_pid=''
rl_monitor_pid=''
rl_ray_pid=''
rl_guard_pid=''
rl_launcher_pid=$$
guard_processes() {
  while sleep 10; do
    for rl_required_pid in "$rl_ssh_pid" "$rl_ray_pid"; do
      if ! kill -0 "$rl_required_pid" 2>/dev/null; then
        rl_lifecycle_log "ERROR: required process $rl_required_pid exited; stopping this node launcher"
        kill -TERM "$rl_launcher_pid"
        return 1
      fi
    done
  done
}
cleanup() {
  local rl_exit_code=$? rl_pid rl_pending rl_deadline
  trap - EXIT
  rl_lifecycle_log "Cleaning up job-owned processes; node exit code=$rl_exit_code"
  for rl_pid in "$rl_guard_pid" "$rl_ssh_pid" "$rl_monitor_pid" "$rl_ray_pid"; do
    if [[ -n "$rl_pid" ]]; then kill "$rl_pid" 2>/dev/null || true; fi
  done
  rl_deadline=$((SECONDS + 30))
  while true; do
    rl_pending=0
    for rl_pid in "$rl_guard_pid" "$rl_ssh_pid" "$rl_monitor_pid" "$rl_ray_pid"; do
      if [[ -n "$rl_pid" ]] && kill -0 "$rl_pid" 2>/dev/null; then
        rl_pending=$((rl_pending + 1))
        if [[ $SECONDS -ge $rl_deadline ]]; then kill -KILL "$rl_pid" 2>/dev/null || true; fi
      fi
    done
    if [[ $rl_pending -eq 0 || $SECONDS -ge $rl_deadline ]]; then break; fi
    sleep 1
  done
  rl_lifecycle_log "Node launcher exiting with code=$rl_exit_code"
  return "$rl_exit_code"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
ssh -p 44222 -i /home/jeff/.ssh/gmi -o IdentitiesOnly=yes -o BatchMode=yes \
  -o ConnectTimeout=15 -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 \
  -o ServerAliveCountMax=3 -N -L 127.0.0.1:28080:10.2.201.187:18080 \
  jeff@157.66.255.187 > "$RL_RUN_DIR/h3-tunnel-$SLURM_JOB_ID-$rl_node.log" 2>&1 &
rl_ssh_pid=$!
uv run --no-sync python - <<'PY'
import socket, time
deadline=time.monotonic()+30
while time.monotonic()<deadline:
    try:
        with socket.create_connection(('127.0.0.1',28080),timeout=2): break
    except OSError: time.sleep(1)
else: raise RuntimeError('H3 tunnel did not start')
PY
uv run --no-sync python -m examples.video_agents.rl status --run-dir "$RL_RUN_DIR" \
  > "$RL_RUN_DIR/h3-status-$SLURM_JOB_ID-$rl_node.json"
uv run --no-sync python "$RL_RUN_DIR/gpu_metrics.py" --run-dir "$RL_RUN_DIR" --job-id "$SLURM_JOB_ID" \
  > "$RL_RUN_DIR/gpu-telemetry-$SLURM_JOB_ID-$rl_node.log" 2>&1 &
rl_monitor_pid=$!
if [[ "$rl_node" == "$RL_HEAD_NODE" ]]; then
  ray start --head --node-ip-address="$rl_node_ip" --port="$rl_ray_port" \
    --num-cpus=64 --num-gpus=8 --object-store-memory=17179869184 \
    --include-dashboard=false --temp-dir="$rl_ray_dir" --disable-usage-stats --block \
    > "$RL_RUN_DIR/ray-$SLURM_JOB_ID-$rl_node.log" 2>&1 &
  rl_ray_pid=$!
  uv run --no-sync python - <<'PY'
import os, socket, time
host,port=os.environ['RAY_ADDRESS'].split(':')
deadline=time.monotonic()+180
while time.monotonic()<deadline:
    try:
        with socket.create_connection((host,int(port)),timeout=2): break
    except OSError: time.sleep(2)
else: raise RuntimeError('Ray head did not start')
import ray
ray.init(address=os.environ['RAY_ADDRESS'],logging_level='ERROR')
deadline=time.monotonic()+300
while time.monotonic()<deadline:
    nodes=[n for n in ray.nodes() if n['Alive']]
    if len(nodes)==2 and sum(n['Resources'].get('GPU',0) for n in nodes)==16:
        print('RAY_CLUSTER_READY: 2 nodes, 16 GPUs',flush=True)
        break
    time.sleep(2)
else: raise RuntimeError('Ray did not register both allocated nodes with 16 GPUs')
ray.shutdown()
PY
  guard_processes &
  rl_guard_pid=$!
  bash "$RL_RUN_DIR/train.sh" execute
  kill "$rl_guard_pid" 2>/dev/null || true
  wait "$rl_guard_pid" 2>/dev/null || true
  rl_guard_pid=''
  rl_publish_driver_success_and_wait
else
  uv run --no-sync python - <<'PY'
import os, socket, time
host,port=os.environ['RAY_ADDRESS'].split(':')
deadline=time.monotonic()+180
while time.monotonic()<deadline:
    try:
        with socket.create_connection((host,int(port)),timeout=2): break
    except OSError: time.sleep(2)
else: raise RuntimeError('Ray head did not become reachable')
PY
  ray start --address="$RAY_ADDRESS" --node-ip-address="$rl_node_ip" \
    --num-cpus=64 --num-gpus=8 --object-store-memory=17179869184 \
    --temp-dir="$rl_ray_dir" --disable-usage-stats --block \
    > "$RL_RUN_DIR/ray-$SLURM_JOB_ID-$rl_node.log" 2>&1 &
  rl_ray_pid=$!
  guard_processes &
  rl_guard_pid=$!
  rl_wait_for_driver_and_acknowledge
fi
