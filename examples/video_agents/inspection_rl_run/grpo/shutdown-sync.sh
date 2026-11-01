#!/usr/bin/env bash

rl_lifecycle_log() {
  printf '%s [%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rl_node" "$*" \
    | tee -a "$RL_RUN_DIR/lifecycle-$SLURM_JOB_ID-$rl_node.log"
}

rl_publish_driver_success_and_wait() {
  local rl_peer rl_pending rl_deadline
  local -a rl_peers
  mapfile -t rl_peers < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
  [[ ${#rl_peers[@]} -eq $SLURM_NNODES ]] || return 1
  touch "$rl_control/driver.done"
  rl_lifecycle_log 'Driver returned success; waiting for worker acknowledgments'
  rl_deadline=$((SECONDS + ${RL_SHUTDOWN_ACK_TIMEOUT:-240}))
  while true; do
    rl_pending=0
    for rl_peer in "${rl_peers[@]}"; do
      if [[ "$rl_peer" != "$rl_node" && ! -f "$rl_control/worker-$rl_peer.done" ]]; then
        rl_pending=$((rl_pending + 1))
      fi
    done
    if [[ $rl_pending -eq 0 ]]; then
      rl_lifecycle_log 'All workers acknowledged driver success; shutdown may begin'
      return 0
    fi
    if [[ $SECONDS -ge $rl_deadline ]]; then
      rl_lifecycle_log "ERROR: $rl_pending workers did not acknowledge completion before the deadline"
      return 1
    fi
    sleep 2
  done
}

rl_wait_for_driver_and_acknowledge() {
  while [[ ! -f "$rl_control/driver.done" ]]; do
    if ! kill -0 "$rl_ray_pid" "$rl_ssh_pid" 2>/dev/null; then
      rl_lifecycle_log 'ERROR: a required local process exited before driver completion'
      return 1
    fi
    sleep 2
  done
  touch "$rl_control/worker-$rl_node.done"
  rl_lifecycle_log 'Observed driver success and wrote worker acknowledgment'
}
