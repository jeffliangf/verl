#!/usr/bin/env bash
#SBATCH --job-name=rl-inspection-v1-1-16h100
#SBATCH --partition=all
#SBATCH --constraint=h100
#SBATCH --nodelist=gmicloud-loki-gpu-[004,005]
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --mem=1024G
#SBATCH --time=3-00:00:00
#SBATCH --no-requeue
#SBATCH --output=/home/jeff/rl-benchmarks/inspection-v1-1-group8-16h100-20261101/grpo/slurm-%j.log
set -euo pipefail
source /home/jeff/rl-benchmarks/inspection-v1-1-group8-16h100-20261101/grpo/cluster-env.sh
srun --kill-on-bad-exit=1 uv run --no-sync python "$RL_RUN_DIR/storage.py"
uv run --no-sync python "$RL_RUN_DIR/initialize.py"
mapfile -t rl_hosts < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
export RL_HEAD_NODE=${rl_hosts[0]}
RL_HEAD_IP=$(scontrol show node "$RL_HEAD_NODE" --oneliner | sed -n 's/.* NodeAddr=\([^ ]*\).*/\1/p')
export RL_HEAD_IP
[[ -n "$RL_HEAD_IP" && ${#rl_hosts[@]} -eq 2 ]]
printf 'RL_START job=%s head=%s nodes=%s\n' "$SLURM_JOB_ID" "$RL_HEAD_NODE" "$SLURM_JOB_NODELIST"
srun --kill-on-bad-exit=1 uv run --no-sync python "$RL_RUN_DIR/stage-model.py"
srun --kill-on-bad-exit=1 uv run --no-sync python "$RL_RUN_DIR/gemini-preflight.py"
# Static rendezvous preserves the numeric master address for every worker.
# Elastic c10d otherwise advertises a hostname that is not resolvable on all nodes.
srun --kill-on-bad-exit=1 bash -c '
  exec uv run --no-sync torchrun --nnodes=2 --nproc-per-node=8 \
    --rdzv-backend=static --node-rank="$SLURM_PROCID" \
    --master-addr="$RL_HEAD_IP" --master-port=27901 --rdzv-conf=timeout=180 \
    --rdzv-id="$SLURM_JOB_ID" "$RL_RUN_DIR/nccl-preflight.py"
'
srun --kill-on-bad-exit=1 --wait=300 bash "$RL_RUN_DIR/ray-node.sh"
