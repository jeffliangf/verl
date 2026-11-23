#!/bin/bash
# Serve the RL step-300 rewriter on this node: INSTANCES vLLM servers (TP GPUs each) on localhost, each restarted
# if it exits, behind gateway.py on 0.0.0.0:$PORT (Bearer token in secrets/api_token). The gateway starts at once
# and queues work until a backend is healthy. Instance i runs on the i-th group of TP GPUs of the job's Slurm
# allocation, so the job can share its node with other jobs.
set -uo pipefail
umask 022   # logs and task records are read by the utopai group
SVC=/home/jeff/rl-benchmarks/rl-rewriter-s300-service
PY=/home/jeff/.venvs/video-agents-h100-f169e786/bin/python
MODEL=/home/jeff/rl-benchmarks/rl-rewriter-vs-context-ir-candidate8-20261112/model/global_step_300-hf
INSTANCES=${INSTANCES:-1} TP=${TP:-4} PORT=${PORT:-18400} BASE_PORT=${BASE_PORT:-18201}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-40960} MAX_NUM_SEQS=${MAX_NUM_SEQS:-64} MAX_TOKENS=${MAX_TOKENS:-24576}
HOST=$(hostname -s); LOGS=$SVC/logs; JOB=${SLURM_JOB_ID:-local}
# The job's GPUs as Slurm exposes them (cgroup ConstrainDevices: local indices); outside Slurm, 0..INSTANCES*TP-1.
IFS=, read -ra GPUS <<< "${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((INSTANCES * TP - 1)))}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
# As the RL jobs (common/cluster-env.sh): FlashInfer all-reduce off (its JIT lock fails on the NFS home with
# ENOLCK), compiler caches on node-local disk.
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_ALLREDUCE_USE_FLASHINFER=0
C=/tmp/jeff-rl-rewriter-service-cache
export TRITON_CACHE_DIR=$C/triton TORCHINDUCTOR_CACHE_DIR=$C/inductor CUDA_CACHE_PATH=$C/cuda VLLM_CACHE_ROOT=$C/vllm
mkdir -p "$LOGS"
echo "[$(date -Is)] $HOST job=$JOB instances=$INSTANCES tp=$TP gpus=${GPUS[*]} port=$PORT" \
     "max_model_len=$MAX_MODEL_LEN max_num_seqs=$MAX_NUM_SEQS max_tokens=$MAX_TOKENS"
if (( ${#GPUS[@]} < INSTANCES * TP )); then
  echo "[$(date -Is)] $INSTANCES x TP $TP needs $((INSTANCES * TP)) GPUs, the job has ${#GPUS[@]}; exiting"
  exit 1
fi

supervise_vllm() {
  local i=$1 port=$((BASE_PORT + $1)) n=0 gpus
  gpus=$(IFS=,; echo "${GPUS[*]:$((i * TP)):$TP}")
  while :; do
    n=$((n + 1))
    echo "[$(date -Is)] vLLM $i start #$n gpus=$gpus port=$port"
    CUDA_VISIBLE_DEVICES=$gpus "$PY" -m vllm.entrypoints.openai.api_server --model "$MODEL" \
      --served-model-name rewriter-s300 --host 127.0.0.1 --port "$port" --tensor-parallel-size "$TP" \
      --dtype bfloat16 --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$MAX_NUM_SEQS" \
      --gpu-memory-utilization 0.88 --generation-config vllm --limit-mm-per-prompt '{"image": 0, "video": 0}' \
      --seed 0 >> "$LOGS/vllm-$HOST-$port-$JOB.log" 2>&1
    echo "[$(date -Is)] vLLM $i exited with $? (start #$n); restarting in 30 s"
    sleep 30
  done
}

backends=""
for i in $(seq 0 $((INSTANCES - 1))); do
  supervise_vllm "$i" &
  backends+="${backends:+,}http://127.0.0.1:$((BASE_PORT + i))"
done
trap 'kill $(jobs -p) 2>/dev/null; pkill -TERM -P $$ 2>/dev/null' EXIT
export REWRITER_BACKENDS=$backends REWRITER_TOKEN_FILE=$SVC/secrets/api_token REWRITER_LOG_DIR=$LOGS \
       REWRITER_MAX_MODEL_LEN=$MAX_MODEL_LEN REWRITER_MAX_TOKENS=$MAX_TOKENS REWRITER_SLOTS_PER_BACKEND=$MAX_NUM_SEQS
while :; do
  echo "[$(date -Is)] gateway start on 0.0.0.0:$PORT backends=$backends"
  "$PY" -m uvicorn --app-dir "$SVC" gateway:app --host 0.0.0.0 --port "$PORT" --no-access-log \
    --timeout-keep-alive 30 >> "$LOGS/gateway-$HOST-$JOB.log" 2>&1
  echo "[$(date -Is)] gateway exited with $?; restarting in 10 s"
  sleep 10
done
