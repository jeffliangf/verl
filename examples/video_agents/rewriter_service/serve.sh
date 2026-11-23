#!/bin/bash
# Serve the RL step-300 rewriter on this node behind gateway.py on 0.0.0.0:$PORT (Bearer token in
# secrets/api_token). Per group of TP GPUs (INSTANCES groups):
#   - a text-only vLLM server on BASE_PORT+i, image limit 0: text to video, built and served exactly as before
#     2026-11-23 (only its memory share changed);
#   - with I2V_ENGINE=1 (default), an image-enabled vLLM server (one image per prompt) on I2V_BASE_PORT+i for
#     image-to-video requests only. Image-enabled vLLM compiles a different graph, which shifts text-to-video
#     log-probabilities at bf16 rounding level, so text never goes there. It starts only once the text server
#     answers /health, so the two servers' memory profiling never overlaps. Each server gets GPU_UTIL of the memory.
#   I2V_ENGINE=0 runs one image-enabled server per group for both tasks (GPU_UTIL 0.88).
# Every server is restarted if it exits. The gateway starts at once and queues work until a backend is healthy.
# Instance i runs on the i-th group of TP GPUs of the job's Slurm allocation, so the job can share its node with
# other jobs.
set -uo pipefail
umask 022   # logs and task records are read by the utopai group
SVC=/home/jeff/rl-benchmarks/rl-rewriter-s300-service
PY=/home/jeff/.venvs/video-agents-h100-f169e786/bin/python
MODEL=/home/jeff/rl-benchmarks/rl-rewriter-vs-context-ir-candidate8-20261112/model/global_step_300-hf
INSTANCES=${INSTANCES:-1} TP=${TP:-4} PORT=${PORT:-18400} BASE_PORT=${BASE_PORT:-18201}
I2V_ENGINE=${I2V_ENGINE:-1} I2V_BASE_PORT=${I2V_BASE_PORT:-18301}
if (( I2V_ENGINE )); then GPU_UTIL=${GPU_UTIL:-0.44}; else GPU_UTIL=${GPU_UTIL:-0.88}; fi
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
echo "[$(date -Is)] $HOST job=$JOB instances=$INSTANCES tp=$TP gpus=${GPUS[*]} port=$PORT i2v_engine=$I2V_ENGINE" \
     "gpu_util=$GPU_UTIL max_model_len=$MAX_MODEL_LEN max_num_seqs=$MAX_NUM_SEQS max_tokens=$MAX_TOKENS"
if (( ${#GPUS[@]} < INSTANCES * TP )); then
  echo "[$(date -Is)] $INSTANCES x TP $TP needs $((INSTANCES * TP)) GPUs, the job has ${#GPUS[@]}; exiting"
  exit 1
fi

healthy() { "$PY" -c "import sys, urllib.request; urllib.request.urlopen(sys.argv[1], timeout=5)" \
              "http://127.0.0.1:$1/health" >/dev/null 2>&1; }

supervise_vllm() {   # group index, port, images per prompt (0 = text only), port that must be healthy first or ""
  local i=$1 port=$2 images=$3 after=$4 n=0 gpus root=$C/vllm
  gpus=$(IFS=,; echo "${GPUS[*]:$((i * TP)):$TP}")
  # vLLM 0.29 keys its AOT compile cache (torch_aot_compile/<hash>) without the image limit and loads it without
  # guard checks, so a text-only and an image-enabled server sharing a cache root load each other's compiled
  # forward; the image server then crashes in profile_run (2026-11-23). Image servers get their own root; the
  # text-only server keeps the original one.
  (( images )) && root=$C/vllm-images
  while :; do
    if [ -n "$after" ]; then
      until healthy "$after"; do sleep 10; done
    fi
    n=$((n + 1))
    echo "[$(date -Is)] vLLM $i:$port start #$n gpus=$gpus images=$images cache=$root"
    CUDA_VISIBLE_DEVICES=$gpus VLLM_CACHE_ROOT=$root "$PY" -m vllm.entrypoints.openai.api_server --model "$MODEL" \
      --served-model-name rewriter-s300 --host 127.0.0.1 --port "$port" --tensor-parallel-size "$TP" \
      --dtype bfloat16 --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$MAX_NUM_SEQS" \
      --gpu-memory-utilization "$GPU_UTIL" --generation-config vllm \
      --limit-mm-per-prompt "{\"image\": $images, \"video\": 0}" \
      --seed 0 >> "$LOGS/vllm-$HOST-$port-$JOB.log" 2>&1
    echo "[$(date -Is)] vLLM $i:$port exited with $? (start #$n); restarting in 30 s"
    sleep 30
  done
}

backends="" i2v_backends=""
for i in $(seq 0 $((INSTANCES - 1))); do
  if (( I2V_ENGINE )); then
    supervise_vllm "$i" $((BASE_PORT + i)) 0 "" &
    supervise_vllm "$i" $((I2V_BASE_PORT + i)) 1 $((BASE_PORT + i)) &
    i2v_backends+="${i2v_backends:+,}http://127.0.0.1:$((I2V_BASE_PORT + i))"
  else
    supervise_vllm "$i" $((BASE_PORT + i)) 1 "" &
  fi
  backends+="${backends:+,}http://127.0.0.1:$((BASE_PORT + i))"
done
trap 'kill $(jobs -p) 2>/dev/null; pkill -TERM -P $$ 2>/dev/null' EXIT
export REWRITER_BACKENDS=$backends REWRITER_I2V_BACKENDS=${i2v_backends:-$backends} \
       REWRITER_TOKEN_FILE=$SVC/secrets/api_token REWRITER_LOG_DIR=$LOGS \
       REWRITER_MAX_MODEL_LEN=$MAX_MODEL_LEN REWRITER_MAX_TOKENS=$MAX_TOKENS REWRITER_SLOTS_PER_BACKEND=$MAX_NUM_SEQS
while :; do
  echo "[$(date -Is)] gateway start on 0.0.0.0:$PORT backends=$backends i2v_backends=${i2v_backends:-$backends}"
  "$PY" -m uvicorn --app-dir "$SVC" gateway:app --host 0.0.0.0 --port "$PORT" --no-access-log \
    --timeout-keep-alive 30 >> "$LOGS/gateway-$HOST-$JOB.log" 2>&1
  echo "[$(date -Is)] gateway exited with $?; restarting in 10 s"
  sleep 10
done
