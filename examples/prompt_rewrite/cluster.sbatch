#!/usr/bin/env bash
#SBATCH --job-name=prompt-rewrite-sft
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --mem=1024G
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=outputs/prompt_rewrite/slurm-%j.log
set -euo pipefail
umask 077

# Submit from the staged checkout; create outputs/prompt_rewrite before sbatch.
export UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT:-/raid/jeff/prompt-rewrite-venv}
export PATH="$HOME/.local/bin:$UV_PROJECT_ENVIRONMENT/bin:$PATH"
export UV_CACHE_DIR=${UV_CACHE_DIR:-/raid/jeff/uv-cache}
export HF_HOME=${HF_HOME:-/raid/jeff/huggingface}
export TRITON_CACHE_DIR=/raid/jeff/prompt-rewrite-cache/triton
export TORCHINDUCTOR_CACHE_DIR=/raid/jeff/prompt-rewrite-cache/inductor
export CUDA_CACHE_PATH=/raid/jeff/prompt-rewrite-cache/cuda
export NCCL_SOCKET_IFNAME=bond0
export GLOO_SOCKET_IFNAME=bond0
export NCCL_IB_HCA='=roce_rail1:1'
export NCCL_CROSS_NIC=0
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

: "${MODEL_PATH:?Set MODEL_PATH to the staged immutable base-model snapshot}"
RUN_DIR=${RUN_DIR:-outputs/prompt_rewrite/qwen38-sft-b64-l4096}
mkdir -p "$RUN_DIR"
nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu \
    --format=csv -l 10 > "$RUN_DIR/gpus.csv" &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT

uv run --frozen --extra fsdp python -m examples.prompt_rewrite.sft \
    --run-dir "$RUN_DIR" --nproc-per-node 8 --eval-before-training --execute \
    -- "model.path=$MODEL_PATH" "trainer.total_epochs=${TOTAL_EPOCHS:-3}" "$@"
