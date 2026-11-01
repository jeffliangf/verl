#!/usr/bin/env bash
set -euo pipefail
source /home/jeff/rl-benchmarks/inspection-v1-1-group8-16h100-20261101/grpo/cluster-env.sh
rl_mode=${1:-preview}
rl_args=()
if [[ "$rl_mode" == execute ]]; then rl_args=(--execute); fi
uv run --no-sync python -m examples.video_agents.inspection_rl train \
  --run-dir "$RL_RUN_DIR" --data-dir "$RL_DATA_DIR" "${rl_args[@]}" -- \
  "actor_rollout_ref.model.path=${RL_MODEL_PATH:-$RL_MODEL_SOURCE}" \
  trainer.nnodes=2 trainer.n_gpus_per_node=8 \
  video_rl.h3_max_inflight=20 video_rl.timeouts.h3=150 \
  data.train_batch_size=32 data.seed=42 \
  actor_rollout_ref.actor.ppo_mini_batch_size=32 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.fsdp_config.use_torch_compile=false \
  actor_rollout_ref.ref.fsdp_config.use_torch_compile=false \
  actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
  actor_rollout_ref.rollout.n=8 video_rl.judge_concurrency=8 video_rl.judge_streaming=true actor_rollout_ref.rollout.agent.num_workers=8 \
  'actor_rollout_ref.actor.checkpoint.save_contents=[model,optimizer,extra]' \
  trainer.total_training_steps=10000 trainer.total_epochs=100 \
  trainer.val_before_train=true trainer.val_only=false \
  trainer.save_freq=5 trainer.test_freq=25 trainer.max_actor_ckpt_to_keep=null \
  trainer.checkpoint_callback_class=run_callback.ExperimentCheckpoint \
  'trainer.logger=[console,tensorboard,file]' \
  trainer.experiment_name=inspection-v1-1-group8-16h100-20261101 \
  trainer.resume_mode=disable trainer.resume_from_path=null trainer.del_local_ckpt_after_load=false \
  ray_kwargs.ray_init.num_cpus=null ++ray_kwargs.ray_init.address=auto \
  "ray_kwargs.ray_init.runtime_env.py_executable=$UV_PROJECT_ENVIRONMENT/bin/python"
