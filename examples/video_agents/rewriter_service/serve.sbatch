#!/bin/bash
#SBATCH -J rl-rewriter-s300-svc
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH -c 64
#SBATCH --mem=320G
#SBATCH -t UNLIMITED
#SBATCH -o /home/jeff/rl-benchmarks/rl-rewriter-s300-service/logs/serve-%N-%j.out
# One vLLM server (TP 4) plus the gateway on half a node (4 GPUs, 64 CPUs, 320 GB), with no time limit: partition
# `all` has MaxTime=UNLIMITED. Submit under umask 022 so the group can read the job log:
#   umask 022; sbatch -w <node> serve.sbatch
# The 2026-11-13..19 setup (two servers on a whole node):
#   INSTANCES=2 sbatch --exclusive --gres=gpu:8 --mem=0 -w <node> serve.sbatch
exec bash /home/jeff/rl-benchmarks/rl-rewriter-s300-service/serve.sh
