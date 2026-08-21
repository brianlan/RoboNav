#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CONFIG=$1
GPUS=$2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-$((20000 + RANDOM % 10000))}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.$((1 + RANDOM % 254))"}  # 1 - 254
export MKL_NUM_THREADS=$GPUS

PYTHONPATH="$REPO_ROOT":$PYTHONPATH \
torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    "$SCRIPT_DIR"/train.py \
    "$CONFIG" \
    --launcher pytorch \
    --skip-train-confirmation \
    ${@:3}
