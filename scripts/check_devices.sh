#!/usr/bin/env bash
# 列出本机 CUDA device 索引、名字、显存与当前占用.
# 用途: 多 seed 并行启动前确认 cuda:0 / cuda:1 实际指向哪两张卡.
# 用法: bash scripts/check_devices.sh

set -eu

echo "=== nvidia-smi -L (索引 → 名字 / UUID) ==="
nvidia-smi -L
echo

echo "=== 显存与利用率 ==="
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
           --format=csv
echo

echo "=== 当前 GPU 上运行的进程 ==="
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
           --format=csv
