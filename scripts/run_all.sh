#!/usr/bin/env bash
# 5 task × 2 seed × {CFM, DP}: 每 task 内先 CFM 双 seed 并行 (cuda:0/1),
# wait, 再 DP 双 seed 并行, wait. task 之间串行. 共 20 run, 10 批次.
# task list 来源 notes/decisions.md 2026-05-12.
# plan/c2_plan.md §5,§8 写 3 seed; 此处用 2 seed (双卡天然配对, 不留空闲).

set -u

cd ~/shared-nvme/FlowPolicy-Libero
git pull
mkdir -p logs runs

TASKS=(
  open_the_top_drawer_and_put_the_bowl_inside
  push_the_plate_to_the_front_of_the_stove
  turn_on_the_stove
  put_the_cream_cheese_in_the_bowl
  put_the_wine_bottle_on_top_of_the_cabinet
)

for task in "${TASKS[@]}"; do
  for method in cfm dp; do
    echo "=== $(date +%H:%M:%S)  Task: $task  Method: $method ==="
    for i in 0 1; do
      seed=$((42 + i))
      python -m src.train \
        --config "configs/${method}_default.yaml" \
        --task_name "$task" \
        --seed $seed \
        --device cuda:$i \
        --output_dir "runs/${method}_${task}_seed${seed}" \
        > "logs/${method}_${task}_seed${seed}.log" 2>&1 &
    done
    wait
    echo "=== $(date +%H:%M:%S)  Done: $task / $method ==="
  done
done

echo "=== $(date +%H:%M:%S)  ALL DONE ==="
