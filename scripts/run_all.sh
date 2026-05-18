#!/usr/bin/env bash
# 5 task × 3 seed × {CFM, DP} = 30 run, 共 15 批.
# 每批 cuda:0=CFM cuda:1=DP 同 task 同 seed 并行, seed 与 task 串行.
# task list 来源 notes/decisions.md 2026-05-12.
# 3 seed 对齐 plan/c2_plan.md §5,§8 锁定值; 前版 commit 4a96902 的 2 seed 配置废弃.

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
  put_the_bowl_on_top_of_the_cabinet
  put_the_bowl_on_the_stove
  put_the_bowl_on_the_plate
)

SEEDS=(42 43 44)

# Resume 策略: output_dir 下有 last_{0,1}.pt -> 续训; 否则从头.
# 与 src/train.py --resume_dir 行为一致 (ckpt cfg 覆盖, 仅 yaml epoch_max / early_stop 可 override).
for task in "${TASKS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    echo "=== $(date +%H:%M:%S)  Task: $task  Seed: $seed ==="
    for i in 0 1; do
      if [ $i -eq 0 ]; then method=cfm; else method=dp; fi
      out_dir="runs/${method}_${task}_seed${seed}"
      if [ -f "$out_dir/last_0.pt" ] || [ -f "$out_dir/last_1.pt" ]; then
        resume_flag="--resume_dir $out_dir"
      else
        resume_flag=""
      fi
      python -m src.train \
        --config "configs/${method}_default.yaml" \
        --task_name "$task" \
        --seed $seed \
        --device cuda:$i \
        --output_dir "$out_dir" \
        $resume_flag \
        > "logs/${method}_${task}_seed${seed}.log" 2>&1 &
    done
    wait
    echo "=== $(date +%H:%M:%S)  Done: $task / seed $seed ==="
  done
done

echo "=== $(date +%H:%M:%S)  ALL DONE ==="
