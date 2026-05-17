#!/usr/bin/env bash
# Smoke test: 双卡同时跑 CFM (cuda:0) + DP (cuda:1) 各一遍, 5-10 分钟完成.
# 覆盖路径: dataset 构造 → phase 1 训练 → loss EMA plateau → phase 2 切换 →
#         training-time evaluate → set_init_state rollout → R3M.unwrap →
#         construct_eval_obs → ckpt save → 早停退出.
# 注意: 用前确认 run_all.sh 没在跑 (会抢同两张卡); 跑完检查 logs/smoke_*.log 末尾.

set -u

cd ~/shared-nvme/FlowPolicy-Libero
mkdir -p logs runs

echo "=== $(date +%H:%M:%S)  SMOKE TEST 启动 ==="

python -m src.train --config configs/cfm_smoke.yaml --device cuda:0 \
  > logs/smoke_cfm.log 2>&1 &
CFM_PID=$!

python -m src.train --config configs/dp_smoke.yaml --device cuda:1 \
  > logs/smoke_dp.log 2>&1 &
DP_PID=$!

wait $CFM_PID
CFM_EXIT=$?
wait $DP_PID
DP_EXIT=$?

echo "=== $(date +%H:%M:%S)  SMOKE 完成: CFM exit=$CFM_EXIT, DP exit=$DP_EXIT ==="

if [ $CFM_EXIT -eq 0 ] && [ $DP_EXIT -eq 0 ]; then
  echo "PASS: 两个 method smoke 都正常退出."
  echo "建议再 grep 验证 phase 2 触发 + best ckpt 写盘:"
  echo "  grep -E 'phase|best|save|evaluate' logs/smoke_cfm.log logs/smoke_dp.log | tail"
  exit 0
else
  echo "FAIL: 至少一个 method 异常退出. 查 log 找 Traceback:"
  echo "  grep -B 2 -A 20 'Traceback' logs/smoke_cfm.log logs/smoke_dp.log"
  exit 1
fi
