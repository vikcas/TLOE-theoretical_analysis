#!/usr/bin/env bash
# Reproduces every number in the report.
set -euo pipefail
cd "$(dirname "$0")/exact"
for f in verify_tloe.py verify_independence_and_kl.py kl_bridge.py identifiability.py eigen_check.py extra_checks.py \
         weber_conditional_check.py positive_control_erlang.py stage_controls.py autoscaler_table.py \
         autoscaler_sufficiency.py autoscaler_lumped_db.py compare_metrics.py cross_scale_chain_rule.py ps_vs_fcfs_erlang.py blocking_minimal_statistic.py fig_evidence.py; do
  echo "=== $f"; python3 "$f"
done
cd ../finite_sample
echo "=== tr_signal_strength.py"; python3 tr_signal_strength.py
echo "=== tf_power.py"; python3 tf_power.py
echo "=== tf_power_A.py"; python3 tf_power_A.py
for m in J B3; do echo "=== study.py $m"; python3 study.py "$m" 5 1; done
for m in A B2; do echo "=== study.py $m"; python3 study.py "$m" 5 0; done
echo "=== summarise.py"; python3 summarise.py
