#!/bin/bash
# usage: wait_kernel.sh <slug>  -- polls until COMPLETE (exit 0) or ERROR/CANCEL (exit 1); network errors are retried
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
for i in $(seq 1 400); do
  a=$(kaggle kernels status adityasridhar077/$1 2>&1)
  echo "$(date +%H:%M) $1 $(echo "$a" | grep -o 'KernelWorkerStatus\.[A-Z_]*' || echo 'net?')"
  echo "$a" | grep -q "KernelWorkerStatus.COMPLETE" && exit 0
  echo "$a" | grep -qE "KernelWorkerStatus.(ERROR|CANCEL)" && exit 1
  sleep 120
done
exit 1
