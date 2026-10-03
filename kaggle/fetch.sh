#!/bin/bash
# usage: fetch.sh <slug> <dir> -- downloads kernel output, retrying on network errors
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
for i in 1 2 3 4 5 6; do
  rm -rf $2; kaggle kernels output adityasridhar077/$1 -p $2 >/dev/null 2>&1 && [ -f $2/log.txt ] && { echo "fetched $1 to $2"; exit 0; }
  sleep 60
done
echo "fetch failed $1"; exit 1
