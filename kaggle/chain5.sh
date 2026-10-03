#!/bin/bash
# rerun group2 (xenc2 scores) with France calibration, fetch to out_group5
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
cd /c/adi/amazon/student_resource/kaggle
(cd k_group2 && kaggle kernels push -p . 2>&1 | tail -1)
sleep 60
bash wait_kernel.sh amazon-er-group2 && bash fetch.sh amazon-er-group2 out_group5
echo CHAIN5 DONE
