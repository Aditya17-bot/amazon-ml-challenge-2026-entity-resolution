#!/bin/bash
# predict2 -> xenc2 -> group2, fetch each output
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
cd /c/adi/amazon/student_resource/kaggle
set -e
bash wait_kernel.sh amazon-er-predict2 && bash fetch.sh amazon-er-predict2 output_v7
(cd k_xenc2 && kaggle kernels push -p . 2>&1 | tail -1)
sleep 60
bash wait_kernel.sh amazon-er-xenc2 && bash fetch.sh amazon-er-xenc2 output_x2
(cd k_group2 && kaggle kernels push -p . 2>&1 | tail -1)
sleep 60
bash wait_kernel.sh amazon-er-group2 && bash fetch.sh amazon-er-group2 out_group3
echo CHAIN DONE
