#!/bin/bash
# 3-encoder ensemble (x2,x3,x4) + all tested features -> out_group13 (US/India candidate for per-country merge)
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
cd /c/adi/amazon/student_resource/kaggle
(cd k_group3 && kaggle kernels push -p . 2>&1 | tail -1)
sleep 60
bash wait_kernel.sh amazon-er-group3 && bash fetch.sh amazon-er-group3 out_group13
echo CHAIN13 DONE
