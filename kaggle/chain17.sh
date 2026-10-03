#!/bin/bash
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
cd /c/adi/amazon/student_resource/kaggle
(cd k_group8 && kaggle kernels push -p . 2>&1 | tail -1)
sleep 60
bash wait_kernel.sh amazon-er-group8 && bash fetch.sh amazon-er-group8 out_group17
echo CHAIN17 DONE
