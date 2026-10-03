#!/bin/bash
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
cd /c/adi/amazon/student_resource/kaggle
(cd k_group7 && kaggle kernels push -p . 2>&1 | tail -1)
sleep 60
bash wait_kernel.sh amazon-er-group7 && bash fetch.sh amazon-er-group7 out_group15
echo CHAIN15 DONE
