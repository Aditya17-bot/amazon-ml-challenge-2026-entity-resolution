#!/bin/bash
# early rebuild result: group on xenc5 only -> out_group12
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
cd /c/adi/amazon/student_resource/kaggle
true
(cd k_group5 && kaggle kernels push -p . 2>&1 | tail -1)
sleep 60
bash wait_kernel.sh amazon-er-group5 && bash fetch.sh amazon-er-group5 out_group12
echo CHAIN8 DONE
