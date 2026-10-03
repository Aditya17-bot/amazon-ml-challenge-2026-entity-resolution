#!/bin/bash
# parallel test: all new features + expected-F on xenc2 -> out_group11
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
cd /c/adi/amazon/student_resource/kaggle
(cd k_group2b && kaggle kernels push -p . 2>&1 | tail -1)
sleep 60
bash wait_kernel.sh amazon-er-group2b && bash fetch.sh amazon-er-group2b out_group11
echo CHAIN11 DONE
