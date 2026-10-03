#!/bin/bash
# smoke test of new group.py on xenc2 scores (group2 kernel, new version) -> out_group10
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
cd /c/adi/amazon/student_resource/kaggle
(cd k_group2 && kaggle kernels push -p . 2>&1 | tail -1)
sleep 60
bash wait_kernel.sh amazon-er-group2 && bash fetch.sh amazon-er-group2 out_group10
echo CHAIN7 DONE
