#!/bin/bash
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
cd /c/adi/amazon/student_resource/kaggle
(cd k_group6 && kaggle kernels push -p . 2>&1 | tail -1)
sleep 60
bash wait_kernel.sh amazon-er-group6 && bash fetch.sh amazon-er-group6 out_group14
echo CHAIN14 DONE
