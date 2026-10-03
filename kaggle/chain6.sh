#!/bin/bash
# France rebuild: test3 -> predict3 -> xenc5 + xenc6 (parallel) -> group4
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
cd /c/adi/amazon/student_resource/kaggle
push() { (cd k_$1 && kaggle kernels push -p . 2>&1 | tail -1); }
bash wait_kernel.sh amazon-er-test3 || { echo "STOP test3 failed"; exit 1; }
push predict3; sleep 60
bash wait_kernel.sh amazon-er-predict3 || { echo "STOP predict3 failed"; exit 1; }
bash fetch.sh amazon-er-predict3 output_p3
push xenc5; push xenc6; sleep 60
(bash wait_kernel.sh amazon-er-xenc5 && bash fetch.sh amazon-er-xenc5 output_x5) &
(sleep 20; bash wait_kernel.sh amazon-er-xenc6 && bash fetch.sh amazon-er-xenc6 output_x6) &
wait
sufs=""
[ -f output_x5/xenc_meta.json ] && sufs="5"
[ -f output_x6/xenc_meta.json ] && sufs="${sufs:+$sufs,}6"
[ -z "$sufs" ] && { echo "STOP no encoder finished"; exit 1; }
echo "encoders: $sufs"
sed -i "s/^EXTRA_ENV = .*/EXTRA_ENV = {\"GROUP_XENC\": \"$sufs\"}/" k_group4/run.py
push group4; sleep 60
bash wait_kernel.sh amazon-er-group4 && bash fetch.sh amazon-er-group4 out_group6
echo CHAIN6 DONE
