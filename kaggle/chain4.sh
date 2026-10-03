#!/bin/bash
# after chain3: push group3 with every encoder that completed, then fetch it
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
cd /c/adi/amazon/student_resource/kaggle
until grep -q "CHAIN3 DONE" chain3.log; do sleep 60; done
sufs="2"
[ -f output_x3/xenc_meta.json ] && sufs="$sufs,3"
[ -f output_x4/xenc_meta.json ] && sufs="$sufs,4"
echo "encoders: $sufs"
[ "$sufs" = "2" ] && { echo "no new encoder finished, not pushing"; exit 1; }
sed -i "s/^EXTRA_ENV = .*/EXTRA_ENV = {\"GROUP_XENC\": \"$sufs\"}/" k_group3/run.py
(cd k_group3 && kaggle kernels push -p . 2>&1 | tail -1)
sleep 60
bash wait_kernel.sh amazon-er-group3 && bash fetch.sh amazon-er-group3 out_group4
echo CHAIN4 DONE
