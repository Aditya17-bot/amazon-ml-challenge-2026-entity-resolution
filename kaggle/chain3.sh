#!/bin/bash
# wait for xenc3 + xenc4, fetch both outputs
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
cd /c/adi/amazon/student_resource/kaggle
(bash wait_kernel.sh amazon-er-xenc3; bash fetch.sh amazon-er-xenc3 output_x3) &
(sleep 30; bash wait_kernel.sh amazon-er-xenc4; bash fetch.sh amazon-er-xenc4 output_x4) &
wait
echo CHAIN3 DONE
