cd /c/adi/amazon/student_resource
while tasklist | grep -q " 6792 "; do sleep 30; done
echo "upload process finished"
for i in $(seq 1 120); do s=$(kaggle datasets status adityasridhar077/amazon-er-data 2>&1); [ "$s" = "ready" ] && break; sleep 30; done
echo "data status: $s"
kaggle datasets files adityasridhar077/amazon-er-data 2>&1 | head -12
for m in train test; do (cd kaggle/k_$m && kaggle kernels push -p . 2>&1 | tail -2); done
echo PUSHED
