#!/bin/bash
set -e
export LD_LIBRARY_PATH=/root/go/soft/env/cuda-9.0/lib64:/root/go/soft/cuda10-cudnn7.6.5.32/lib64:$LD_LIBRARY_PATH:/usr/lib64/:/usr/local/lib/
export CUDA_VISIBLE_DEVICES=7

fixed_teacher="127.0.0.1:19290,127.0.0.1:19291,127.0.0.1:19292,127.0.0.1:19293,127.0.0.1:19294,127.0.0.1:19295,127.0.0.1:19296,127.0.0.1:19297"

for w in {1..10}
do
    for T in {1..20} 
    do
        wf=$((echo scale=1 ; echo $w / 10 ) | bc )
        Tf=$((echo scale=1 ; echo $T ) | bc )
        python3.6 -u distill.py \
            --fixed_teacher $fixed_teacher \
            --opt=AdamW \
            --s_weight $wf \
            --train_range 10 \
            --LR 1e-4 \
            --kl 0 \
            --T $Tf \
            --epoch_num 20 > log/d_w${wf}_T${Tf}.log 2>&1
    done
done

exit 0

nohup python3.6 -u distill.py \
    --fixed_teacher $fixed_teacher \
    --s_weight 0.05 \
    --epoch_num 20 > d_2.log 2>&1 &

nohup python3.6 -u distill.py \
    --fixed_teacher $fixed_teacher \
    --opt=Adam \
    --LR=5e-5 \
    --s_weight 0.05 \
    --epoch_num 20 > d_3.log 2>&1 &
