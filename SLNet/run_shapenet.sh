#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

EMBED_DIM=(16 32)
STD_MODE='BN1D' #('BN1D' 'B111' 'BN11' '1111')
EPOCH=350   # 400
BS=128
TBS=64

for i in {1..1}; do
    for EMBED_DIM in "${EMBED_DIM[@]}"; do
        python partseg_shapenet.py --embed_dim "$EMBED_DIM" --std_mode "$STD_MODE" --epochs "$EPOCH" --batch_size "$BS" --test_batch_size "$TBS" 
    done
done

