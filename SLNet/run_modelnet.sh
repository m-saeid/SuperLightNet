#!/bin/bash

N=1024                          # 1024
EMBED_Modelnet='adaptive' # ('adaptive' 'gaussian' 'cosine')                # ('gaussian' 'grouped_conv_mlp' 'cosine' 'mlp')   # mlp
EMBED_Scanobject='adaptive'
INITIAL_DIM=3                   # 3
EMBED_DIM=16 # 32 64) # (16 32) # 64)            # 16
ALPHA_BETA='yes' # 'no')         # no
SIGMA=0.4            # 0.4
ALPHA=1000.0          # 100.0
BETA=100.0             # 1.0

RES_DIM_RATIO=0.25
BIAS=false    #####################                  # Flase
USE_XYZ=true  #####################                  # True
NORM_MODE='anchor'         # ('nearest_to_mean' 'anchor' 'center')
STD_MODE='BN11'            # ('BN1D' 'BN11' '1111' 'B111')   # 'B111'          ######## 2

DIM_RATIO='2-2-2-1'                # '2-2-2-1'

NUM_BLOCKS1='1-1-2-1'                # '1-1-2-1'
TRANSFER_MODE='mlp-mlp-mlp-mlp'         # 'mlp-mlp-mlp-mlp'
BLOCK1_MODE='mlp-mlp-mlp-mlp'           # 'mlp-mlp-mlp-mlp'

NUM_BLOCKS2='1-1-2-1'        # '1-1-2-1'
BLOCK2_MODE='mlp-mlp-mlp-mlp'

K_ModelNet='32-32-32-32' # '64-64-64-64')      # '24-24-24-24'  ########### 3
K_ScanObject='24-24-24-24' # '32-32-32-32'
SAMPLING_MODE='fps-fps-fps-fps'       # 'fps-fps-fps-fps'
SAMPLING_RATIO='2-2-2-2'                # '2-2-2-2'

CLASSIFIER_MODE='mlp_very_large'  # 'mlp_very_very_large' 'mlp_very_large' 'mlp_large' 'mlp_medium' 'mlp_small' 'mlp_very_small'

BATCH_SIZE_ModelNet=50
BATCH_SIZE_ScanObject=50

EPOCH_ModelNet=300
EPOCH_ScanObject=200

LEARNING_RATE_ModelNet=0.1
LEARNING_RATE_ScanObject=0.01

MIN_LR_ModelNet=0.005
MIN_LR_ScanObject=0.005

WEIGHT_DECAY_ModelNet=2e-4
WEIGHT_DECAY_ScanObject=1e-4

#SEED=42
WORKERS_ModelNet=6
WORKERS_ScanObject=6
EMA='yes'

# SIGMA=(0.3 0.35 0.2 0.25 0.27 0.33 0.37 0.4 0.45 0.5 0.22 0.23 0.24 0.26 0.28 0.32 0.36 0.42 0.47 0.55 0.6 0.7 0.8 0.9 1.0)
SEED=(10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200 210 220 230 240 250 260 270 280 290 300)
#EMA=('yes' 'no')

for i in {1..100}; do
#    for EMBED_DIM in "${EMBED_DIM[@]}"; do
#        for RES_DIM_RATIO in "${RES_DIM_RATIO[@]}"; do
            for SEED in "${SEED[@]}"; do
                #for EMA in "${EMA[@]}"; do
                    python cls_modelnet.py --n "$N" --embed "$EMBED_Modelnet" --initial_dim "$INITIAL_DIM" --embed_dim "$EMBED_DIM" \
                    --res_dim_ratio "$RES_DIM_RATIO" --norm_mode "$NORM_MODE" --std_mode "$STD_MODE" --sigma "$SIGMA" \
                    --dim_ratio "$DIM_RATIO" --num_blocks1 "$NUM_BLOCKS2" --transfer_mode "$TRANSFER_MODE" \
                    --block1_mode "$BLOCK1_MODE" --num_blocks2 "$NUM_BLOCKS2" --block2_mode "$BLOCK2_MODE" --k_neighbors "$K_ModelNet" \
                    --sampling_mode "$SAMPLING_MODE" --sampling_ratio "$SAMPLING_RATIO" --classifier_mode "$CLASSIFIER_MODE" \
                    --batch_size "$BATCH_SIZE_ModelNet" --epoch "$EPOCH_ModelNet" --learning_rate "$LEARNING_RATE_ModelNet" --min_lr "$MIN_LR_ModelNet" \
                    --weight_decay "$WEIGHT_DECAY_ModelNet" --seed "$SEED" --workers "$WORKERS_ModelNet" \
                    --alpha_beta "$ALPHA_BETA" --ema "$EMA"
                    # || { echo "Error in cls_modelnet.py"; exit 1; }
                    echo "====================================================================="
                    # --use_xyz True --bias False
                #done
            done
#        done
#    done
done

