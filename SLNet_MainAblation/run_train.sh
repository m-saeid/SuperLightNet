N=1024                          # 1024
EMBED='mlp'                     # mlp
INITIAL_DIM=3                   # 3
EMBED_DIM=16                    # 16
RES_DIM_RATIO=(0.25 0.5 0.75)   # 0.25                   ######## 3
BIAS=0                          # 0 Flase
USE_XYZ=1                       # 1 True
NORM_MODE=('anchor' 'center')   # 'anchor'          ######## 2

DIM_RATIO='2-2-2-1'                   # '2-2-2-1'

NUM_BLOCKS1='1-1-2-1'                # '1-1-2-1'
TRANSFER_MODE='mlp-mlp-mlp-mlp'         # 'mlp-mlp-mlp-mlp'
BLOCK1_MODE='mlp-mlp-mlp-mlp'           # 'mlp-mlp-mlp-mlp'

NUM_BLOCKS2='1-1-2-1'        # '1-1-2-1'
BLOCK2_MODE='mlp-mlp-mlp-mlp'

K=('20-20-20-20' '24-24-24-24' '32-32-32-32') # '64-64-64-64')      # '24-24-24-24'  ########### 3
SAMPLING_MODE='fps-fps-fps-fps'       # 'fps-fps-fps-fps'
SAMPLING_RATIO='2-2-2-2'                # '2-2-2-2'

CLASSIFIER_MODE='mlp'

BATCH_SIZE=140      #modelnet40 and scanobject

EPOCH=300
LEARNING_RATE=0.1   # modelnet40
MIN_LR=0.005        # modelnet40
WEIGHT_DECAY=2e-4   # modelnet40
SEED=42
WORKERS=6           # modelnet40



for res_dim_ratio in "${RES_DIM_RATIO[@]}"; do
    for norm_mode in "${NORM_MODE[@]}"; do
        for k in "${K[@]}"; do

            echo "batch size: $BATCH_SIZE"
            python cls_modelnet.py --n "$N" --embed "$EMBED" --initial_dim "$INITIAL_DIM" --embed_dim "$EMBED_DIM" \
            --res_dim_ratio "$res_dim_ratio" --use_xyz True --norm_mode "$norm_mode" \
            --dim_ratio "$DIM_RATIO" --num_blocks1 "$NUM_BLOCKS1" --transfer_mode "$TRANSFER_MODE" \
            --block1_mode "$BLOCK1_MODE" --num_blocks2 "$NUM_BLOCKS2" --block2_mode "$BLOCK2_MODE" --k_neighbors "$k" \
            --sampling_mode "$SAMPLING_MODE" --sampling_ratio "$SAMPLING_RATIO" --classifier_mode "$CLASSIFIER_MODE" \
            --batch_size "$BATCH_SIZE" --epoch "$EPOCH" --learning_rate "$LEARNING_RATE" --min_lr "$MIN_LR" \
            --weight_decay "$WEIGHT_DECAY" --seed "$SEED" --workers "$WORKERS"
            #--checkpoint "$" --msg "$" --bias False
            echo "====================================================================="
            echo "====================================================================="
            echo "====================================================================="
        done
    done
done


for res_dim_ratio in "${RES_DIM_RATIO[@]}"; do
    for norm_mode in "${NORM_MODE[@]}"; do
        for k in "${K[@]}"; do

            python cls_scanobject.py --n "$N" --embed "$EMBED" --initial_dim "$INITIAL_DIM" --embed_dim "$EMBED_DIM" \
            --res_dim_ratio "$res_dim_ratio"  --use_xyz "$USE_XYZ" --norm_mode "$norm_mode" \
            --dim_ratio "$DIM_RATIO" --num_blocks1 "$NUM_BLOCKS1" --transfer_mode "$TRANSFER_MODE" \
            --block1_mode "$BLOCK1_MODE" --num_blocks2 "$NUM_BLOCKS2" --block2_mode "$BLOCK2_MODE" --k_neighbors "$k" \
            --sampling_mode "$SAMPLING_MODE" --sampling_ratio "$SAMPLING_RATIO" --classifier_mode "$CLASSIFIER_MODE" \
            --batch_size "$BATCH_SIZE" \
            --epoch 200 \
            --seed "$SEED"
            
            # --learning_rate "$LEARNING_RATE" --min_lr "$MIN_LR" \ --bias "$BIAS"
            # --weight_decay "$WEIGHT_DECAY" --seed "$SEED" --workers "$WORKERS"
            #--checkpoint "$" --msg "$"
            echo "====================================================================="
            echo "====================================================================="
            echo "====================================================================="

        done
    done
done








"""
    parser.add_argument('--n', type=int, default=1024, help='Point Number') 
    parser.add_argument('--embed', type=str, default='mlp', help='[mlp, tpe, gpe, pointhop]')
    parser.add_argument('--initial_dim', type=int, default=3, help='initial_dim')
    parser.add_argument('--embed_dim', type=int, default=16, help='embed_dim')
    parser.add_argument('--res_dim_ratio', type=int, default=0.25,
                        help='In residual blocks, the tensor dimension is changed by MLP in a certain ratio and returns to the original dimension.')
    parser.add_argument('--bias', type=bool, default=False, help='bias')
    parser.add_argument('--use_xyz', type=bool, default=True, help='Connecting initial 3D points to features')
    parser.add_argument('--norm_mode', type=str, default='anchor', help='[center, anchor]')
    
    parser.add_argument('--dim_ratio', type=str, default='2-2-2-1', help='dim_ratio')
    
    parser.add_argument('--num_blocks1', type=str, default='1-1-2-1', help='num_blocks1')
    parser.add_argument('--transfer_mode', type=str, default='mlp-mlp-mlp-mlp', help='transfer_mode')
    parser.add_argument('--block1_mode', type=str, default='mlp-mlp-mlp-mlp', help='block1_mode')
    
    parser.add_argument('--num_blocks2', type=str, default='1-1-2-1', help='num_blocks2')
    parser.add_argument('--block2_mode', type=str, default='mlp-mlp-mlp-mlp', help='block2_mode')

    parser.add_argument('--k_neighbors', type=str, default='24-24-24-24', help='k_neighbors')
    parser.add_argument('--sampling_mode', type=str, default='fps-fps-fps-fps', help='sampling_mode')
    parser.add_argument('--sampling_ratio', type=str, default='2-2-2-2', help='sampling_ratio')

    parser.add_argument('--classifier_mode', type=str, default='mlp', help='classifier_mode')

    parser.add_argument('-c', '--checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--msg', type=str, help='message after checkpoint')

    parser.add_argument('--batch_size', type=int, default=32, help='batch size in training')
    parser.add_argument('--epoch', default=300, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.1, type=float, help='learning rate in training')
    parser.add_argument('--min_lr', default=0.005, type=float, help='min lr')
    parser.add_argument('--weight_decay', type=float, default=2e-4, help='decay rate')
    parser.add_argument('--seed', type=int, help='random seed')
    parser.add_argument('--workers', default=6, type=int, help='workers')
"""



#python cls_modelnet.py --embed mlp --batch_size 50 --epoch 1
#python cls_scanobject.py --embed mlp --batch_size 50 --epoch 1

#python partseg_shapenet.py --embed mlp --batch_size 64 --epoch 1
#python semseg_s3dis.py --embed mlp --batch_size 64 --epoch 1














#python cls_modelnet.py --embed mlp --batch_size 50 --epoch 1
#python cls_modelnet.py --embed gaussian --batch_size 50 --epoch 1
#python cls_modelnet.py --embed fourier --batch_size 50 --epoch 1

#python cls_scanobject.py --embed mlp --batch_size 50 --epoch 1
#python cls_scanobject.py --embed gaussian --batch_size 50 --epoch 1
#python cls_scanobject.py --embed fourier --batch_size 50 --epoch 1

#python partseg_shapenet.py --embed mlp --batch_size 64 --epoch 1
#python partseg_shapenet.py --embed gaussian --batch_size 64 --epoch 1
#python partseg_shapenet.py --embed fourier --batch_size 64 --epoch 1


###################################################################################


#python semseg_s3dis.py --embed mlp --batch_size 24 --epoch 1



#python cls_modelnet.py --embed mlp --batch_size 50
#python cls_modelnet.py --embed tpe --embed_dim 18 --batch_size 50
#python cls_modelnet.py --embed gpe --batch_size 50
#python cls_modelnet.py --embed pointhop --batch_size 256

#python cls_scanobject.py --embed mlp --batch_size 50
#python cls_scanobject.py --embed tpe --embed_dim 18 --batch_size 50
#python cls_scanobject.py --embed gpe --batch_size 50
#python cls_scanobject.py --embed pointhop --batch_size 256

#python seg_shapenet.py --embed mlp --batch_size 128
#python seg_shapenet.py --embed tpe --embed_dim 36 --batch_size 64
#python seg_shapenet.py --embed gpe --batch_size 128