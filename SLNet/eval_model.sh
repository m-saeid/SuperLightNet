
# Set which GPU(s) to use
export CUDA_VISIBLE_DEVICES=1  # Use GPU 1s

python eval_model.py --embd_dim 16
# python eval_model.py --embd_dim 32