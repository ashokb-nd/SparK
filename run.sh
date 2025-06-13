echo "## start ##"
python pretrain/download_data.py
echo "running spark job..."
torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 \
    pretrain/main.py \
    --exp_name=spark \
    --exp_dir=logdir/ \
    --model=convnext_tiny \
    --input_size=224 \
    --bs=200 
    # --init_weight=/ashok/SparK/logdir/old_checkpoint.pth