torchrun --nnodes=1 --master_port=12345 --nproc_per_node=1 train.py --config configs/training/v1/image_finetune.yaml  --wandb
torchrun --nnodes=1 --master_port=12345 --nproc_per_node=1 train.py --config configs/training/v1/202506-512-8f.yaml  --wandb
