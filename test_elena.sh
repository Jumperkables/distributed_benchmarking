torchrun \
    --nnodes=2 \
    --nproc-per-node=1 \
    --node-rank=1 \
    --rdzv-id=nccl-minimal-01 \
    --rdzv-backend=c10d \
    --rdzv-endpoint=192.168.1.124:29500 \
    test_connection.py