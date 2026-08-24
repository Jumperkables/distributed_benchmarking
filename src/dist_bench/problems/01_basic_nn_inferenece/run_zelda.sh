torchrun \
    --nnodes=2 \
    --nproc-per-node=2 \
    --node-rank=0 \
    --rdzv-id=mlsys-test-01 \
    --rdzv-backend=c10d \
    --rdzv-endpoint=192.168.1.124:29500 \
    ddp_minimal.py