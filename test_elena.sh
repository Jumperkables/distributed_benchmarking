export CUDA_VISIBLE_DEVICES=0
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_CPP_LOG_LEVEL=INFO
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
torchrun \
    --nnodes=2 \
    --nproc-per-node=1 \
    --node-rank=1 \
    --rdzv-id=nccl-minimal-01 \
    --rdzv-backend=c10d \
    --rdzv-endpoint=192.168.1.124:29500 \
    test_connection.py