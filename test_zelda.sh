export CUDA_VISIBLE_DEVICES=0
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_CPP_LOG_LEVEL=INFO
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_SOCKET_IFNAME='=enp15s0'
export NCCL_IB_DISABLE=1
torchrun \
    --nnodes=2 \
    --nproc-per-node=1 \
    --node-rank=0 \
    --rdzv-id=nccl-minimal-01 \
    --rdzv-endpoint=192.168.1.124:29501 \
    test_connection.py