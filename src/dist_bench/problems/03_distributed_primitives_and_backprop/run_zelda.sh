#export TORCH_DISTRIBUTED_DEBUG=DETAIL
#export TORCH_CPP_LOG_LEVEL=INFO
#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=ALL
USE_BOTH_NODES=false
if [ "$USE_BOTH_NODES" = false ] ; then
    echo "Running intra node setup on zelda only"
    export CUDA_VISIBLE_DEVICES=0
    NNODES=2
    NPROC_PER_NODE=1
    source ~/venvs/dist_bench_132/bin/activate
else
    echo "Running inter node setup across zelda and elena"
    NNODES=1
    NPROC_PER_NODE=2
    source ~/venvs/dist_bench_126/bin/activate
fi
which python
export NCCL_SOCKET_IFNAME=enp15s0

torchrun \
    --nnodes=$NNODES \
    --nproc-per-node=$NPROC_PER_NODE \
    --node-rank=0 \
    --master-addr=192.168.1.124 \
    --master-port=29501 \
    primitives.py