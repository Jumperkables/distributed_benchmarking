#export TORCH_DISTRIBUTED_DEBUG=DETAIL
#export TORCH_CPP_LOG_LEVEL=INFO
#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=ALL

USE_BOTH_NODES=true

if [ "$USE_BOTH_NODES" = false ]; then
    echo "Running intra-node setup on zelda only"
    NODES=1
    PROCS_PER_NODE=2
    NODE_RANK=0
    source ~/venvs/dist_bench_132/bin/activate
else
    echo "Running inter-node setup across zelda and elena"
    NODES=2
    PROCS_PER_NODE=1
    NODE_RANK=0
    source ~/venvs/dist_bench_126/bin/activate
fi

which python
export NCCL_SOCKET_IFNAME=enp15s0

torchrun \
    --nnodes="$NODES" \
    --nproc-per-node="$PROCS_PER_NODE" \
    --node-rank="$NODE_RANK" \
    --master-addr=192.168.1.124 \
    --master-port=29500 \
    primitives.py