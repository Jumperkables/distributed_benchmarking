#export TORCH_DISTRIBUTED_DEBUG=DETAIL
#export TORCH_CPP_LOG_LEVEL=INFO
#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=ALL
export CUDA_VISIBLE_DEVICES=0
export NCCL_SOCKET_IFNAME=enp2s0
source ~/venvs/dist_bench_126/bin/activate
which python

torchrun \
    --nnodes=2 \
    --nproc-per-node=1 \
    --node-rank=1 \
    --master-addr=192.168.1.124 \
    --master-port=29500 \
    $1