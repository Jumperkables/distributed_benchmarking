# standard imports
import os

# 3rd party imports
import torch
import torch.distributed as dist

# local imports
from dist_bench.common.my_utils import (
    get_dist_env_info,
    print_dist_env_info,
    rprint,

    RHEADER
)


####################
# Collectives
def broadcast():
    print("TODO")


def scatter():
    print("TODO")


def all_reduce():
    print("TODO")


def reduce():
    print("Todo")


def all_gather():
    print("TODO")


def gather():
    print("TODO")


####################
# P2P
def send():
    print("TODO")


def recv():
    print("TODO")


####################
# Blocking vs non-blocking
def blocking():
    print("TODO")


# Main
def main():
    # Print dist info
    print_dist_env_info()
    host, rank, local_rank, world_size, master_addr, master_port = get_dist_env_info()
    device = torch.device(f"cuda:{local_rank}")

    # Init process
    dist.init_process_group(
        backend="nccl",
        device_id=device,
    )

    #
    x = torch.tensor([float(rank + 1)], device=device)
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    expected = 3.0
    if x.item() != expected:
        raise RuntimeError(RHEADER+f"expected {expected}, got {x.item()}")
    dist.barrier(device_ids=[local_rank])
    dist.destroy_process_group()



if __name__ == "__main__":
    main()