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

# Global
HOST, RANK, LOCAL_RANK, WORLD_RANK, MASTER_ADDR, MASTER_PORT = get_dist_env_info()
DEVICE = torch.device(f"cuda:{LOCAL_RANK}")


####################
# Collectives
def broadcast():
    val = float(RANK)
    rank_b = 1
    if RANK == rank_b:
        x = torch.ones(1024, device=DEVICE)*val
    else:
        x = torch.ones(1024, device=DEVICE)*999
    dist.broadcast(x, src=rank_b)
    rprint(x)
    dist.barrier(device_ids=[LOCAL_RANK])
    success = torch.all(x == 1)
    rprint(f"BROADCAST {success}: Expected all values in tensor == 1")


def all_to_all():
    host, rank, local_rank, world_size, master_addr, master_port = get_dist_env_info()


def scatter():
    rank_s = 0  # Scatter from rank 0 as the source
    if RANK == rank_s:
        to_scatter = [torch.ones(10, device=DEVICE)*1, torch.ones(10, device=DEVICE)*2]
    else:
        to_scatter = None   # Must be None on the non-source ranks i believe
    output_tensor = torch.zeros(10, device=DEVICE)
    dist.scatter(output_tensor, to_scatter, src=rank_s)
    success = torch.all(output_tensor == RANK+1)
    rprint(f"Scatter {success}: Expected all values in tensor == RANK+1, here {output_tensor[0]}")



def reduce_scatter():
    host, rank, local_rank, world_size, master_addr, master_port = get_dist_env_info()


def all_reduce():
    host, rank, local_rank, world_size, master_addr, master_port = get_dist_env_info()


def reduce():
    host, rank, local_rank, world_size, master_addr, master_port = get_dist_env_info()


def all_gather():
    host, rank, local_rank, world_size, master_addr, master_port = get_dist_env_info()


def gather():
    host, rank, local_rank, world_size, master_addr, master_port = get_dist_env_info()


####################
# P2P
def send():
    host, rank, local_rank, world_size, master_addr, master_port = get_dist_env_info()


def recv():
    host, rank, local_rank, world_size, master_addr, master_port = get_dist_env_info()


####################
# Blocking vs non-blocking
def blocking_vs_nonblocking():
    host, rank, local_rank, world_size, master_addr, master_port = get_dist_env_info()



# Main
def main():
    # Print dist info
    print_dist_env_info()


    # Init process
    dist.init_process_group(
        backend="nccl",
        device_id=DEVICE,
    )

    ####################
    # Collectives
    broadcast()
    all_to_all()
    reduce_scatter()
    scatter()
    all_reduce()
    reduce()
    all_gather()
    gather()


    ####################
    # P2P
    send()
    recv()


    ####################
    # Blocking vs non-blocking
    blocking_vs_nonblocking()


    ####################
    # Cleanup
    dist.destroy_process_group()



if __name__ == "__main__":
    main()