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
HOST, RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT = get_dist_env_info()
DEVICE = torch.device(f"cuda:{LOCAL_RANK}")
torch.cuda.set_device(DEVICE)


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
    dist.barrier(device_ids=[LOCAL_RANK])
    success = torch.all(x == 1)
    rprint(f"BROADCAST {success}: Expected all values in tensor == 1")


def all_to_all():
    inputs = torch.arange(2, device=DEVICE) + RANK * 2
    inputs = list(inputs.chunk(2))
    outputs = list(torch.empty([2], device=DEVICE, dtype=torch.int64).chunk(2))
    dist.all_to_all(outputs, inputs)
    rprint(f"ALL_TO_ALL: Expected a permuted version of the input tensors{outputs}")


def scatter():
    rank_s = 0  # Scatter from rank 0 as the source
    if RANK == rank_s:
        to_scatter = [torch.ones(10, device=DEVICE)*1, torch.ones(10, device=DEVICE)*2]
    else:
        to_scatter = None   # Must be None on the non-source ranks i believe
    output_tensor = torch.zeros(10, device=DEVICE)
    dist.scatter(output_tensor, to_scatter, src=rank_s)
    success = torch.all(output_tensor == RANK+1)
    rprint(f"SCATTER {success}: Expected all values in tensor == RANK+1, here {output_tensor[0]}")



def reduce_scatter():
    output = torch.zeros(4, device=DEVICE, dtype=torch.float32)
    input_list = [
        torch.arange(4, device=DEVICE, dtype=torch.float32)    * (10**RANK),
        torch.arange(4, 8, device=DEVICE, dtype=torch.float32) * (10**RANK),
    ]
    rprint(input_list)
    dist.reduce_scatter(output, input_list)
    rprint(f"REDUCE_SCATTER: {output}")



def all_reduce():
    tensor = torch.ones(10, device=DEVICE) * (RANK+1)
    dist.all_reduce(tensor, op=dist.ReduceOp.PRODUCT)
    rprint(f"ALL_REDUCE: Expecting the same tensor on EACH node filled with values {WORLD_SIZE*(WORLD_SIZE+1)/2} {tensor}")

    premul_sum_factor = 0.1
    tensor = torch.ones(10, device=DEVICE) * (RANK+1)
    dist.all_reduce(tensor, op=dist.ReduceOp.PREMUL_SUM(premul_sum_factor))
    rprint(f"ALL_REDUCE: -PREMUL_SUM check- Expecting the same tensor on EACH node filled with values { premul_sum_factor* WORLD_SIZE * (WORLD_SIZE + 1) / 2} {tensor}")



def reduce():
    rank_r = 0
    tensor = torch.ones(10, device=DEVICE) * (RANK+1)
    dist.reduce(tensor, dst=rank_r, op=dist.ReduceOp.SUM)
    rprint(f"REDUCE: Expecting a tensor filled with {WORLD_SIZE*(WORLD_SIZE+1)/2} on dst node {rank_r}: {tensor}")


def all_gather():
    # Try out the coalesced and object variants
    # I think all nodes get the gather output
    rank_g = 1
    tensor_list = [torch.zeros(1, device=DEVICE), torch.zeros(2, device=DEVICE)]
    tensor = torch.ones(RANK+1, device=DEVICE)*(RANK+1)
    dist.all_gather(tensor_list, tensor)
    rprint(f"ALL_GATHER: Expect each rank to have the same gathered list: {tensor_list}")


def all_gather_single():
    rank_g = 0
    size_t = 3
    output_tensor_stacked = torch.zeros((WORLD_SIZE, size_t), device=DEVICE)
    output_tensor_concat = torch.zeros(WORLD_SIZE * size_t, device=DEVICE)
    input_tensor = torch.ones(size_t, device=DEVICE) * (RANK+1)
    dist.all_gather_single(output_tensor_stacked, input_tensor)
    dist.all_gather_single(output_tensor_concat, input_tensor)
    rprint("ALL_GATHER_SINGLE: Stacked format", output_tensor_stacked)
    rprint("ALL_GATHER_SINGLE: Concat format", output_tensor_concat)


def all_gather_coalesced():
    # Varying sized tensors are allowed here
    output_tensor_list = [
        [
            torch.tensor([-1, -1], device=DEVICE),
            torch.tensor([-1], device=DEVICE),
            torch.tensor([-1, -1, -1], device=DEVICE)
        ] for _ in range(WORLD_SIZE)
    ]
    input_tensor_list = [
        torch.ones(2, device=DEVICE) * (RANK+1),
        torch.ones(1, device=DEVICE) * (RANK+1),
        torch.ones(3, device=DEVICE) * (RANK+1),
    ]
    dist.all_gather_coalesced(output_tensor_list, input_tensor_list)
    rprint(f"ALL_GATHER_COALESCED: Deprecated, but should expect a coalesced form of tensors from both ranks {output_tensor_list}")


def all_gather_object():
    # This example i borrow from docs, mine wasnt working
    gather_objects = [None for _ in range(WORLD_SIZE)]  # any picklable object
    gather_objects[RANK] = f"Hello Edd boy {RANK}"
    output = [None for _ in gather_objects]
    dist.all_gather_object(output, gather_objects[dist.get_rank()])
    rprint(f"ALL_GATHER_OBJECT: Should have dictionaries with rank aligned numbers in {output}")
    #torch.distributed.collective_utils.all_gather_object_enforce_type()


def gather():
    # An inverse to scatter in a sense
    rank_g = 0  # destination rank for gather
    size_t = 2
    if RANK == rank_g:
        gather_list = [torch.zeros(size_t, device=DEVICE) for _ in range(size_t)]
    else:
        gather_list = None
    to_gather = (torch.ones(size_t, device=DEVICE) * RANK)
    dist.gather(to_gather, gather_list, dst=rank_g)
    rprint(f"GATHER: Expected gather rank to contain list of tensors, and everything else as None: {gather_list}")

####################
# P2P
def send_and_recv():
    rank_src = 0
    rank_dst = 1
    tensor = torch.ones(10, device=DEVICE) * (RANK+1)
    rprint(f"SEND_AND_RECV: Before send from {rank_src} to {rank_dst}: {tensor}")
    if RANK == rank_src:
        dist.send(tensor, rank_dst)
    if RANK == rank_dst:
        dist.recv(tensor, rank_src)
    rprint(f"SEND_RECV: After send from {rank_src} to {rank_dst}: {tensor}")



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
    scatter()
    gather()
    all_gather()
    all_gather_single()
    #all_gather_coalesced() #<- keeping this cos theres a maybe typo in the docs i can PR =-) so i'll remember where it is
    all_gather_object()
    reduce()
    all_reduce()
    reduce_scatter()
    all_to_all()


    ####################
    # P2P
    send_and_recv()


    ####################
    # Blocking vs non-blocking


    ####################
    # Cleanup
    dist.destroy_process_group()



if __name__ == "__main__":
    main()