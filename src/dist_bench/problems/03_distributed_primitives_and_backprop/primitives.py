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



def main():
    # Print dist info
    print_dist_env_info()
    host, rank, local_rank, world_size, master_addr, master_port = get_dist_env_info()

    # Set CUDA device
    torch.cuda.set_device(local_rank)

    # Init process
    dist.init_process_group(
        backend="nccl",
        device_id=local_rank,
    )
    rprint(f"NCCL initialized")
    x = torch.tensor([float(rank + 1)], device="cuda")
    rprint(f"before all_reduce: {x.item()}")
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    rprint(f"after all_reduce: {x.item()}")
    expected = 3.0
    if x.item() != expected:
        raise RuntimeError(RHEADER+f"expected {expected}, got {x.item()}")
    rprint(f"SUCCESS")
    dist.barrier(device_ids=[local_rank])
    rprint(f"Destroying process group", flush=True)
    dist.destroy_process_group()
    rprint(f"Done", flush=True)



if __name__ == "__main__":
    main()