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
    device = torch.device(f"cuda:{local_rank}")

    # Init process
    dist.init_process_group(
        backend="nccl",
        device_id=device,
    )
    rprint(f"NCCL initialized")
    x = torch.tensor([float(rank + 1)], device=device)
    rprint(f"before all_reduce: {x.item()}")
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    rprint(f"after all_reduce: {x.item()}")
    expected = 3.0
    if x.item() != expected:
        raise RuntimeError(RHEADER+f"expected {expected}, got {x.item()}")
    rprint(f"SUCCESS")
    dist.barrier(device_ids=[local_rank])
    rprint(f"Destroying process group")
    dist.destroy_process_group()
    rprint(f"Done")



if __name__ == "__main__":
    main()