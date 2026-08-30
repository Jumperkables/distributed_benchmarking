# standard imports
import time

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

# Globals
#torch.manual_seed(42)
HOST, RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT = get_dist_env_info()
DEVICE = torch.device(f"cuda:{LOCAL_RANK}")
torch.cuda.set_device(DEVICE)




def main():
    print_dist_env_info()
    rprint("Setting up process group...")
    dist.init_process_group(backend="nccl", device_id=DEVICE)

    # Elastic launch - Fault tolerance

    dist.destroy_process_group()



if __name__ == "__main__":
    main()