# ChatGPT generated demo file for me to have a lil play around with the visualize_sharing command i found in the docs
# standard imports

# 3rd party imports
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, Shard
from torch.distributed.tensor.debug import visualize_sharding

# local imports
from dist_bench.common.my_utils import rprint


def main():
    dist.init_process_group("nccl")

    rank = dist.get_rank()
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)

    # 2 GPUs -> a 1D tensor-parallel-style mesh
    mesh = init_device_mesh(
        "cuda",
        (2,),
        mesh_dim_names=("tp",),
    )

    # Every rank starts with the same logical tensor.
    x = torch.arange(16, device=device).reshape(4, 4)

    # Shard dimension 0 across the 2 ranks.
    x_sharded = distribute_tensor(
        x,
        mesh["tp"],
        placements=[Shard(0)],
    )

    #rprint(f"Rank {rank}")
    #rprint("Logical size:", x_sharded.shape)
    #rprint("Local tensor:", x_sharded.to_local())

    # The useful bit you're practicing with:
    visualize_sharding(x_sharded)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()