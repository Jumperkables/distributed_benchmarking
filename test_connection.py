import os
import torch
import torch.distributed as dist


#####################
print("HOST:", os.uname().nodename, flush=True)
print("RANK:", os.environ.get("RANK"), flush=True)
print("LOCAL_RANK:", os.environ.get("LOCAL_RANK"), flush=True)
print("WORLD_SIZE:", os.environ.get("WORLD_SIZE"), flush=True)
print("MASTER_ADDR:", os.environ.get("MASTER_ADDR"), flush=True)
print("MASTER_PORT:", os.environ.get("MASTER_PORT"), flush=True)
#####################



rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])

print(
    f"[rank {rank}] starting: local_rank={local_rank}, "
    f"world_size={world_size}",
    flush=True,
)

torch.cuda.set_device(local_rank)

print(
    f"[rank {rank}] GPU: {torch.cuda.get_device_name(local_rank)}",
    flush=True,
)

print(f"[rank {rank}] initializing NCCL...", flush=True)

dist.init_process_group(
    backend="nccl",
    device_id=local_rank,
)


print(f"[rank {rank}] NCCL initialized", flush=True)

x = torch.tensor([float(rank + 1)], device="cuda")

print(f"[rank {rank}] before all_reduce: {x.item()}", flush=True)

dist.all_reduce(x, op=dist.ReduceOp.SUM)

print(f"[rank {rank}] after all_reduce: {x.item()}", flush=True)

expected = 3.0

if x.item() != expected:
    raise RuntimeError(
        f"rank {rank}: expected {expected}, got {x.item()}"
    )


print(f"[rank {rank}] SUCCESS", flush=True)

dist.barrier(device_ids=[local_rank])

print(f"[rank {rank}] destroying process group", flush=True)

dist.destroy_process_group()

print(f"[rank {rank}] done", flush=True)