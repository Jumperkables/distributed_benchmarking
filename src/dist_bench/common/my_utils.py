# standard imports
import os

# 3rd party imports
import torch
import torch.distributed as dist
import torch.nn as nn

# local imports

# Env variables
LOCAL_RANK = int(os.environ["LOCAL_RANK"])
RANK = int(os.environ["RANK"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])



def get_model_param_count(model: nn.Module, req_grad_only: bool = True) -> int:
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad or req_grad_only)
    rprint("Number of model parameters: ", num_params, f"req_grad_only: {req_grad_only}")
    return num_params



def rprint(*args, **kwargs):
    print(f"[Rank:{RANK}|Local rank:{LOCAL_RANK}|World size: {WORLD_SIZE}]", *args, **kwargs)



def setup_distributed():
    # Make sure nothing strange has happened to env variables
    if os.environ["RANK"] != str(RANK):
        raise ValueError(f"Rank has changed somehow: from {RANK} to {os.environ['RANK']}")
    if os.environ["LOCAL_RANK"] != str(LOCAL_RANK):
        raise ValueError(f"Local rank has changed somehow: from {LOCAL_RANK} to {os.environ['LOCAL_RANK']}")
    if os.environ["WORLD_SIZE"] != str(WORLD_SIZE):
        raise ValueError(f"World size has changed somehow: from {WORLD_SIZE} to {os.environ['WORLD_SIZE']}")

    # Setup
    torch.cuda.set_device(LOCAL_RANK)
    dist.init_process_group(backend="nccl", init_method="env://")
    device = torch.device("cuda", LOCAL_RANK)
    return RANK, LOCAL_RANK, WORLD_SIZE, device



def cleanup_distributed():
    dist.destroy_process_group()