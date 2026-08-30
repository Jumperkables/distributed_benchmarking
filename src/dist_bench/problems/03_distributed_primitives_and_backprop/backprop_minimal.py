# standard imports

# 3rd party imports
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

# local imports
from dist_bench.common.my_utils import (
    get_dist_env_info,
    print_dist_env_info,
    rprint,

    RHEADER
)

# Globals
torch.manual_seed(42)
HOST, RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT = get_dist_env_info()
DEVICE = torch.device(f"cuda:{LOCAL_RANK}")
torch.cuda.set_device(DEVICE)



class DatasetSmol(Dataset):
    def __init__(self, size_dset: int=10_000, size_nn_in: int=1, size_nn_out: int=1):
        self.size_dset = size_dset
        self.data = torch.randn((size_dset, size_nn_in), dtype=torch.float32)
        self.labels = torch.randn((size_dset, size_nn_out), dtype=torch.float32)


    def __len__(self):
        return self.size_dset


    def __getitem__(self, idx):
        return self.data[idx+(RANK*2)], self.labels[idx+(RANK*2)]   # Rank 0 and 1 processes being offset by one batch lets me compare my torch.dist reductions versus a doubling of batch size for correctness



class MLPSmol(nn.Module):
    def __init__(self):
        super(MLPSmol, self).__init__()
        self.linear1 = nn.Linear(1, 2)  # Tiny parameter count for gradient tracking
        self.linear2 = nn.Linear(2, 1)


    def forward(self, x):
        """
        I like using this kind of naming scheme when inspecting gradients by hand, this is not general PEP
        """
        x_l1 = self.linear1(x)
        x_l2_l1 = self.linear2(x_l1)
        return x_l2_l1



def main():
    print_dist_env_info()
    rprint("Setting up process group...")
    dist.init_process_group(backend="nccl", device_id=DEVICE)
    rprint("Setting up dataset and model object...")

    # Dataset
    dataset = DatasetSmol()
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    # Model
    model = MLPSmol()
    model.to(DEVICE)

    rprint("Iterating across examples...")
    for batch in dataloader:
        # Standard model processing
        inputs, labels = batch
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        outputs = model(inputs)
        loss = F.mse_loss(outputs, labels)

        # Calculate gradients
        loss.backward()

        # Use torch.dist to manage gradient communication across nodes
        ## Correct way to is to average .grad parameters across both
        if len(inputs) == 2:
            for name, param in model.named_parameters():
                before_dist = f"Node {RANK}: {name}.grad {param.grad}"
                dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
                after_dist = f"Node {RANK}: {name}.grad {param.grad}"
                rprint("Compare these values:")
        else:
            for name, param in model.named_parameters():
                bsz_4 = f"Should be equivalent to above:\n{name}.grad {param.grad}"
                print("Inspect")
        rprint("bsz = 4 on node 0 should be equivalent to bsz = 2 averaged across node 0 and 1")

    # Tidy everything up
    dist.destroy_process_group()



if __name__ == "__main__":
    main()