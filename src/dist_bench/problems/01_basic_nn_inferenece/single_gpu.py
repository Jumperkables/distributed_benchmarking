# standard imports
import time

# 3rd party imports
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

# local imports
from shared import *
from dist_bench.common.my_utils import get_model_param_count

def main():
    # parameters
    input_dim = 1024
    hidden_dim = 2048
    output_dim = 10
    dataset_size = 10000
    batch_size = 128
    device = torch.device("cuda:0")

    # init
    dataset = SyntheticDataset(dataset_size, input_dim, label_size=output_dim)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    model = MLP(input_dim, hidden_dim, output_dim).to(device)
    optimizer = optim.Adam(model.parameters())
    get_model_param_count(model)

    # inference loop
    print(f"Running inference on {device}...")
    time_per_batch = []
    for b_idx, batch in enumerate(dataloader):
        # Start timing
        inputs, labels = batch
        optimizer.zero_grad()
        start_time = time.time()
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = F.mse_loss(outputs, labels)
        loss.backward()
        optimizer.step()

        # End timing
        end_time = time.time()
        ms = (end_time - start_time) * 1000
        print(f"b_idx: {b_idx} - {ms:.3f} ms")
        time_per_batch.append(ms)
    time_per_batch = torch.tensor(time_per_batch)
    print("Analyse time per batch here")


if __name__ == '__main__':
    main()