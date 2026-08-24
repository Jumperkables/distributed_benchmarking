# standard imports

# 3rd party imports
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# local imports


class SyntheticDataset(Dataset):
    def __init__(self, size, dim, label_size):
        self.size = size
        self.dim = dim
        self.data = torch.randn(size, dim)
        self.label_size = label_size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx], torch.ones(self.label_size, dtype=torch.float32)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)