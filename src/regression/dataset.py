import numpy as np
from torch.utils.data import Dataset, DataLoader, sampler
import os
import torch

class SimulatedDataset(Dataset):
    """
    Dataset to load simulated data which generated from random
    rotations.
    """
    def __init__(self, path: str):
        with np.load(path) as data:
            self.cloud = torch.as_tensor(data["cloud"],dtype=torch.float32)
            self.quat  = torch.as_tensor(data["quat"],dtype=torch.float32)

    def __len__(self):
        return len(self.cloud)

    def __getitem__(self, index: int):
        curr_cloud = self.cloud[index]#.view(-1), option for flatten before model
        curr_quat = self.quat[index]
        return curr_cloud, curr_quat

    