import numpy as np
from typing import List
from torch.utils.data import Dataset, DataLoader, sampler
import os
import torch
from simulator.cloud_from_random import generate_random_quat
import regression.file_util as F
from scipy.spatial.transform import Rotation as R

class SimulatedDataset(Dataset):
    """
    Dataset to load simulated data which generated from random
    rotations.
    """
    def __init__(self, path: str):
        with np.load(path) as data:
            self.cloud = torch.as_tensor(data["cloud"], dtype=torch.float32)
            self.quat  = torch.as_tensor(data["quat"], dtype=torch.float32)

    def __len__(self):
        return len(self.cloud)

    def __getitem__(self, index: int):
        curr_cloud = self.cloud[index]#.view(-1), option for flatten before model
        curr_quat = self.quat[index]
        
        return curr_cloud, curr_quat


class ModelNetDataset(Dataset):
    """
    Dataset to load ModelNet40 mesh data
    """
    def __init__(self, base_path: str, category_list: list, num_sample: int, 
                sigma: float, num_rot: int):
        all_files = []
        for c in category_list:
            curr_path = "/".join([base_path, c, "train"])
            curr_list = F.list_files_in_dir(curr_path)
            all_files += curr_list

        self.select_files = []
        for f in all_files:
            curr_vert = F.read_off_file(f)
            if len(curr_vert) != num_sample:
                continue
            self.select_files.append(f)

        self.sigma =sigma
        self.num_sample = num_sample
        self.num_rot = num_rot

    def __len__(self):
        return self.num_rot

    def __getitem__(self, index: int):
        random_pick = np.random.randint(len(self.select_files))
        orig_cloud = torch.as_tensor(F.read_off_file(self.select_files[random_pick]), dtype=torch.float32)

        random_indices = torch.randperm(len(orig_cloud))
        num_points = int(self.num_sample)
        picked_indices = random_indices[:num_points]  
        source_cloud = orig_cloud[picked_indices]

        curr_rot = generate_random_quat()
        r = R.from_quat(curr_rot)
        rot_mat = r.as_matrix()
        rot_mat_tensor = torch.as_tensor(rot_mat, dtype=torch.float32)

        rotate_cloud = torch.matmul(source_cloud, rot_mat_tensor)
        noise = self.sigma*torch.randn_like(source_cloud)
        target_cloud = rotate_cloud + noise
        
        concatenate_cloud = torch.empty(2, num_points, 3, dtype=torch.float32)
        
        concatenate_cloud[0,:,:] = source_cloud
        concatenate_cloud[1,:,:] = target_cloud

        return concatenate_cloud, torch.as_tensor(r.as_quat(),dtype=torch.float32)






            

    