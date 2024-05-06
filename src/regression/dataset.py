import numpy as np
from typing import List, Tuple, Union, Dict
from torch.utils.data import Dataset, DataLoader, sampler
import os
import glob
import torch
from simulator.quat_util import generate_random_quat
import regression.file_util as F
import regression.adj_util as A
from util.optimal_svd import direct_SVD
import util.pascal3d_annot as P

from scipy.spatial.transform import Rotation as R


class SimulatedDataset(Dataset):
    """
    Dataset to load simulated data which generated from random
    rotations.
    """
    def __init__(self, path: str, svd: bool):
        with np.load(path) as data:
            self.cloud = torch.as_tensor(data["cloud"], dtype=torch.float32)
            self.quat  = torch.as_tensor(data["quat"], dtype=torch.float32)
            self.get_svd = svd

    def __len__(self):
        return len(self.cloud)

    def __getitem__(self, index: int):
        curr_cloud = self.cloud[index] # .view(-1), option for flatten before model
        curr_quat = self.quat[index]

        if self.get_svd:
            return curr_cloud, torch.as_tensor(direct_SVD(curr_cloud), dtype=torch.float32)
        
        return curr_cloud, curr_quat


class ModelNetDataset(Dataset):
    """
    Dataset to load ModelNet40 mesh data
    """
    def __init__(self, base_path: str, category_list: list, num_sample: int, 
                sigma: float, num_rot: int, range_max: int, range_min: int):
        all_files = []
        for c in category_list:
            curr_path = "/".join([base_path, c, "train"])
            curr_list = F.list_files_in_dir(curr_path)
            all_files += curr_list

        self.select_files = []
        for f in all_files:
            curr_vert = F.read_off_file(f)
            if len(curr_vert) > range_max or len(curr_vert) < range_min:
                continue
            self.select_files.append(f)

        self.sigma = sigma
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


class Pascal3DDataset(Dataset):
    def __init__(self, category:List[str], num_sample:int, base_path:str, resize:int) -> None:
        """@args: base_path should be the top dir of Pascal data
        i.e. on rusty cluster it should be: 
        /mnt/home/clin/ceph/dataset/pascal_3d/PASCAL3D+_release1.0/
        """
        super().__init__()
        self.category = category # we must provide a (list of) category
        self.num_sample = num_sample
        self.resize_shape = resize
        self.base_path = base_path

        self.all_annos = []
        self.all_image = []
        for c in self.category:
            curr_anno_path = base_path + "Annotations/" + c + "_pascal/"
            curr_image_path = base_path + "Images/" + c + "_pascal/"
            self.all_annos += F.list_files_in_dir(curr_anno_path)
            self.all_image += F.list_files_in_dir(curr_image_path)

    def __len__(self):
        return self.num_sample

    def __getitem__(self, index) -> Tuple[torch.Tensor, Dict[str, Union[torch.Tensor, str]]]:
        random_pick = np.random.randint(len(self.all_annos))
        curr_file = self.all_annos[random_pick]
        curr_id = curr_file[-15:-4] # slice the id from the abs path
        curr_category = curr_file[len(self.base_path):-15] # slice the category from the abs path
        print("current category:", curr_category)

        img_loader = P.RoILoaderPascal(self.category, curr_id,
                                       self.resize_shape, 
                                       self.base_path+"Annotations"+curr_category, 
                                       self.base_path+"images"+curr_category)
        curr_img = img_loader()
        curr_anno = P.read_annotaions(self.anno_path + curr_id + ".mat")

        curr_dict = {"category":curr_anno["category"],
                     "a":A.deg_to_rad(torch.tensor(curr_anno["view"]["azimuth"])),
                     "e":A.deg_to_rad(torch.tensor(curr_anno["view"]["elevation"])),
                     "t":A.deg_to_rad(torch.tensor(curr_anno["view"]["theta"]))}

        return torch.as_tensor(curr_img, dtype=torch.float32), curr_dict

    
class KittiOdometryDataset(Dataset):
    """
    Dataset to load kitti' velodyne lidar data of odometry
    """
    def __init__(self, base_path:str, seq_num:int) -> None:
        super().__init__()
        self.velo_path = F.get_all_bins(base_path + seq_num)

    def __len__(self):
        return len(self.velo_path)

    def __getitem__(self, index: int):
        return F.get_velo(self.velo_path[index])






            

    