import numpy as np
from typing import List, Tuple, Union, Dict
from torch.utils.data import Dataset, DataLoader, sampler
import cv2 as cv
import os
import glob
import torch
import logging

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
    def __init__(self, category:List[str], num_sample:int, pascal_path:str, syn_path:str, 
                 imagenet_path:str, resize:int,
                 sampling_weights:Dict[str,float]) -> None:
        """@args: base_path should be the top dir of Pascal data
        i.e. on rusty cluster it should be: 
        /mnt/home/clin/ceph/dataset/pascal_3d/PASCAL3D+_release1.0/
        """
        super().__init__()
        self.category = category # we must provide a (list of) category
        self.resize_shape = resize
        self.pascal_path = pascal_path
        self.imagenet_path = imagenet_path
        self.syn_path = syn_path

        self.num_sample = num_sample

        self.pascal_annos = []
        self.all_pascal = []
        for c in self.category:
            curr_anno_path = pascal_path + "Annotations/" + c + "_pascal/"
            curr_image_path = pascal_path + "Images/" + c + "_pascal/"
            self.pascal_annos += F.list_files_in_dir(curr_anno_path)
            self.all_pascal += F.list_files_in_dir(curr_image_path)

        self.imagenet_annos = []
        self.all_imagenet = []
        for c in self.category:
            curr_anno_path = imagenet_path + "Annotations/" + c + "_imagenet/"
            curr_image_path = imagenet_path + "Images/" + c + "_imagenet/"
            self.imagenet_annos += F.list_files_in_dir(curr_anno_path)
            self.all_imagenet += F.list_files_in_dir(curr_image_path)

        self.all_syn = []
        for c in self.category:
            curr_folder = self.syn_path + P.category_folderid(c)
            curr_subdirs = F.list_subdir_in_dir(curr_folder)

            for sub in curr_subdirs:
                curr_files = F.list_files_in_dir(sub)
                self.all_syn += curr_files

        # Apply sampling ratios to limit the number of examples used
        self.sampled_pascal = self.sample_files(self.all_pascal, sampling_weights['pascal'])
        self.sampled_imagenet = self.sample_files(self.all_imagenet, sampling_weights['imagenet'])
        self.sampled_synthetic = self.sample_files(self.all_syn, sampling_weights['synthetic'])

    def __len__(self):
        return len(self.sampled_pascal) + len(self.sampled_imagenet) + len(self.sampled_synthetic)

    def __getitem__(self, index:int) -> List[Tuple[torch.Tensor, Dict[str, Union[torch.Tensor, str]]]]:
        # subfolder's path contains info of Pascal's category, i.e.
        # "Images/aeroplane_pascal/2009_004203.jpg"
        # also for Sythetic ImageNet data, i.e.
        # "../syn_images_cropped_bkg_overlaid/02691156/..", then maps "02691156" to the category
        # random_pick = np.random.randint(len(self.all_annos))

        total_len = len(self.sampled_pascal) + len(self.sampled_imagenet) + len(self.sampled_synthetic)

        if not 0 <= index < total_len:
            raise IndexError(f"Requested index {index} is out of range in total")

        if index < len(self.sampled_pascal):
            return self.handle_pascal(index)
        elif index < len(self.sampled_pascal) + len(self.sampled_imagenet):
            return self.handle_imagenet(index - len(self.sampled_pascal))
        else:
            syn_index = index - (len(self.sampled_pascal) + len(self.sampled_imagenet))
            return self.handle_synthetic(syn_index)
        
    def sample_files(self, file_list:List[str], sampling_ratio:float) -> List[str]:
        sample_size = int(len(file_list)*sampling_ratio)
        return np.random.choice(file_list, sample_size, replace=False)

    def handle_pascal(self, index:int) -> List[Tuple[torch.Tensor, Dict[str, Union[torch.Tensor, str]]]]:
        data_list = []

        curr_file = self.sampled_pascal[index]
        file_name = os.path.basename(curr_file)    # This gets '2009_004203.jpg'

        curr_id = os.path.splitext(file_name)[0]  
        curr_category = os.path.basename(os.path.dirname(curr_file)) # extract the category from the abs path as "xxx_pascal"

        curr_annos = P.read_annotaions(self.pascal_path + "Annotations/" + curr_category + "/" + curr_id + ".mat")

        for anno in curr_annos:
            img_loader = P.RoILoaderPascal(self.category, curr_id,
                                           self.resize_shape, anno, 
                                           self.pascal_path + "Images/" + curr_category + "/",
                                           "pascal3d") 
            curr_img = img_loader()

            curr_dict = P.compose_euler_dict(anno)
            data_list.append((torch.as_tensor(curr_img, dtype=torch.float32), curr_dict))

        return data_list

    def handle_imagenet(self, index:int) -> List[Tuple[torch.Tensor, Dict[str, Union[torch.Tensor, str]]]]:
        data_list = []

        curr_file = self.sampled_imagenet[index]
        file_name = os.path.basename(curr_file)    # This gets '2009_004203.jpg'

        curr_id = os.path.splitext(file_name)[0]  
        curr_category = os.path.basename(os.path.dirname(curr_file)) # extract the category from the abs path as "xxx_pascal"

        curr_annos = P.read_annotaions(self.imagenet_path+ "Annotations/" + curr_category + "/" + curr_id + ".mat")

        for anno in curr_annos:
            img_loader = P.RoILoaderPascal(self.category, curr_id,
                                           self.resize_shape, anno, 
                                           self.imagenet_path + "Images/" + curr_category + "/",
                                           "imagenet") 
            curr_img = img_loader()

            curr_dict = P.compose_euler_dict(anno)
            data_list.append((torch.as_tensor(curr_img, dtype=torch.float32), curr_dict))

        return data_list

    def handle_synthetic(self, index:int) -> List[Tuple[torch.Tensor, Dict[str, Union[torch.Tensor, str]]]]:
        data_list = []
        
        image_path = self.sampled_synthetic[index]

        folder_path = os.path.dirname(image_path)  # gets the directory path

        # If you need the category folder specifically
        category_folder_path = os.path.dirname(folder_path)  # moves one level up to '02691156'
        category_folder_name = os.path.basename(category_folder_path)
        curr_category = P.folderid_category(category_folder_name)

        curr_dict = P.compose_syn_image_dict(image_path, curr_category)

        image_loader = P.RoILoader(self.resize_shape)
        curr_image = cv.imread(image_path)

        trans_image = image_loader.transform(curr_image)

        data_list.append((torch.as_tensor(trans_image, dtype=torch.float32), curr_dict))

        return data_list


    






            

    