import torch
import numpy as np

from scipy.spatial.transform import Rotation as R

def generate_data_from_random(num_batches:int, points_each_cloud:int, sigma:float, 
                    rot_format:str, norm:bool, max_angle:int, one_source:bool, 
                    uniform_rand=False,dtype=torch.double) -> torch.Tensor:
    """
    generate batches of random rotation from scipy; 
    generate batches of source point clouds from random;
    final clouds are generated by rotate source clouds respectively 
    then add noises
    """
    angle_list = R.random(num_batches).as_euler(rot_format, degrees=True)
    scale_factor = 1
    if max_angle != 180:
        scale_factor = max_angle/180
    angle_list = angle_list*scale_factor
    rot = R.from_euler(rot_format, angle_list, degrees=True)
    rot_mat, rot_quat = rot.as_matrix(), rot.as_quat()
    rot_mat_tensor = torch.from_numpy(rot_mat)
    
    if one_source:
        if uniform_rand:
            source_cloud = 2*torch.rand(1, 3, points_each_cloud) - 1 #rescale from [0,1] to [-1,1]
        else:
            source_cloud = torch.randn(1, 3, points_each_cloud, dtype=dtype).expand(num_batches, -1, -1)
    else:
        if uniform_rand:
            source_cloud = 2*torch.rand(num_batches, 3, points_each_cloud, dtype=dtype) - 1
        else:
            source_cloud = torch.randn(num_batches, 3, points_each_cloud, dtype=dtype)
    if norm:
        source_cloud = source_cloud/source_cloud.norm(dim=1,keepdim=True)
    rotate_cloud = torch.matmul(rot_mat_tensor, source_cloud) #y = R(q)*x

    noise = sigma*torch.randn_like(source_cloud)
    target_cloud = rotate_cloud + noise

    return rot_quat, source_cloud, target_cloud

def generate_batches(num_batches:int, points_each_cloud:int, 
                    sigma:float, rot_format:str, norm:bool, max_angle:int, 
                    one_source:bool, uniform=False, dtype=torch.double) -> torch.Tensor:
    """
    concatenate source&target cloud for input data; 
    quat as ground truth
    """
    quat_, source_, target_ = generate_data_from_random(num_batches,points_each_cloud,
                                sigma, rot_format, norm, max_angle, one_source, uniform_rand)
    

    concatenate_cloud = torch.empty(num_batches,2,points_each_cloud,3,dtype=dtype)

    concatenate_cloud[:,0,:,:] = source_.transpose(1,2)
    concatenate_cloud[:,1,:,:] = target_.transpose(1,2)

    return quat_, concatenate_cloud

def generate_random_quat() -> np.ndarray:
    """this code generates a random quaternion
    NOTE: this is actually the correct way to do a uniform random rotation in SO3
    """
    
    quat = np.random.uniform(-1, 1, 4)  # note this is a half-open interval, so 1 is not included but -1 is
    norm = np.sqrt(np.sum(quat**2))

    while not (0.2 <= norm <= 1.0):
        quat = np.random.uniform(-1, 1, 4)
        norm = np.sqrt(np.sum(quat**2))
    
    quat = quat / norm
    return quat