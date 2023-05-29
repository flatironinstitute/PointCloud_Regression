import torch
import numpy as np
import torch.utils.data
import argparse
import matplotlib.pylab as plt

import regression.adj_util as A
import regression.metric as M
import util.optimal_svd as svd

def load_npz(file_path: str) -> torch.Tensor:
    with np.load(file_path) as data:
        cloud = torch.as_tensor(data["cloud"],dtype=torch.float32)
        quat  = torch.as_tensor(data["quat"],dtype=torch.float32)

    return cloud, quat

def get_svd_quat(cloud: torch.Tensor) -> torch.Tensor:
    """
    calculate list of quat from optimal SVD method
    return tensor for angle diff calculation
    """
    b, _, _, _ = cloud.shape
    quat_list = torch.empty(b, 4)
    for i in range(len(cloud)):
        curr_quat = torch.as_tensor(svd.direct_SVD(cloud[i]),dtype=torch.float32)
        quat_list[i] = curr_quat

    return quat_list

def get_batch_angle_diff(cloud: torch.Tensor, true_quat: torch.Tensor):
    #the input can be wrap up as a dict for different predicted quat
    svd_quat = get_svd_quat(cloud)
    svd_list = M.quat_angle_diff(svd_quat, true_quat, reduce=False)

    return svd_list

def generate_dict(svd_list:torch.Tensor):
    save_data = {}
    save_data["svd_list"] = svd_list.detach().numpy()
    return save_data

def main():
    parser = argparse.ArgumentParser(description = 'load cloud from npz and generate figure')
    parser.add_argument('npz_path', type = str, help = 'path of npz file')
    parser.add_argument('output_path', type = str, help = 'path of save figures and text')

    args = parser.parse_args()

    cloud_data  = args.npz_path
    save_path = args.output_path

    cloud, true_quat = load_npz(cloud_data)

    list_svd_diff = get_batch_angle_diff(cloud, true_quat)

    delta_q_dict = generate_dict(list_svd_diff)
    np.savez(save_path+"/svd_angle_diff.npz", **delta_q_dict)

if __name__ == '__main__':
    main()