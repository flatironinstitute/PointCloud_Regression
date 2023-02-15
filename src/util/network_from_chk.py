import torch
import numpy as np
import pytorch_lightning as pl
import torch.utils.data
import argparse
import matplotlib.pylab as plt

import regression.trainer as tr
import regression.dataset as ds
import regression.config as cf
import regression.metric as M
import util.optimal_svd as svd

def read_check_point(path: str):
    model = tr.MLPTrainer.load_from_checkpoint(path)
    model.eval()
    return model

def forward_loaded_model(loaded_model, cloud: torch.Tensor) -> torch.Tensor:
    #cloud data are load and convert from numpy load
    print("shape of tensor: ", cloud.shape)
    b, _, _, _ = cloud.shape
    pred_list = torch.empty(b, 4)
    for i in range(len(cloud)):
        curr_quat = loaded_model(cloud[i].view(-1))
        pred_list[i] = curr_quat

    return pred_list

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
        curr_quat = svd.direct_SVD(cloud[i])
        quat_list[i] = curr_quat

    return quat_list

def get_batch_angle_diff(cloud: torch.Tensor, pred_quat: np.ndarray, true_quat: np.ndarray):
    svd_quat = get_svd_quat(cloud)
    svd_list = M.quat_angle_diff(svd_quat, true_quat, reduce=False)
    net_list = M.quat_angle_diff(pred_quat, true_quat, reduce=False)
    return svd_list, net_list #two torch.Tensors

def generate_fig(svd_list:torch.Tensor, net_list:torch.Tensor):
    n = np.arange(len(svd_list))
    m = np.arange(len(net_list))
    
    fig, ax = plt.subplots()

    ax.plot(n, svd_list, 'ro')
    ax.plot(m, net_list, 'b^')
    
    return fig

def generate_dict(svd_list:torch.Tensor, net_list:torch.Tensor):
    save_data = {}
    save_data[str(svd_list)] = svd_list
    save_data[str(net_list)] = net_list
    return save_data

def main():
    parser = argparse.ArgumentParser(description = 'load cloud from npz and generate figure')
    parser.add_argument('npz_path', type = str, help = 'path of npz file')
    parser.add_argument('chkpt_path', type = str, help = 'path of lightning check point')
    parser.add_argument('output_path', type = str, help = 'path of save figures and text')

    args = parser.parse_args()

    check_point = args.chkpt_path
    cloud_data  = args.npz_path
    save_path = args.output_path

    pointnet_model = read_check_point(check_point)
    cloud, true_quat = load_npz(cloud_data)
    pred_quat = forward_loaded_model(pointnet_model, cloud)

    list_svd_diff, list_net_diff = get_batch_angle_diff(cloud, pred_quat, true_quat)
    delta_q_fig = generate_fig(list_svd_diff, list_net_diff)
    delta_q_fig.savefig(save_path+"angle_diff.png")

    delta_q_dict = generate_dict(list_svd_diff, list_net_diff)
    np.savez(save_path+"angle_diff.npz", **delta_q_dict)

if __name__ == '__main__':
    main()
