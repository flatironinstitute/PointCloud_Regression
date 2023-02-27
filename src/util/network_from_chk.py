import torch
import numpy as np
import pytorch_lightning as pl
import torch.utils.data
import argparse
import matplotlib.pylab as plt

import regression.trainer as tr
import regression.dataset as ds
import regression.config as cf
import regression.adj_util as A
import regression.metric as M
import util.optimal_svd as svd

def read_check_point(path: str):
    model = tr.PointNetTrainer.load_from_checkpoint(path)
    model.eval()
    return model

def forward_loaded_model(loaded_model, cloud: torch.Tensor,adj_option: bool) -> torch.Tensor:
    #cloud data are load and convert from numpy load
    print("shape of tensor: ", cloud.shape)
    b, _, _, _ = cloud.shape
    pred_list = torch.empty(b, 4)
    for i in range(len(cloud)):
        curr_pred = loaded_model(cloud[i]) #no flatten needed, as feat net will do the downsampling
        if adj_option:
            curr_quat = A.adj_to_quat(curr_pred)
        else:
            curr_quat = curr_pred
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
        curr_quat = torch.as_tensor(svd.direct_SVD(cloud[i]),dtype=torch.float32)
        quat_list[i] = curr_quat

    return quat_list

def get_batch_angle_diff(cloud: torch.Tensor, pred_quat_adj: torch.Tensor, pred_quat_chr: torch.Tensor, true_quat: torch.Tensor):
    #the input can be wrap up as a dict for different predicted quat
    svd_quat = get_svd_quat(cloud)
    svd_list = M.quat_angle_diff(svd_quat, true_quat, reduce=False)
    net_list_adj = M.quat_angle_diff(pred_quat_adj, true_quat, reduce=False)
    net_list_chr = M.quat_angle_diff(pred_quat_chr, true_quat, reduce=False)
    return svd_list, net_list_adj, net_list_chr #three torch.Tensors

def generate_fig(svd_list:torch.Tensor, net_list_adj:torch.Tensor, net_list_chr:torch.Tensor):
    #need wrap up more lines; need add title and axes
    n = np.arange(len(svd_list))
    m = np.arange(len(net_list_adj))
    k = np.arange(len(net_list_chr))
    
    fig, ax = plt.subplots()

    ax.plot(n, svd_list.detach().numpy(), 'ro', label = 'SVD Optim')
    ax.plot(m, net_list_adj.detach().numpy(), 'b^', label = 'Pred by Adj Frob')
    ax.plot(k, net_list_chr.detach().numpy(), 'gx', label = 'Pred by Chordal Sqr')

    ax.set_xlabel('Index of Point Cloud')
    ax.set_ylabel('Differences in Angle')
    
    return fig

def generate_dict(svd_list:torch.Tensor, net_list_adj:torch.Tensor, net_list_chr:torch.Tensor):
    save_data = {}
    save_data["svd_list"] = svd_list.detach().numpy()
    save_data["net_list_adj"] = net_list_adj.detach().numpy()
    save_data["net_list_chr"] = net_list_chr.detach().numpy()
    return save_data

def main():
    parser = argparse.ArgumentParser(description = 'load cloud from npz and generate figure')
    parser.add_argument('npz_path', type = str, help = 'path of npz file')
    parser.add_argument('chkpt_path_adj', type = str, help = 'path of lightning check point with adjugate training')
    parser.add_argument('chkpt_path_chr', type = str, help = 'path of lightning check point with chordal training')
    parser.add_argument('output_path', type = str, help = 'path of save figures and text')

    args = parser.parse_args()

    check_point_adj = args.chkpt_path_adj
    check_point_chr = args.chkpt_path_chr
    cloud_data  = args.npz_path
    save_path = args.output_path

    pointnet_model_adj = read_check_point(check_point_adj)
    pointnet_model_chr = read_check_point(check_point_chr)
    cloud, true_quat = load_npz(cloud_data)
    pred_quat_adj = forward_loaded_model(pointnet_model_adj, cloud, True)
    pred_quat_chr = forward_loaded_model(pointnet_model_chr, cloud, False)

    list_svd_diff, list_adj_diff, list_chr_diff = get_batch_angle_diff(cloud, pred_quat_adj, pred_quat_chr, true_quat)
    delta_q_fig = generate_fig(list_svd_diff, list_adj_diff, list_chr_diff)
    delta_q_fig.savefig(save_path+"/angle_diff.png")

    delta_q_dict = generate_dict(list_svd_diff, list_adj_diff, list_chr_diff)
    np.savez(save_path+"/angle_diff.npz", **delta_q_dict)

if __name__ == '__main__':
    main()
