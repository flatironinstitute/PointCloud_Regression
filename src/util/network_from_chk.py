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

def forward_loaded_model(loaded_model, cloud: torch.Tensor,net_option:str) -> torch.Tensor:
    #cloud data are load and convert from numpy load
    print("shape of tensor: ", cloud.shape)
    b, _, _, _ = cloud.shape

    curr_pred = loaded_model(cloud) #no flatten needed, as feat net will do the downsampling
    if net_option == "adjugate":
        print("predicted adjugate vec has shape: ", curr_pred.shape)
        pred_adj = A.vec_to_adj(curr_pred)
        pred_quat = A.batch_adj_to_quat(pred_adj)
        print('predicted adjugate has shape: ', pred_quat.shape)
    elif net_option == "a-matrix":
        pred_quat = A.vec_to_quat(curr_pred)
    elif net_option == "chordal":
        pred_quat = curr_pred
        print('predicted chordal has shape: ', pred_quat.shape)
    else:
        print("predicted 6D vec has shape: ", curr_pred.shape)
        rot_mat = A.sixdim_to_rotmat(curr_pred)
        pred_quat = A.rotmat_to_quat(rot_mat)
        print('predicted 6D has shape: ', pred_quat.shape)

    return pred_quat

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

def get_batch_angle_diff(cloud: torch.Tensor, true_quat: torch.Tensor,
                        pred_quat_adj: torch.Tensor = None, 
                        pred_quat_chr: torch.Tensor = None, 
                        pred_quat_amt: torch.Tensor = None,
                        pred_quat_six: torch.Tensor = None):
    #the input can be wrap up as a dict for different predicted quat
    svd_quat = get_svd_quat(cloud)
    svd_list = M.quat_angle_diff(svd_quat, true_quat, reduce=False)
    net_list_adj = M.quat_angle_diff(pred_quat_adj, true_quat, reduce=False) if pred_quat_adj is not None else None
    net_list_chr = M.quat_angle_diff(pred_quat_chr, true_quat, reduce=False) if pred_quat_chr is not None else None
    net_list_amt = M.quat_angle_diff(pred_quat_amt, true_quat, reduce=False) if pred_quat_amt is not None else None
    net_list_six = M.quat_angle_diff(pred_quat_six, true_quat, reduce=False) if pred_quat_six is not None else None
    return svd_list, net_list_adj, net_list_chr, net_list_amt, net_list_six#4 torch.Tensors

def generate_fig(svd_list:torch.Tensor, net_list_adj:torch.Tensor, net_list_chr:torch.Tensor, net_list_amt:torch.Tensor):
    #need wrap up more lines; need add title and axes
    l = np.arange(len(svd_list))
    m = np.arange(len(net_list_adj))
    n = np.arange(len(net_list_chr))
    k = np.arange(len(net_list_amt))
    
    fig, ax = plt.subplots(figsize=(10,7),dpi = 100)

    ax.plot(l, svd_list.detach().numpy(), 'ro', label = 'SVD Optimization')
    ax.plot(m, net_list_adj.detach().numpy(), 'b^', label = 'Adj Frob')
    #ax.plot(n, net_list_chr.detach().numpy(), 'y*', label = 'Chordal Sqr')
    ax.plot(n, net_list_amt.detach().numpy(), 'gx', label = 'A-Matrix')

    ax.set_title('Angle Differences via different training method')
    ax.set_xlabel('Index of Point Cloud')
    ax.set_ylabel('Differences in Angle')
    ax.legend()

    return fig

def generate_dict(svd_list:torch.Tensor, 
                net_list_adj: torch.Tensor = None, 
                net_list_chr: torch.Tensor = None, 
                net_list_amt: torch.Tensor = None,
                net_list_six: torch.Tensor = None):
    save_data = {}
    save_data["svd_list"] = svd_list.detach().numpy()
    save_data["net_list_adj"] = net_list_adj.detach().numpy() if net_list_adj is not None else []
    save_data["net_list_chr"] = net_list_chr.detach().numpy() if net_list_chr is not None else []
    save_data["net_list_amt"] = net_list_amt.detach().numpy() if net_list_amt is not None else []
    save_data["net_list_six"] = net_list_six.detach().numpy() if net_list_six is not None else []
    return save_data

def main():
    parser = argparse.ArgumentParser(description = 'load cloud from npz and generate figure')
    parser.add_argument('npz_path', type = str, help = 'path of npz file')
    parser.add_argument('chkpt_path_adj', type = str, help = 'path of lightning check point with adjugate training')
    parser.add_argument('chkpt_path_chr', type = str, help = 'path of lightning check point with chordal training')
    parser.add_argument('chkpt_path_amt', type = str, help = 'path of lightning check point with a-matrix training')
    parser.add_argument('chkpt_path_six', type = str, help = 'path of lightning check point with 6D training')
    parser.add_argument('output_path', type = str, help = 'path of save figures and text')

    args = parser.parse_args()

    #we set the path as string None, if the path is not available
    check_point_adj = args.chkpt_path_adj
    check_point_chr = args.chkpt_path_chr
    check_point_amt = args.chkpt_path_amt
    check_point_six = args.chkpt_path_six

    cloud_data  = args.npz_path
    save_path = args.output_path

    pointnet_model_adj = read_check_point(check_point_adj) if check_point_adj != "None" else None
    pointnet_model_chr = read_check_point(check_point_chr) if check_point_chr != "None" else None
    pointnet_model_amt = read_check_point(check_point_amt) if check_point_amt != "None" else None
    pointnet_model_six = read_check_point(check_point_six) if check_point_six != "None" else None

    cloud, true_quat = load_npz(cloud_data)
    pred_quat_adj = forward_loaded_model(pointnet_model_adj, cloud, "adjugate") if pointnet_model_adj != None else None
    pred_quat_chr = forward_loaded_model(pointnet_model_chr, cloud, "chordal") if pointnet_model_chr != None else None
    pred_quat_amt = forward_loaded_model(pointnet_model_amt, cloud, "a-matrix") if pointnet_model_amt != None else None
    pred_quat_six = forward_loaded_model(pointnet_model_six, cloud, "six-d") if pointnet_model_six != None else None

    list_svd_diff, list_adj_diff, list_chr_diff, list_amt_diff, list_six_diff = \
                get_batch_angle_diff(cloud, true_quat, pred_quat_adj, pred_quat_chr, pred_quat_amt, pred_quat_six)
    #delta_q_fig = generate_fig(list_svd_diff, list_adj_diff, list_chr_diff, list_amt_diff)
    #delta_q_fig.savefig(save_path+"/angle_diff.png")

    delta_q_dict = generate_dict(list_svd_diff, list_adj_diff, list_chr_diff, list_amt_diff, list_six_diff)
    np.savez(save_path+"/angle_diff_one_src.npz", **delta_q_dict)

if __name__ == '__main__':
    main()
