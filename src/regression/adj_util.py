import numpy as np
import torch

def quat_to_adj(q: torch.Tensor) -> torch.Tensor:
    """
    convert quaternion to the 4x4 adjugate matrix
    """
    adjugate = []
    qx,qy,qz,q0 = q
    q_ordered = [float(q0),float(qx),float(qy),float(qz)]
    adjugate.append([float(q0)* item for item in q_ordered])
    adjugate.append([float(qx)* item for item in q_ordered])
    adjugate.append([float(qy)* item for item in q_ordered])
    adjugate.append([float(qz)* item for item in q_ordered])
    return torch.as_tensor(adjugate)

def batch_quat_to_adj(q_batch:torch.Tensor) -> torch.Tensor:
    """
    convert a batch of quaternion from DL to adj for training
    """
    b, _ = q_batch.shape
    adj_batch = torch.empty(b,4,4)

    for i in range(len(q_batch)):
        curr_adj = quat_to_adj(q_batch[i])
        adj_batch[i] = curr_adj
    return adj_batch

def adj_to_quat(adj_mat: torch.Tensor) -> torch.Tensor:
    norms = [adj_mat[0].norm(),adj_mat[1].norm(),adj_mat[2].norm(),adj_mat[3].norm()]
    max_idx = norms.index(max(norms))
    q_pred = adj_mat[max_idx]/adj_mat[max_idx].norm()
    q0,qx,qy,qz = q_pred
    q_pred_order = [qx,qy,qz,q0]
    return torch.as_tensor(q_pred_order)

def batch_adj_to_quat(adj_batch: torch.Tensor) -> torch.Tensor:
    b, _, _ = adj_batch.shape
    q_batch = torch.empty(b,4)

    for i in range(len(adj_batch)):
        curr_q = adj_to_quat(adj_batch[i])
        q_batch[i] = curr_q
    return q_batch

def vec_to_adj(vec: torch.Tensor) -> torch.Tensor:
    """
    convert the batch of 10 dim vector from network's output
    to the adjugate matrices
    """
    if vec.dim() < 2:
        vec.unsqueeze(dim = 0)

    idx = torch.triu_indices(4,4)
    adj = vec.new_zeros(vec.shape[0],4,4)
    adj[:,idx[0],idx[1]] = vec
    adj[:,idx[1],idx[0]] = vec
    return adj.squeeze()