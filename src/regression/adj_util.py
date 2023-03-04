import numpy as np
import torch

# def quat_to_adj(q: torch.Tensor) -> torch.Tensor:
#     """
#     convert quaternion to the 4x4 adjugate matrix
#     """
#     adjugate = []
#     qx,qy,qz,q0 = q
#     q_ordered = [float(q0),float(qx),float(qy),float(qz)]
#     adjugate.append([float(q0)* item for item in q_ordered])
#     adjugate.append([float(qx)* item for item in q_ordered])
#     adjugate.append([float(qy)* item for item in q_ordered])
#     adjugate.append([float(qz)* item for item in q_ordered])
#     return torch.as_tensor(adjugate)
def quat_to_adj(q: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion to the 4x4 adjugate matrix.
    """
    q_cpu = q.cpu()
    qx, qy, qz, q0 = q_cpu.unbind()
    q_ordered = [q0, qx, qy, qz]
    adjugate = torch.tensor([
        [q0 * item for item in q_ordered],
        [qx * item for item in q_ordered],
        [qy * item for item in q_ordered],
        [qz * item for item in q_ordered]
    ], device=q.device)
    print("device of adj after single conversion: ", adjugate.device)
    return adjugate


def batch_quat_to_adj(q: torch.Tensor) -> torch.Tensor:
    """
    Convert batch of quaternions to batch of 4x4 adjugate matrices.
    """
    qx, qy, qz, q0 = q[:, 1], q[:, 2], q[:, 3], q[:, 0]
    q_ordered = torch.stack([q0, qx, qy, qz], dim=1).to(q.device)
    adjugate = torch.stack([
        q0[:, None] * q_ordered,
        qx[:, None] * q_ordered,
        qy[:, None] * q_ordered,
        qz[:, None] * q_ordered
    ], dim=2)
    print("device of adj after batch conversion: ", adjugate.device)

    return adjugate

def adj_to_quat(adj_mat: torch.Tensor) -> torch.Tensor:
    norms = adj_mat[:4].norm(axis=1)
    max_idx = norms.index(max(norms))
    q_pred = adj_mat[max_idx]/adj_mat[max_idx].norm()
    q0,qx,qy,qz = q_pred
    q_pred_order = [qx,qy,qz,q0]
    return torch.as_tensor(q_pred_order, device=adj_mat.device)

def batch_adj_to_quat(adj_batch: torch.Tensor) -> torch.Tensor:
    b, _, _ = adj_batch.shape
    q_batch = torch.empty(b,4, device=adj_batch.device)

    for i in range(len(adj_batch)):
        curr_q = adj_to_quat(adj_batch[i])
        q_batch[i] = curr_q
    return q_batch

def vec_to_adj(vec: torch.Tensor) -> torch.Tensor:
    """
    convert the batch of 10 dim vector from network's output
    to the adjugate matrices; equivalent to [Avec to A]
    """
    if vec.dim() < 2:
        vec.unsqueeze(dim = 0)

    idx = torch.triu_indices(4,4)
    adj = vec.new_zeros(vec.shape[0],4,4)
    adj[:,idx[0],idx[1]] = vec
    adj[:,idx[1],idx[0]] = vec
    return adj.squeeze()

def vec_to_quat(vec: torch.Tensor) -> torch.Tensor:
    """
    convert a vector to unit quaternion
    args:
    vec: 10 dim vector (default) from netwrok training
    return:
    4 dim unit quaternion
    """
    adj = vec_to_adj(vec)
    _, evs = torch.symeig(adj, eigenvectors=True)
    return evs[:,:,0].squeeze()