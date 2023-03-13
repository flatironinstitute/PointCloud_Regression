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
    return torch.as_tensor(adjugate, device=q.device)

def batch_quat_to_adj(q_batch:torch.Tensor) -> torch.Tensor:
    """
    Convert batch of quaternions to batch of 4x4 adjugate matrices
    """
    b, _ = q_batch.shape
    adj_batch = torch.empty(b, 4, 4, device=q_batch.device)
    for i in range(len(q_batch)):
         curr_adj = quat_to_adj(q_batch[i])
         adj_batch[i] = curr_adj
    #print("device of adj after batch conversion: ", adj_batch.device)
    return adj_batch

def adj_to_quat(adj_mat: torch.Tensor) -> torch.Tensor:
    norms = adj_mat[:4].norm(dim=1)
    max_idx = norms.argmax()
    q_pred = adj_mat[max_idx]/adj_mat[max_idx].norm()
    q0,qx,qy,qz = q_pred
    q_pred_order = [qx,qy,qz,q0]
    return torch.as_tensor(q_pred_order, device=adj_mat.device) #each new created tensor must specify the device, otherwise default send to cpu

def batch_adj_to_quat(adj_batch: torch.Tensor) -> torch.Tensor:
    b, _, _ = adj_batch.shape
    q_batch = torch.empty(b, 4, device=adj_batch.device)

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
        vec.unsqueeze_(dim = 0)

    idx = torch.triu_indices(4,4)
    adj = vec.new_zeros(vec.shape[0],4,4)
    adj[:,idx[0],idx[1]] = vec
    adj[:,idx[1],idx[0]] = vec
    return adj#.squeeze()

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
    if evs.dim() < 3:
        evs.unsqueeze_(dim = 0)

    return evs[:,:,0]#.squeeze()

# N x 3 -> N x 3 (unit norm)
def normalize_vectors(vecs):
    if vecs.dim() < 2:
        vecs = vecs.unsqueeze(dim=0)
    return vecs/vecs.norm(dim=1, keepdim=True, p=2)
    
# N x 3, N x 3 -> N x 3 (cross product)
def cross_product(u, v):
    assert(u.dim() == v.dim())
    if u.dim() < 2:
        u = u.unsqueeze(dim=0)
        v = v.unsqueeze(dim=0)
    batch = u.shape[0]
    i = u[:,1]*v[:,2] - u[:,2]*v[:,1]
    j = u[:,2]*v[:,0] - u[:,0]*v[:,2]
    k = u[:,0]*v[:,1] - u[:,1]*v[:,0]
    return torch.cat((i.view(batch,1), j.view(batch,1), k.view(batch,1)),1)

def sixdim_to_rotmat(sixdim: torch.Tensor) -> torch.Tensor:
    if sixdim.dim() < 2:
        sixdim = sixdim.unsqueeze(dim=0)
    x_raw = sixdim[:,0:3]#batch*3
    y_raw = sixdim[:,3:6]#batch*3
        
    x = normalize_vectors(x_raw)
    z = cross_product(x,y_raw) 
    z = normalize_vectors(z)
    y = cross_product(z,x)
    x = x.view(-1,3,1)
    y = y.view(-1,3,1)
    z = z.view(-1,3,1)
    rotmat = torch.cat((x,y,z), 2) #batch*3*3
    return rotmat