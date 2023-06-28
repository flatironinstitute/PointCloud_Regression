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

def vec_to_rot(vec: torch.Tensor, trace_norm: bool=False) -> torch.Tensor:
    "convert one 10-dim vec to a rotation matrix"
    trace = vec[0] + vec[4] + vec[7] + vec[9]
    if trace_norm:
        vec /= trace
    rot_mat = torch.empty(3,3,device=vec.device)
    rot_mat[0][0] = vec[0] + vec[4] - vec[7] - vec[9]
    rot_mat[0][1] = 2*(-vec[3] + vec[5])
    rot_mat[0][2] = 2*(vec[2] + vec[6])
    rot_mat[1][0] = 2*(vec[3] + vec[5])
    rot_mat[1][1] = vec[0] - vec[4] + vec[7] - vec[9]
    rot_mat[1][2] = 2*(-vec[1] + vec[8])
    rot_mat[2][0] = 2*(-vec[2] + vec[6])
    rot_mat[2][1] = 2*(vec[1] + vec[8])
    rot_mat[2][2] = vec[0] - vec[4] - vec[7] + vec[9]

    return rot_mat

def batch_vec_to_rot(vec_batch:torch.Tensor, trace_norm: bool=False) -> torch.Tensor:
    b, _ = vec_batch.shape
    rot_batch = torch.empty(b, 3, 3, device=vec_batch.device)

    for i in range(len(vec_batch)):
        curr_rot = vec_to_rot(vec_batch[i],trace_norm)
        rot_batch[i] = curr_rot

    return rot_batch

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

def adj_to_vec(adj: torch.Tensor) -> torch.Tensor:
    """
    convert batch of adjugate matrix to 10 dim vector
    """
    batch = len(adj)

    mask = torch.triu(torch.ones(4,4,device=adj.device))
    upper_tri = torch.masked_select(adj, mask.bool())
    vectors = upper_tri.view(batch,10)

    return vectors

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

def batch_vec_to_quat(vec_batch:torch.Tensor) -> torch.Tensor:
    """convert batch of 10-vec to quaternions
    """
    b, _ = vec_batch.shape
    quat_batch = torch.empty(b, 4, device=vec_batch.device)

    for i in range(len(vec_batch)):
        curr_quat = vec_to_quat(vec_batch[i])
        quat_batch[i] = curr_quat
        
    return quat_batch

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

def rotmat_to_quat(mat, ordering='xyzw'):
    """Convert a rotation matrix to a unit length quaternion.
        Valid orderings are 'xyzw' and 'wxyz'.
    """
    if mat.dim() < 3:
        R = mat.unsqueeze(dim=0)
    else:
        R = mat

    assert(R.shape[1] == R.shape[2])
    assert(R.shape[1] == 3)

    #Row first operation
    R = R.transpose(1,2)
    q = R.new_empty((R.shape[0], 4))

    cond1_mask = R[:, 2, 2] < 0.
    cond1a_mask = R[:, 0, 0] > R[:, 1, 1]
    cond1b_mask = R[:, 0, 0] < -R[:, 1, 1]

    if ordering=='xyzw':
        v_ind = torch.arange(0,3)
        w_ind = 3
    else:
        v_ind = torch.arange(1,4)
        w_ind = 0    

    mask = cond1_mask & cond1a_mask
    if mask.any():
        t = 1 + R[mask, 0, 0] - R[mask, 1, 1] - R[mask, 2, 2]
        q[mask, w_ind] =  R[mask, 1, 2]- R[mask, 2, 1]
        q[mask, v_ind[0]] = t
        q[mask, v_ind[1]] = R[mask, 0, 1] + R[mask, 1, 0]
        q[mask, v_ind[2]] = R[mask, 2, 0] + R[mask, 0, 2]
        q[mask, :] *= 0.5 / torch.sqrt(t.unsqueeze(dim=1))

    mask = cond1_mask & cond1a_mask.logical_not()
    if mask.any():
        t = 1 - R[mask,0, 0] + R[mask,1, 1] - R[mask,2, 2]
        q[mask, w_ind] =  R[mask,2, 0]-R[mask,0, 2]
        q[mask, v_ind[0]] = R[mask,0, 1]+R[mask,1, 0]
        q[mask, v_ind[1]] = t
        q[mask, v_ind[2]] = R[mask,1, 2]+R[mask,2, 1]
        q[mask, :] *= 0.5 / torch.sqrt(t.unsqueeze(dim=1))

    mask = cond1_mask.logical_not() & cond1b_mask
    if mask.any():
        t = 1 - R[mask,0, 0] - R[mask,1, 1] + R[mask,2, 2]
        q[mask, w_ind] =  R[mask,0, 1]-R[mask,1, 0]
        q[mask, v_ind[0]] = R[mask,2, 0]+R[mask,0, 2]
        q[mask, v_ind[1]] = R[mask,1, 2]+R[mask,2, 1]
        q[mask, v_ind[2]] = t
        q[mask, :] *= 0.5 / torch.sqrt(t.unsqueeze(dim=1))

    mask = cond1_mask.logical_not() & cond1b_mask.logical_not()
    if mask.any():
        t = 1 + R[mask, 0, 0] + R[mask,1, 1] + R[mask,2, 2]
        q[mask, w_ind] = t
        q[mask, v_ind[0]] = R[mask,1, 2]-R[mask,2, 1]
        q[mask, v_ind[1]] = R[mask,2, 0]-R[mask,0, 2]
        q[mask, v_ind[2]] = R[mask,0, 1]-R[mask,1, 0]
        q[mask, :] *= 0.5 / torch.sqrt(t.unsqueeze(dim=1))
    
    return q.squeeze()

def quat_to_rotmat(quat, ordering='xyzw'):
    """Form a rotation matrix from a unit length quaternion.
        Valid orderings are 'xyzw' and 'wxyz'.
    """
    if quat.dim() < 2:
        quat = quat.unsqueeze(dim=0)

    if not allclose(quat.norm(p=2, dim=1), 1.):
        print("Warning: Some quaternions not unit length ... normalizing.")
        quat = quat/quat.norm(p=2, dim=1, keepdim=True)

    if ordering is 'xyzw':
        qx = quat[:, 0]
        qy = quat[:, 1]
        qz = quat[:, 2]
        qw = quat[:, 3]
    elif ordering is 'wxyz':
        qw = quat[:, 0]
        qx = quat[:, 1]
        qy = quat[:, 2]
        qz = quat[:, 3]
    else:
        raise ValueError(
            "Valid orderings are 'xyzw' and 'wxyz'. Got '{}'.".format(ordering))

    # Form the matrix
    mat = quat.new_empty(quat.shape[0], 3, 3)

    qx2 = qx * qx
    qy2 = qy * qy
    qz2 = qz * qz

    mat[:, 0, 0] = 1. - 2. * (qy2 + qz2)
    mat[:, 0, 1] = 2. * (qx * qy - qw * qz)
    mat[:, 0, 2] = 2. * (qw * qy + qx * qz)

    mat[:, 1, 0] = 2. * (qw * qz + qx * qy)
    mat[:, 1, 1] = 1. - 2. * (qx2 + qz2)
    mat[:, 1, 2] = 2. * (qy * qz - qw * qx)

    mat[:, 2, 0] = 2. * (qx * qz - qw * qy)
    mat[:, 2, 1] = 2. * (qw * qx + qy * qz)
    mat[:, 2, 2] = 1. - 2. * (qx2 + qy2)

    return mat.squeeze_()

def allclose(mat1, mat2, tol=1e-6):
    """Check if all elements of two tensors are close within some tolerance.
    Either tensor can be replaced by a scalar.
    """
    return isclose(mat1, mat2, tol).all()


def isclose(mat1, mat2, tol=1e-6):
    """Check element-wise if two tensors are close within some tolerance.
    Either tensor can be replaced by a scalar.
    """
    return (mat1 - mat2).abs_().lt(tol)