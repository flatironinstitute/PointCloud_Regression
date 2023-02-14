import numpy as np
import torch

from scipy.spatial.transform import Rotation as R

def direct_SVD(cloud: torch.Tensor) -> np.ndarray:
    """
    calculate the relative rotation of a pair of cloud
    via direct optimal method by SVD decomposition;
    input is in the format of concatenated two clouds
    """
    X = cloud[0]
    Y = cloud[1] #H or E = (x-x_0)(y-y_0)
    X_T = np.transpose(X)
    E = np.dot(X_T,Y)
    u, s, vh = np.linalg.svd(E, full_matrices=True)
    D = np.identity(3)
    vut = np.dot(np.transpose(vh),np.transpose(u))
    D_entry = np.linalg.det(vut)
    D[-1,-1] = np.sign(D_entry)
    R_opt_intermediate = np.dot(np.transpose(vh),D)
    R_opt = np.dot(R_opt_intermediate,np.transpose(u))
    r = R.from_matrix(R_opt)
        
    return r.as_quat()