import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

import simulator.quat_util as Q
import util.pascal3d_annot as P

def test_quat2rot() -> None:
    """test fn to verify the Scipy uses JPL convention of quaternion
    in simulations, one should stick with a fixed convention
    """
    # generate a random quat
    quat_ = Q.generate_random_quat()
    r_scipy = R.from_quat(quat_)

    rot_scipy = r_scipy.as_matrix()
    rot_manual = Q.quat_to_rot(quat_, 'JPL')

    assert np.allclose(rot_scipy, rot_manual, atol=1e-5)

def test_mask_out() -> None:
    torch.manual_seed(42)
    batch_size = 5
    n_category = 10
    n_dim = 3

    x = torch.randn(batch_size, n_category, n_dim)
    labels = torch.randint(0, n_category, (batch_size,))
    mask_out = P.MaskOut(n_category)
    output = mask_out(x, labels)

    assert output.shape == (batch_size, n_dim)
    for i in range(batch_size):
        assert torch.allclose(output[i], x[i, labels[i]])