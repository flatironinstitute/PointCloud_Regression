import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

import simulator.quat_util as Q

def test_quat2rot():
    """test fn to verify the Scipy uses JPL convention of quaternion
    in simulations, one should stick with a fixed convention
    """
    # generate a random quat
    quat_ = Q.generate_random_quat()
    r_scipy = R.from_quat(quat_)

    rot_scipy = r_scipy.as_matrix()
    rot_manual = Q.quat_to_rot(quat_, 'JPL')

    assert np.allclose(rot_scipy, rot_manual, atol=1e-5)

