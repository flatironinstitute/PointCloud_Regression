import os
import numpy as np
from typing import List
import glob

###read all files from a dir###
def list_files_in_dir(top_dir: str) -> list:
    """ method that lists all files of a given directory
    :param 
        top_dir: directory in which the files are searched in
    """
    top_dir = os.path.abspath(top_dir)
    files = [os.path.join(top_dir, x) for x in os.listdir(top_dir)
                if os.path.isfile(os.path.join(top_dir, x))]
    return files

def read_off_file(file_path: str) -> np.ndarray:
    with open(file_path) as f:
        lines = f.readlines()

    data_info = [float(i) for i in lines[1].split()]
    num_pts = int(data_info[0])
    clouds = lines[2:num_pts+2]

    vertices = []
    for line in clouds:
        row = line.strip().split()
        curr_pt = [float(i) for i in row]
        vertices.append(curr_pt)
    vertices = np.array(vertices)
    return vertices

###read kitti's velodyne lidar###
###calibration are in the image directory###
def get_velo(file_name:str) -> np.ndarray:
    points = np.fromfile(file_name, dtype=np.float32).reshape(-1, 4)
    velo_file = points[:, :3] 
    return velo_file

def get_all_bins(seq_path:str) -> List:
    velo_bins = sorted(glob.glob(seq_path + '/*.bin'))
    return velo_bins