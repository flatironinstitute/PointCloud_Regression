import torch
import numpy as np
import pytorch_lightning as pl

import regression.trainer as tr

def read_check_point(path: str):
    model = tr.MLPTrainer.load_from_checkpoint(path)
    model.eval()
    return model

if __name__ == '__main__':
    #path = "/Users/clin/Documents/pointnet_out/epoch-99-step-2300.ckpt"
    path = "/mnt/home/clin/ceph/pointnet_run/try/lightning_logs/version_2123679/checkpoints/epoch=99-step=2300.ckpt"
    m = read_check_point(path)