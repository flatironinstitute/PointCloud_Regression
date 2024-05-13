import torch
import numpy as np
import pytorch_lightning as pl
import torch.utils.data
import argparse
import matplotlib.pylab as plt

import regression.trainer_pascal as tr
import regression.dataset as ds
import regression.config as cf
import regression.adj_util as A
import regression.metric as M

def read_check_point(path: str):
    model = tr.RegNetTrainer.load_from_checkpoint(path)
    model.eval()
    return model