import numpy as np
import hydra
import logging
import omegaconf

import torch
import torch.nn as nn
import torch.utils.data
import torch.utils.tensorboard as tb

import pytorch_lightning as pl
import dataclasses

import regression.config as cf
from regression.model import Regress2DNet
import regression.metric as M
import regression.adj_util as A
from regression.dataset import Pascal3DDataset

class RegNetTrainer(pl.LightningModule):
    hparams: cf.RegNetTrainingConfig

    def __init__(self, config: cf.RegNetTrainingConfig) -> None:
        super().__init__()
        if not omegaconf.OmegaConf.is_config(config):
            config = omegaconf.OmegaConf.structured(config)

        config.network.n_class = len(config.data.category)
        self.category2idx = dict(zip(config.data.category, range(len(config.data.category))))

        self.save_hyperparameters(config)

        if config.network.regress_option == 'adjugate':
            regress_dim = 10
        elif config.network.regress_option == 'svd':
            regress_dim = 9

        self.regnet = Regress2DNet(config.network.n_class, regress_dim)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.regnet(x)
    
    def training_log(self, batch, loss:torch.Tensor, geodesic:torch.Tensor) -> None:
        self.log('train/frobenius loss respect to g.t.', loss)
        self.log('train/geodesic distance respect to g.t.', geodesic)
    
    def training_step(self, batch, batch_idx:int) -> torch.Tensor | np.Dict[str, np.Any]:
        image, anno = batch
        curr_category = anno["category"]
        rot = self(image, self.category2idx[curr_category])

        anno_a, anno_e, anno_t = anno['a'], anno['e'], anno['t']

        anno_euler = A.batch_euler_to_rot(anno_a, anno_e, anno_t)

        loss =  M.frobenius_norm_loss(rot, anno_euler)
        geodesic = M.geodesic_batch_mean(rot, anno_euler)

        self.training_log(batch, loss, geodesic)
        return loss
    
class RegNetDataModule(pl.LightningModule):
    hparams: cf.PascalDataConfig

    def __init__(self, config: cf.PascalDataConfig) -> None:
        super().__init__()
        self.config = config

        self.save_hyperparameters(config)

        self.ds = Pascal3DDataset(self.config.category, self.config.num_sample,
                                  self.config.file_path, self.config.crop)
        
        self.ds_train = None
        self.ds_val = None

    def setup(self, stage: str = None) -> None:
        # Define steps that should be done on 
        # every GPU, like splitting data, applying
        # transforms etc.
        if self.config.limit is not None:
            limit = min(self.config.limit, len(self.ds))
            self.ds, _ = torch.utils.data.random_split(self.ds, [limit, len(self.ds) - limit])

        num_train_samples = int(len(self.ds) * self.config.train_prop)

        self.ds_train, self.ds_val = torch.utils.data.random_split(
            self.ds, [num_train_samples, len(self.ds) - num_train_samples],
            torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.ds_train, self.batch_size, shuffle=True, num_workers=self.config.num_data_workers)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.ds_val, self.batch_size, shuffle=False, num_workers=self.config.num_data_workers)


        