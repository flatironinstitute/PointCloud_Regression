import numpy as np
import hydra
import logging
import omegaconf

import torch
import torch.nn as nn
import torch.utils.tensorboard as tb

import pytorch_lightning as pl
import regression.config as cf
from regression.model import FeedForward
import regression.metric as M
import regression.adj_util as A

class FeedForwardTrainer(pl.LightningModule):
    hparams: cf.NetworkConfig

    def __init__(self, config: cf.NetworkConfig) -> None:
        super().__init__()
        if not omegaconf.OmegaConf.is_config(config):
            config = omegaconf.OmegaConf.structured(config)

        self.save_hyperparameters(config)
        self.feed_forward = FeedForward(config.num_layer,config.hidden_size,
                                        config.adj_option)
        self.cf = config

    def forward(self, x):
        return self.feed_forward(x) 

    def training_log(self, batch, pred:torch.Tensor, quat:torch.Tensor, loss: float, batch_idx: int):
        net_option = self.cf.model_config.adj_option
        if net_option == "adjugate": #if output was 10 dim, pass the converted adj to log
            self.log('train/frob_loss', loss)
            vectors = A.adj_to_vec(A.batch_quat_to_adj(pred))
            writer = tb.SummaryWriter()
            writer.add_text('train/learned adj', str(vectors.tolist()))
            writer.close()

            self.log('train/g.t. adj', A.quat_to_adj(quat))
            angle_diff = M.quat_angle_diff(pred, quat)
        elif net_option == "a-matrix":
            angle_diff = M.quat_angle_diff(pred, quat)
            self.log('train/a-mat quat chordal loss', loss)
        elif net_option == "sid-d":
            angle_diff = M.quat_angle_diff(pred, quat)
            self.log('train/6d quat frob loss', loss)
        else:
            angle_diff = M.quat_angle_diff(pred, quat)
            self.log('train/chordal_square', loss)
        self.log('train/angle difference respect to g.t.', angle_diff)

    def training_step(self, batch, batch_idx: int):
        cloud, quat = batch
        pred = self(cloud)
        #loss can also wrap up separately for different options
        network_option = self.cf.adj_option
        loss_create = M.LossFactory()
        loss_computer = loss_create.create(network_option)

        loss, pred_quat = loss_computer.compute_loss(pred, quat, self.cf)
        
        self.training_log(batch, pred_quat, quat, loss, batch_idx)
        return loss

    def validation_log(self, batch, pred:torch.Tensor, quat:torch.Tensor, loss: float, batch_idx: int):
        net_option = self.cf.model_config.adj_option
        if net_option == "adjugate":
            self.log('val/frob_loss', loss)
            vectors = A.adj_to_vec(A.batch_quat_to_adj(pred))
            writer = tb.SummaryWriter()
            writer.add_text('val/learned adj', str(vectors.tolist()))
            writer.close()
            self.log('val/g.t. adj', A.quat_to_adj(quat))
            angle_diff = M.quat_angle_diff(pred, quat)
        elif net_option == "a-matrix":
            angle_diff = M.quat_angle_diff(pred, quat)
            self.log('val/a-mat quat chordal loss', loss)
        elif net_option == "sid-d":
            angle_diff = M.quat_angle_diff(pred, quat)
            self.log('val/6d quat frob loss', loss)
        else:
            angle_diff = M.quat_angle_diff(pred, quat)
            self.log('val/chordal_square', loss)
        self.log('val/angle difference respect to g.t.', angle_diff)

    def validation_step(self, batch, batch_idx: int):
        cloud, quat = batch
        pred = self(cloud)

        network_option = self.cf.adj_option
        loss_create = M.LossFactory()
        loss_computer = loss_create.create(network_option)

        loss, pred_quat = loss_computer.compute_loss(pred, quat, self.cf)
        self.validation_log(batch, pred_quat, quat, loss, batch_idx)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.point_net.parameters(), lr=self.hparams.optim.learning_rate)
        return optim
