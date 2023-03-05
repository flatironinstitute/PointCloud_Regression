import numpy as np
import hydra
import logging
import omegaconf

import torch
import torch.nn as nn
import torch.utils.data
import torch.utils.tensorboard

import pytorch_lightning as pl
import dataclasses
import regression.config as cf
from regression.model import PointNet
import regression.metric as M
import regression.adj_util as A
from regression.dataset import SimulatedDataset, ModelNetDataset

class PointNetTrainer(pl.LightningModule):
    hparams: cf.PointNetTrainConfig

    def __init__(self, config: cf.PointNetTrainConfig) -> None:
        super().__init__()
        #The LightningModule automatically save all the hyperparameters 
        #passed to init simply by calling self.save_hyperparameters()
        #with config, we need to structured it before call save_hyperparameters()
        if not omegaconf.OmegaConf.is_config(config):
            config = omegaconf.OmegaConf.structured(config)

        self.save_hyperparameters(config)
        self.point_net = PointNet(config.model_config.hidden_size, 
                                    config.model_config.num_points,
                                    config.model_config.adj_option,
                                    config.model_config.batch_norm)  
        self.cf = config 

    def forward(self, x):
        return self.point_net(x)

    def training_log(self, batch, pred:torch.Tensor, quat:torch.Tensor, loss: float, batch_idx: int):
        net_option = self.cf.model_config.adj_option
        if net_option == "adjugate": #if output was 10 dim, pass the converted adj to log
            self.log('train/frob_loss', loss)
            pred_quat = A.batch_adj_to_quat(pred)
            angle_diff = M.quat_angle_diff(pred_quat, quat)
        elif net_option == "a-matrix":
            angle_diff = M.quat_angle_diff(pred, quat)
            self.log('train/a-mat quat chordal loss', loss)
        else:
            angle_diff = M.quat_angle_diff(pred, quat)
            self.log('train/chordal_square', loss)
        self.log('train/angle difference respect to g.t.', angle_diff)

    def training_step(self, batch, batch_idx: int):
        cloud, quat = batch
        pred = self(cloud)

        #loss can also wrap up separately for different options
        network_option = self.cf.model_config.adj_option
        if network_option == "adjugate":
            adj_pred = A.vec_to_adj(pred)
            adj_quat = A.batch_quat_to_adj(quat)
            loss = M.frobenius_norm_loss(adj_pred, adj_quat)

            #add constrain if specified
            if self.cf.constrain:
                selected_entries = [0, 4, 7, 9]
                norm_sq = torch.sum(pred[:, selected_entries], dim=1)
                norm_penalty = torch.mean((norm_sq - 1)**2)
                loss = loss + norm_penalty

            self.training_log(batch, adj_pred, quat, loss, batch_idx)
        elif network_option == "a-matrix":
            anti_quat = A.vec_to_quat(pred)
            loss = M.chordal_square_loss(anti_quat, quat)
            self.training_log(batch, anti_quat, quat, loss, batch_idx)
        else:
            loss = M.chordal_square_loss(pred, quat)
            self.training_log(batch, pred, quat, loss, batch_idx)
        return loss

    def validation_log(self, batch, pred:torch.Tensor, quat:torch.Tensor, loss: float, batch_idx: int):
        net_option = self.cf.model_config.adj_option
        if net_option == "adjugate":
            self.log('val/frob_loss', loss)
            pred_quat = A.batch_adj_to_quat(pred)
            angle_diff = M.quat_angle_diff(pred_quat, quat)
        elif net_option == "a-matrix":
            angle_diff = M.quat_angle_diff(pred, quat)
            self.log('val/a-mat quat chordal loss', loss)
        else:
            angle_diff = M.quat_angle_diff(pred, quat)
            self.log('val/chordal_square', loss)
        self.log('val/angle difference respect to g.t.', angle_diff)

    def validation_step(self, batch, batch_idx: int):
        cloud, quat = batch
        pred = self(cloud)

        network_option = self.cf.model_config.adj_option
        if network_option == "adjugate":
            adj_pred = A.vec_to_adj(pred)
            adj_quat = A.batch_quat_to_adj(quat) #convert g.t. quat to adj for the loss calc
            print("g.t. adjugate's device: ",adj_quat.device)
            loss = M.frobenius_norm_loss(adj_pred, adj_quat)
            self.validation_log(batch, adj_pred, quat, loss, batch_idx)
        elif network_option == "a-matrix":
            anti_quat = A.vec_to_quat(pred)
            loss = M.chordal_square_loss(anti_quat, quat)
            self.validation_log(batch, anti_quat, quat, loss, batch_idx)
        else:
            loss = M.chordal_square_loss(pred, quat)
            self.validation_log(batch, pred, quat, loss, batch_idx)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.point_net.parameters(), lr=self.hparams.optim.learning_rate)
        return optim

    
class PointNetDataModule(pl.LightningDataModule):
    def __init__(self, config: cf.TrainingDataConfig, batch_size: int) -> None:
        super().__init__()
        self.config = config
        self.batch_size = batch_size
        if self.config.model_net:
            self.ds = ModelNetDataset(hydra.utils.to_absolute_path(self.config.file_path), 
                                    self.config.category, self.config.num_points, self.config.sigma)
        else:
            self.ds = SimulatedDataset(hydra.utils.to_absolute_path(self.config.file_path))

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


@hydra.main(config_path=None, config_name='train', version_base='1.1' ) 
def main(config: cf.PointNetTrainConfig):
    logger = logging.getLogger(__name__)
    trainer = pl.Trainer(
        accelerator=config.device, 
        devices=config.num_gpus,
        log_every_n_steps=config.log_every,
        max_epochs=config.num_epochs)
    
    data_config = config.data
    dm = PointNetDataModule(data_config, config.batch_size)
    model = PointNetTrainer(config)

    trainer.fit(model,dm)

    if trainer.is_global_zero:
        if config.model_config.adj_option == "adjugate":
            logger.info(f'Finished training. Final Frobenius: {trainer.logged_metrics["train/frob_loss"]}')
        elif config.model_config.adj_option == "a-matrix":
            logger.info(f'Finished training. Final Chordal of A-Mat: {trainer.logged_metrics["train/a-mat quat chordal loss"]}')
        else:
            logger.info(f'Finished training. Final Chordal: {trainer.logged_metrics["train/chordal_square"]}')
        
        logger.info(f'Finished training. Final Chordal: {trainer.logged_metrics["train/angle difference respect to g.t."]}')


    

if __name__ == '__main__':
    from hydra.core.config_store import ConfigStore
    cs = ConfigStore()
    cs.store('train', node=cf.PointNetTrainConfig)
    main()
