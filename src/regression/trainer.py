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
from regression.model import PointNet
import regression.metric as M
import regression.adj_util as A
from regression.dataset import SimulatedDataset, ModelNetDataset
from regression.penalties import penalty_sum

class PointNetTrainer(pl.LightningModule):
    hparams: cf.PointNetTrainConfig

    def __init__(self, config: cf.PointNetTrainConfig) -> None:
        super().__init__()
        # The LightningModule automatically save all the hyperparameters 
        # passed to init simply by calling self.save_hyperparameters()
        # with config, we need to structured it before call save_hyperparameters()
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
        cloud, _ = batch
        print("debug input size: ", cloud.shape)
        rmsd_error = M.rmsd_diff(pred, cloud)

        if net_option == "adjugate": #if output was 10 dim, pass the converted adj to log
            self.log('train/frob_loss', loss)

            vectors = A.adj_to_vec(A.batch_quat_to_adj(pred))
            # writer = tb.SummaryWriter()
            # writer.add_text('train/learned adj', str(vectors.tolist()))
            # writer.close()

            angle_diff = M.quat_angle_diff(pred, quat)
            angle_cos = M.quat_cosine_diff(pred, quat)
        elif net_option == "a-matrix":
            angle_diff = M.quat_angle_diff(pred, quat)
            angle_cos = M.quat_cosine_diff(pred, quat)
            self.log('train/a-mat quat chordal loss', loss)
        elif net_option == "six-d":
            angle_diff = M.quat_angle_diff(pred, quat)
            angle_cos = M.quat_cosine_diff(pred, quat)
            self.log('train/6d quat frob loss', loss)
        elif net_option == "chordal":
            angle_diff = M.quat_angle_diff(pred, quat)
            angle_cos = M.quat_cosine_diff(pred, quat)
            self.log('train/chordal_square', loss)
        elif net_option == "l2chordal":
            angle_diff = M.quat_angle_diff(pred, quat)
            angle_cos = M.quat_cosine_diff(pred, quat)
            self.log('train/chordal L2 norm', loss)
        else:
            angle_diff = M.quat_angle_diff(pred, quat)
            angle_cos = M.quat_cosine_diff(pred, quat)
            self.log('train/rmsd loss', loss)

        self.log('train/angle difference respect to g.t.', angle_diff)
        self.log('train/cosine angle difference respect to g.t.', angle_cos)
        self.log('train/rmsd difference respect to g.t.', rmsd_error)

    def training_step(self, batch, batch_idx: int):
        cloud, quat = batch
        pred = self(cloud)
        #loss can also wrap up separately for different options
        network_option = self.cf.model_config.adj_option
        loss_create = M.LossFactory()
        loss_computer = loss_create.create(network_option)
        
        if network_option == "adjugate":
            loss, pred_quat = loss_computer.compute_loss(pred, quat, self.cf)
        elif network_option == "rmsd":
            loss, pred_quat = loss_computer.compute_loss(pred, quat, cloud, 
                                            self.cf.loss_config.rmsd_trace)
        else:
            loss, pred_quat = loss_computer.compute_loss(pred, quat)
        self.training_log(batch, pred_quat, quat, loss, batch_idx)
        return loss

    def validation_log(self, batch, pred:torch.Tensor, quat:torch.Tensor, loss: float, batch_idx: int):
        net_option = self.cf.model_config.adj_option
        cloud, _ = batch
        rmsd_error = M.rmsd_diff(pred, cloud)

        if net_option == "adjugate":
            self.log('val/frob_loss', loss)

            vectors = A.adj_to_vec(A.batch_quat_to_adj(pred))
            # writer = tb.SummaryWriter()
            # writer.add_text('val/learned adj', str(vectors.tolist()))
            # writer.close()

            angle_diff = M.quat_angle_diff(pred, quat)
            angle_cos = M.quat_cosine_diff(pred, quat)
        elif net_option == "a-matrix":
            angle_diff = M.quat_angle_diff(pred, quat)
            angle_cos = M.quat_cosine_diff(pred, quat)
            self.log('val/a-mat quat chordal loss', loss)
        elif net_option == "six-d":
            angle_diff = M.quat_angle_diff(pred, quat)
            angle_cos = M.quat_cosine_diff(pred, quat)
            self.log('val/6d quat frob loss', loss)
        elif net_option == "chordal":
            angle_diff = M.quat_angle_diff(pred, quat)
            angle_cos = M.quat_cosine_diff(pred, quat)
            self.log('val/chordal_square', loss)
        elif net_option == "l2chordal":
            angle_diff = M.quat_angle_diff(pred, quat)
            angle_cos = M.quat_cosine_diff(pred, quat)
            self.log('val/chordal L2 norm', loss)
        else:
            angle_diff = M.quat_angle_diff(pred, quat)
            angle_cos = M.quat_cosine_diff(pred, quat)
            self.log('val/rmsd loss', loss)

        self.log('val/angle difference respect to g.t.', angle_diff)
        self.log('val/cosine angle difference respect to g.t.', angle_cos)
        self.log('val/rmsd difference respect to g.t.', rmsd_error)

    def validation_step(self, batch, batch_idx: int):
        cloud, quat = batch
        pred = self(cloud)

        network_option = self.cf.model_config.adj_option
        loss_create = M.LossFactory()
        loss_computer = loss_create.create(network_option)

        if network_option == "adjugate":
            loss, pred_quat = loss_computer.compute_loss(pred, quat, self.cf)
        elif network_option == "rmsd":
            loss, pred_quat = loss_computer.compute_loss(pred, quat, cloud, 
                                            self.cf.loss_config.rmsd_trace)
        else:
            loss, pred_quat = loss_computer.compute_loss(pred, quat)
        self.validation_log(batch, pred_quat, quat, loss, batch_idx)
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
                                    self.config.category, self.config.num_points, 
                                    self.config.sigma,self.config.num_rot,
                                    self.config.range_max, self.config.range_min)
        else:
            self.ds = SimulatedDataset(hydra.utils.to_absolute_path(self.config.file_path),
                                    self.config.svd_mod)

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
        elif config.model_config.adj_option == "six-d":
            logger.info(f'Finished training. Final Frobenius of 6D: {trainer.logged_metrics["train/6d quat frob loss"]}')
        elif config.model_config.adj_option == "chordal":
            logger.info(f'Finished training. Final Chordal: {trainer.logged_metrics["train/chordal_square"]}')
        elif config.model_config.adj_option == "l2chordal":
            logger.info(f'Finished training. Final Chordal: {trainer.logged_metrics["train/chordal L2 norm"]}')
        else:
            logger.info(f'Finished training. Final RMSD: {trainer.logged_metrics["train/rmsd loss"]}')
        
        logger.info(f'Finished training. Final Angle Difference: {trainer.logged_metrics["train/angle difference respect to g.t."]}')
        logger.info(f'Finished training. Final Cosine Angle Difference: {trainer.logged_metrics["train/cosine angle difference respect to g.t."]}')


    

if __name__ == '__main__':
    from hydra.core.config_store import ConfigStore
    cs = ConfigStore()
    cs.store('train', node=cf.PointNetTrainConfig)
    main()
