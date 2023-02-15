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
from regression.model import FeedForward
import regression.metric as M
from regression.dataset import SimulatedDataset

class MLPTrainer(pl.LightningModule):
    hparams: cf.FeedForwardTrainConfig

    def __init__(self, config: cf.FeedForwardTrainConfig) -> None:
        super().__init__()
        #The LightningModule automatically save all the hyperparameters 
        #passed to init simply by calling self.save_hyperparameters()
        #with config, we need to structured it before call save_... ???
        if not omegaconf.OmegaConf.is_config(config):
            config = omegaconf.OmegaConf.structured(config)

        self.save_hyperparameters(config)
        print('this is test for config')
        self.feed_forward = FeedForward(config.model_config.num_hidden,
                        config.model_config.hidden_size, 
                        config.model_config.num_points)   

    def forward(self, x):
        return self.feed_forward(x)

    def training_log(self, batch, pred:torch.Tensor, quat:torch.Tensor, loss: float, batch_idx: int):
        chordal = M.chordal_square_loss(pred, quat)
        self.log('train/mse_loss', loss)
        self.log('train/chordal_square', chordal)

    def training_step(self, batch, batch_idx: int):
        cloud, quat = batch
        pred = self(cloud)

        #loss = M.chordal_square_loss(pred, quat)
        loss = M.mean_square(pred, quat)
        self.training_log(batch, pred, quat, loss, batch_idx)
        return loss

    def validation_log(self, batch, pred:torch.Tensor, quat:torch.Tensor, loss: float, batch_idx: int):
        chordal = M.chordal_square_loss(pred, quat)
        self.log('val/mse_loss', loss)
        self.log('val/chordal_square', chordal)

    def validation_step(self, batch, batch_idx: int):
        cloud, quat = batch
        pred = self(cloud)

        loss = M.mean_square(pred, quat)
        self.validation_log(batch, pred, quat, loss, batch_idx)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.feed_forward.parameters(), lr=self.hparams.optim.learning_rate)
        return optim

    
class MLPDataModule(pl.LightningDataModule):
    def __init__(self, config: cf.TrainingDataConfig, batch_size: int) -> None:
        super().__init__()
        self.config = config
        self.batch_size = batch_size

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
def main(config: cf.FeedForwardTrainConfig):
    logger = logging.getLogger(__name__)
    trainer = pl.Trainer(
        accelerator=config.device, 
        devices=config.num_gpus,
        log_every_n_steps=config.log_every,
        max_epochs=config.num_epochs)
    
    data_config = config.data
    dm = MLPDataModule(data_config, config.batch_size)
    model = MLPTrainer(config)

    trainer.fit(model,dm)

    if trainer.is_global_zero:
        logger.info(f'Finished training. Final loss: {trainer.logged_metrics["train/mse_loss"]}')
        logger.info(f'Finished training. Final chordal: {trainer.logged_metrics["train/chordal_square"]}')
    

if __name__ == '__main__':
    from hydra.core.config_store import ConfigStore
    cs = ConfigStore()
    cs.store('train', node=cf.FeedForwardTrainConfig)
    main()
