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

from torch.utils.data.dataloader import default_collate


class RegNetTrainer(pl.LightningModule):
    hparams: cf.RegNetTrainingConfig

    def __init__(self, config: cf.RegNetTrainingConfig) -> None:
        super().__init__()
        if not omegaconf.OmegaConf.is_config(config):
            config = omegaconf.OmegaConf.structured(config)

        self.save_hyperparameters(config)

        config.network.n_class = len(config.data.category)
        self.category2idx = dict(zip(config.data.category, range(len(config.data.category))))

        if config.network.regress_option == 'adjugate':
            regress_dim = 10
        elif config.network.regress_option == 'svd':
            regress_dim = 9

        self.regnet = Regress2DNet(config.network.n_class, regress_dim)
    
    def forward(self, x:torch.Tensor, category_id:int) -> torch.Tensor:
        return self.regnet(x, category_id)
    
    def training_log(self, batch, loss:torch.Tensor, geodesic:torch.Tensor) -> None:
        self.log('train/frobenius loss respect to g.t.', loss)
        self.log('train/geodesic distance respect to g.t.', geodesic)
    
    def training_step(self, batch, batch_idx:int) -> torch.Tensor:
        images, annos, categories = batch

        category_indices = torch.tensor([self.category2idx[cat] for cat in categories], device=self.device)
        
        rot = self(images, category_indices)

        anno_a, anno_e, anno_t = annos['a'], annos['e'], annos['t']

        anno_euler = A.batch_euler_to_rot(anno_a, anno_e, anno_t)

        loss =  M.frobenius_norm_loss(rot, anno_euler)
        geodesic = M.geodesic_batch_mean(rot, anno_euler)

        self.training_log(batch, loss, geodesic)
        return loss
    
    def validation_log(self, batch, loss:torch.Tensor, geodesic:torch.Tensor) -> None:
        self.log('val/frobenius loss respect to g.t.', loss)
        self.log('val/geodesic distance respect to g.t.', geodesic)        

    def validation_step(self, batch, batch_idx:int) -> torch.Tensor:
        images, annos, categories = batch

        category_indices = torch.tensor([self.category2idx[cat] for cat in categories], device=self.device)
        
        rot = self(images, category_indices)

        anno_a, anno_e, anno_t = annos['a'], annos['e'], annos['t']

        anno_euler = A.batch_euler_to_rot(anno_a, anno_e, anno_t)

        loss =  M.frobenius_norm_loss(rot, anno_euler)
        geodesic = M.geodesic_batch_mean(rot, anno_euler)


        self.validation_log(batch, loss, geodesic)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.regnet.parameters(), lr=self.hparams.optim.learning_rate)
        return optim
    
class RegNetDataModule(pl.LightningDataModule):
    def __init__(self, config: cf.PascalDataConfig, batch_size: int) -> None:
        super().__init__()
        self.config = config
        self.batch_size = batch_size

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
        return torch.utils.data.DataLoader(self.ds_train, self.batch_size, shuffle=True, 
                                           num_workers=self.config.num_data_workers,
                                           collate_fn=self.custom_collate_fn)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.ds_val, self.batch_size, shuffle=False, 
                                           num_workers=self.config.num_data_workers,
                                           collate_fn=self.custom_collate_fn)
    
    @staticmethod
    def custom_collate_fn(batch):
        flat_list = [item for sublist in batch for item in sublist]
    
        # Separate images and annotations
        images = [item[0] for item in flat_list]
        annotations = [item[1] for item in flat_list]
        categories = [item[1]['category'] for item in flat_list]  
        
        # Use default_collate to properly batch images and handle annotations
        batched_images = default_collate(images)
        batched_annos = []
        keys = annotations[0].keys()  # Assuming all dictionaries have the same structure
        for i in range(len(annotations)):
            single_anno = {key: annotations[i][key] for key in keys}
            batched_annos.append(single_anno)
        batched_categories = default_collate(categories)

        print("batched dict: ", batched_annos)
        
        return batched_images, batched_annos, batched_categories

@hydra.main(config_path=None, config_name='train', version_base='1.1')
def main(config: cf.RegNetTrainingConfig):
    logger = logging.getLogger(__name__)
    trainer = pl.Trainer(
        accelerator=config.device, 
        devices=config.num_gpus,
        log_every_n_steps=config.log_every,
        max_epochs=config.num_epochs)
    
    data_config = config.data

    dm = RegNetDataModule(data_config, config.batch_size)
    model = RegNetTrainer(config)

    trainer.fit(model, dm)

    if trainer.is_global_zero:
        logger.info(f'Finished training. Final Frobenius: {trainer.logged_metrics["train/frobenius loss respect to g.t."]}')
        logger.info(f'Finished training. Final Geodesic distance: {trainer.logged_metrics["train/geodesic distance respect to g.t."]}')

if __name__ == '__main__':
    from hydra.core.config_store import ConfigStore
    cs = ConfigStore()
    cs.store('train', node=cf.RegNetTrainingConfig)
    main()
