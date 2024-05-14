import torch
import numpy as np
import pytorch_lightning as pl
import torch.utils.data
import matplotlib.pylab as plt
import logging
import hydra

import regression.trainer_pascal as tr
import regression.dataset as ds
import regression.adj_util as A
import regression.metric as M

from typing import Dict, List

import util.config as cf

def read_check_point(path: str):
    model = tr.RegNetTrainer.load_from_checkpoint(path)
    model.eval()
    return model

def save_results(results:List[List[Dict[str,torch.Tensor]]], filename:str) -> None:
    """@brief:
    depack and save the results from trainer.predict()
    """
    # Initialize lists to store aggregated results
    losses = []
    geodesics = []

    # Iterate over all batches and collect the results
    for batch_result in results:
        for result in batch_result:
            losses.append(result['loss'].cpu().numpy())  # Convert to CPU and NumPy array if not already
            geodesics.append(result['geodesic'].cpu().numpy())

    # Save results using np.savez
    np.savez(filename, np.array(losses), np.array(geodesics))


class PascalInferDataModule(pl.LightningDataModule):
    def __init__(self, config: cf.PascalInferConfig) -> None:
        self.config = config

    def setup(self, stage=None):
        if stage == 'test' or stage is None:
            self.test_dataset = ds.Pascal3DDataset(
                self.config.category,  
                self.config.num_sample,
                self.config.file_path,
                self.config.crop
            )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.config.batch_size, shuffle=False)
    
@hydra.main(config_path=None, config_name='infer', version_base='1.1')
def main(config: cf.PascalInferConfig):
    logger = logging.getLogger(__name__)
    trainer = pl.Trainer()
    

    dm = PascalInferDataModule(config)
    model = read_check_point(config.file_path)

    results = trainer.predict(model, dm, option_=config.option)
    save_results(results, config.output_path)

if __name__ == '__main__':
    from hydra.core.config_store import ConfigStore
    cs = ConfigStore()
    cs.store('train', node=cf.PascalInferConfig)
    main()
