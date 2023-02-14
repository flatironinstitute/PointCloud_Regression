import torch
import numpy as np
import pytorch_lightning as pl
import torch.utils.data
import hydra
import dataclasses

import regression.trainer as tr
import regression.dataset as ds
import regression.config as cf
import regression.metric as M

from scipy.spatial.transform import Rotation as R

def readCheckPoint(path):
    model = tr.MLPTrainer.load_from_checkpoint(path)
    model.eval()
    return model

def direct_SVD(cloud: torch.Tensor) -> np.ndarray:
    """
    calculate the relative rotation of a pair of cloud
    via direct optimal method by SVD decomposition
    """
    X = cloud[0]
    Y = cloud[1] #H or E = (x-x_0)(y-y_0)
    X_T = np.transpose(X)
    E = np.dot(X_T,Y)
    u, s, vh = np.linalg.svd(E, full_matrices=True)
    D = np.identity(3)
    vut = np.dot(np.transpose(vh),np.transpose(u))
    D_entry = np.linalg.det(vut)
    D[-1,-1] = np.sign(D_entry)
    R_opt_intermediate = np.dot(np.transpose(vh),D)
    R_opt = np.dot(R_opt_intermediate,np.transpose(u))
    r = R.from_matrix(R_opt)
        
    return r.as_quat()

class TestDataModule(pl.LightningDataModule):
    def __init__(self, config: cf.TrainingDataConfig, batch_size: int) -> None:
        super().__init__()
        self.config = config
        self.batch_size = batch_size

        self.ds = ds.SimulatedDataset(hydra.utils.to_absolute_path(self.config.file_path))

        self.ds_test = None

    def setup(self, stage: str = None) -> None:
        # Define steps that should be done on 
        # every GPU, like splitting data, applying
        # transforms etc.
        if self.config.limit is not None:
            limit = min(self.config.limit, len(self.ds))
            self.ds_test, _ = torch.utils.data.random_split(self.ds, [limit, len(self.ds) - limit])

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.ds_test, self.batch_size, shuffle=True, num_workers=self.config.num_data_workers)

@hydra.main(config_path=None, config_name='test', version_base='1.1') 
def main(config: cf.TestConfig):
    data_config = config.data
    load_path = config.chkpt_path
    print("current config ",config)
    print("test path ", load_path)
    dm = TestDataModule(data_config, config.batch_size)
    model = tr.MLPTrainer.load_from_checkpoint(load_path)

    trainer.test(model,dm)

if __name__ == '__main__':
    from hydra.core.config_store import ConfigStore
    cs = ConfigStore()
    cs.store('test', node=cf.TestConfig)
    main()



