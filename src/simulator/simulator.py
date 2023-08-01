import numpy as np
import hydra
import logging
import os

import simulator.cloud_from_random as cloud
import simulator.config as cf

@hydra.main(config_path=None, config_name='simulate', version_base='1.1')
def main(config: cf.SimulatorConfig):
    logger = logging.getLogger(__name__)

    logger.info("generate data from random with norm: " + str(config.source_norm) + 
                " and max angle: " + str(config.max_angle))

    quat, concate_cloud = cloud.generate_batches(config.batch_size,
                            config.num_points, config.sigma, config.rotation_format,
                            config.source_norm, config.max_angle, config.one_source,
                            config.manual, config.uniform_)
    one_src = "one_src_" if config.one_source else ""
    uniform = "unif_" if config.uniform_ else ""

    data_path = config.output_path + 'cloud_and_quat_' + str(config.batch_size) + '_' + one_src + uniform + 'sigma_' + str(config.sigma) + '.npz'
    save_data = {}
    save_data['cloud'] = concate_cloud
    save_data['quat'] = quat

    logger.info(f'saving generated cloud data and its quaternion to: {data_path}')
    np.savez(data_path, **save_data)

if __name__ == '__main__':
    from hydra.core.config_store import ConfigStore
    cs = ConfigStore()
    cs.store('simulate', node=cf.SimulatorConfig)
    main()

