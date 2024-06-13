import numpy as np
import os
from typing import List, Dict
import hydra
import logging

import util.config as cf
import regression.file_util as F
import util.pascal3d_annot as P

def count_pascal(base_path:str, category:List[str]) -> Dict[str,int]:
    counter_dict = {}

    for c in category:
        curr_image_path = base_path + "Images/" + c + "_pascal/"
        curr_files = F.list_files_in_dir(curr_image_path)
        print(f"check pascal's file path with category {c} \n")
        for i in range(3):
            print(curr_files[i])
        counter_dict[c] = len(curr_files)

    return counter_dict

def count_synthetic(base_path:str, category:List[str]) -> Dict[str,int]:
    counter_dict = {}

    for c in category:
        curr_folder = base_path + P.category_folderid(c)
        curr_subdirs = F.list_subdir_in_dir(curr_folder)

        print(f"check top directory of sythetic data of {c} \n")

        for i in range(len(curr_subdirs)):
            print(curr_subdirs[i])

        curr_count = 0
        for sub in curr_subdirs:
            print(f"check sub directory of sythetic data of {sub} \n")
            for i in range(3):
                print(curr_subdirs[i])
            curr_files = F.list_files_in_dir(sub)
            curr_count += len(curr_files)

        counter_dict[c] = curr_count
    
    return counter_dict

@hydra.main(config_path=None, config_name='counter', version_base='1.1' ) 
def main(config:cf.PascalFileCounterConfig):
    cnt_pascal = count_pascal(config.pascal_path, config.category)
    cnt_synthetic = count_synthetic(config.syn_path, config.category)

    counter_file = "./pascal_counter.txt"

    with open (counter_file, 'a') as f:
        f.write("below are counters for each category \n")

        for c in config.category:
            f.write(f"Psacal's {c} has {cnt_pascal[c]} files \n")
            f.write(f"Synthetic's {c} has {cnt_synthetic[c]} files \n")


if __name__ == '__main__':
    from hydra.core.config_store import ConfigStore
    cs = ConfigStore()
    cs.store('counter', node=cf.PascalFileCounterConfig)
    main()

