import numpy as np
import scipy.io

from typing import Dict, Any, Tuple

from torchvision import transforms


    
def read_annotaions(ann_file:str) -> Dict[str, Any]:
    """@args:
    segmented: indicate whether there is a semantic map available
    objects: wrap up all pose related items and bounding box 
    """
    ann_data = scipy.io.loadmat(ann_file)

    img_file = ann_data['record']['filename'][0][0][0]
    segmented = ann_data['record']['segmented'][0][0][0]
    obj = ann_data['record']['objects'][0][0][0]

    category = obj['class'][0]
    if not obj['viewpoint']:
        return {}
    elif 'distance' not in obj['viewpoint'].dtype.names:
        return {}
    elif obj['viewpoint']['distance'][0][0][0][0] == 0:
        return {}


    viewpoint = obj['viewpoint']
    azimuth = viewpoint['azimuth'][0][0][0][0] 
    elevation = viewpoint['elevation'][0][0][0][0] 
    distance = viewpoint['distance'][0][0][0][0]
    focal = viewpoint['focal'][0][0][0][0]
    theta = viewpoint['theta'][0][0][0][0] # in plane rotation of the image
    principal = np.array([viewpoint['px'][0][0][0][0],
                            viewpoint['py'][0][0][0][0]])
    curr_dict = {
            'image_name': img_file,
            'category': category, 
            'bbox': obj['bbox'][0],
            'view':{
                'azimuth': azimuth,
                'elevation': elevation,
                'distance': distance,
                'focal': focal,
                'theta': theta
            },
            'intrinsic':{
                'focal': focal,
                'principal': principal
            }
        }
    
    return curr_dict

class ROILoader:
    """@brief: base class to do image preprocess and augmentation
    we do cropping separately,as cropping is depedent on bbox
    that specified in the annotation,
    that being said, we do resize after cropping
    """
    def __init__(self, resize_shape:int) -> None:
        self.resize = resize_shape
        self.pixel_mean, self.pixel_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

        self.transform = transforms.Compose([
            transforms.ToPILImage(), # most attribute of torchvision needed
            transforms.Resize(self.resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.pixel_mean, std=self.pixel_std)
        ])


