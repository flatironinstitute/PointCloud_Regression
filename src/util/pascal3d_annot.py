import numpy as np
import scipy.io
import cv2 as cv
import torch
import torch.nn as nn
import logging

import regression.adj_util as A

from typing import Dict, Any, Tuple, List

from torchvision import transforms

    
def read_annotaions(ann_file:str) -> List[Dict[str, Any]]:
    """@args:
    segmented: indicate whether there is a semantic map available
    objects: wrap up all pose related items and bounding box 
    @notice:
    some picture may contains multiple objects, thuus there is 
    a list of annotations in a single "ann_file"
    """
    ann_data = scipy.io.loadmat(ann_file)

    annotations = []

    img_file = ann_data['record']['filename'][0][0][0]
    segmented = ann_data['record']['segmented'][0][0][0]
    logging.debug(f"Image file: {img_file}, Segmented: {segmented}")

    objects = ann_data['record']['objects'][0][0][0]

    for o in objects:
        logging.debug(f"Processing object with keys: {o.dtype.names}")
        if not o['viewpoint']:
            logging.error(f"Viewpoint missing in one object from file: {ann_file}")
            continue
        elif 'distance' not in o['viewpoint'].dtype.names:
            logging.error("Distance missing in viewpoint")
            continue
        elif o['viewpoint']['distance'][0][0][0][0] == 0:
            continue

        bbox = o['bbox'][0]
        viewpoint = o['viewpoint']
        azimuth = viewpoint['azimuth'][0][0][0][0] 
        elevation = viewpoint['elevation'][0][0][0][0] 
        distance = viewpoint['distance'][0][0][0][0]
        focal = viewpoint['focal'][0][0][0][0]
        theta = viewpoint['theta'][0][0][0][0] # in plane rotation of the image
        principal = np.array([viewpoint['px'][0][0][0][0],
                                viewpoint['py'][0][0][0][0]])

        curr_dict = {
                'image_name': img_file,
                'category': o['class'][0], # a string 
                'bbox': bbox,
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
        annotations.append(curr_dict)
    
    return annotations

class RoILoader:
    """@brief: base class to do image preprocess and augmentation
    we do cropping separately,as cropping is depedent on bbox
    that specified in the annotation,
    that being said, we do resize after cropping
    """
    def __init__(self, resize_shape:int) -> None:
        self.resize = (resize_shape, resize_shape)
        self.pixel_mean, self.pixel_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

        self.transform = transforms.Compose([
            transforms.ToPILImage(), # most attribute of torchvision's Transform needed
            transforms.Resize(self.resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.pixel_mean, std=self.pixel_std)
        ])

class RoILoaderPascal(RoILoader):
    """@brief, the class to load and crop the pascal3D+ image
    with the bbox, the augmentation and preprocess are 
    inherits from the base class
    @args:
    image_path: the base path of the image folder 
    will be specified in the Pascal3DDataset
    image_id: the id for image and annotation, should be a string
    for example: 2008_003743
    context_scale: scaling factor of ROI
    resize_shape: an integer, as we assume it resize to a square
    anno_info: current annotation info, cause one image may contains
    different objects
    """
    def __init__(self, category:str, image_id:str, resize_shape:int,
                 anno_info:Dict[str,Any], image_path:str, context_pad:int = 16) -> None:
        super().__init__(resize_shape)
        self.anno_info = anno_info
        self.image_path = image_path + image_id + ".jpg"
        self.context_scale = float(resize_shape)/(resize_shape - 2*context_pad)

    def context_padding(self, boxes:np.ndarray) -> np.ndarray:  
        """@args:bbox is a np.ndarray
        we will do clipping in the roi_cropping
        """
        if self.context_scale == 1.0:
            return boxes
        _boxes = boxes.astype(np.float32).copy()
        x1, y1, x2, y2 = _boxes[0], _boxes[1], _boxes[2], _boxes[3]

        # Compute the expanded region
        half_height = (y2 - y1 + 1) / 2.0
        half_width = (x2 - x1 + 1) / 2.0
        center_x = x1 + half_width
        center_y = y1 + half_height

        # Calculate new corners with context scaling
        x1 = np.round(center_x - half_width * self.context_scale)
        x2 = np.round(center_x + half_width * self.context_scale)
        y1 = np.round(center_y - half_height * self.context_scale)
        y2 = np.round(center_y + half_height * self.context_scale)

        # Update _boxes with new values
        _boxes[0] = x1
        _boxes[1] = y1
        _boxes[2] = x2
        _boxes[3] = y2

        return _boxes.astype(np.int32)

    def crop_roi(self, image:np.ndarray, bbox:np.ndarray) -> np.ndarray:
        """clip and shift the image to RGB
        @Note: the final resizing will be specified in self.transforms
        """
        h, w, _ = image.shape
        x1, y1, x2, y2 = self.context_padding(bbox)

        x1 = max(x1, 0)
        y1 = max(y1, 0)
        x2 = min(x2, w-1)
        y2 = min(y2, h-1)

        if x1 >= x2 or y1 >= y2:
            raise ValueError('[bad box] ' + "h, w=%s, %s   %s  %s" % (h, w, '(%s, %s, %s, %s)' % tuple(bbox), '(%s, %s, %s, %s)' % (x1, y1, x2, y2)))

        roi_img = image[y1:y2, x1:x2]
        roi_img = roi_img[:,:,::-1]

        return self.transform(roi_img)

    def __call__(self) -> np.ndarray:
        image = cv.imread(self.image_path)

        bbox = self.anno_info.get('bbox')
        if bbox is None:
            print("Error: 'bbox' key missing in annotation.")

        return self.crop_roi(image, bbox)

class MaskOut(nn.Module):
    def __init__(self, n_category:int) -> None:
        super().__init__()
        self.n_category = n_category

    def forward(self, x:torch.Tensor, label:List[int]) -> torch.Tensor:
        """@args:
        x: the input tensor with a shape BxN_categoryxN_dim,
        N_dim is the dimension of the representation of the viewing angle
        label: the label list of the current batch, should be in the same device as x
        """
        label = torch.as_tensor(label, device=x.device)
        batch, _, _ = x.shape
        assert batch == label.size(0)
        assert (label >= 0).all() and (label < self.n_category).all() # labels out of range

        all_idx = torch.arange(batch)

        masked_shape = (batch, x.size()[2:]) # the second dim is the dim of pose representation
        return x[all_idx, label].view(*masked_shape)
    
def compose_euler_dict(anno:Dict[str,Any]) -> Dict[str, Any]:
    curr_dict = {"category":anno["category"],
                 "a":A.deg_to_rad(torch.tensor(anno["view"]["azimuth"])),
                 "e":A.deg_to_rad(torch.tensor(anno["view"]["elevation"])),
                 "t":A.deg_to_rad(torch.tensor(anno["view"]["theta"]))}
    
    return curr_dict



        




