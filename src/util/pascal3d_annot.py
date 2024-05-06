import numpy as np
import scipy.io
import cv2 as cv
import torch
import torch.nn as nn

from typing import Dict, Any, Tuple, List

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

    category = obj['class'][0] # a string
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

class RoILoader:
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

class RoILoaderPascal(RoILoader):
    """@brief, the class to load and crop the pascal3D+ image
    with the bbox, the augmentation and preprocess are 
    inherits from the base class
    @args:
    image_path&anno_path: the based path of those two folders 
    will be specified in the Pascal3DDataset
    image_id: the id for image and annotation, should be a string
    for example: 2008_003743
    we keep image_id to force the loaded anno and image to be consistent
    context_scale: scaling factor of ROI
    resize_shape: an integer, as we assume it resize to a square
    """
    def __init__(self, category:str, image_id:str, resize_shape:int,
                 anno_path:str, image_path:str, context_pad:int = 16) -> None:
        self.anno_path = anno_path + image_id + ".mat"
        self.image_path = image_path + image_id + ".jpg"
        self.context_scale = float(resize_shape)/(resize_shape - 2*context_pad)

    def context_padding(self, boxes:np.ndarray) -> np.ndarray:  
        """@args:bbox is np.ndarray
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
        x1, x2, y1, y2 = self.context_padding(bbox)

        x1 = max(x1, 0)
        y1 = max(y1, 0)
        x2 = min(x2, w-1)
        y2 = min(y2, h-1)

        if x1 >= x2 or y1 >= y2:
            raise ValueError('[bad box] ' + "h,w=%s,%s   %s  %s" % (h, w, '(%s,%s,%s,%s)' % tuple(bbox), '(%s,%s,%s,%s)' % (x1, y1, x2, y2)))

        roi_img = image[y1:y2, x1:x2]
        roi_img = roi_img[:,:,::-1]

        return self.transform(roi_img)

    def __call__(self) -> np.ndarray:
        anno = read_annotaions(self.anno_path)
        image = cv.imread(self.image_path)

        return self.crop_roi(image, anno['bbox'])

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



        




