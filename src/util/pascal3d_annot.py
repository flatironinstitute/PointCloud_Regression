import numpy as np
import scipy.io

    
def read_annotaions(ann_file:str) -> None:
    """@args:
segmented: indicate whether there is a semantic map available
objects: wrap up all pose related items 
"""
    ann_data = scipy.io.loadmat(ann_file)

    img_file = ann_data['record']['filename'][0][0][0]

    segmented = ann_data['record']['segmented'][0][0][0]

    objects = []
    for obj in ann_data['record']['objects'][0][0][0]:
        if not obj['viewpoint']:
            continue
        elif 'distance' not in obj['viewpoint'].dtype.names:
            continue
        elif obj['viewpoint']['distance'][0][0][0][0] == 0:
            continue


        viewpoint = obj['viewpoint']
        azimuth = viewpoint['azimuth'][0][0][0][0] 
        elevation = viewpoint['elevation'][0][0][0][0] 
        distance = viewpoint['distance'][0][0][0][0]
        focal = viewpoint['focal'][0][0][0][0]
        theta = viewpoint['theta'][0][0][0][0] # in plane rotation of the image
        principal = np.array([viewpoint['px'][0][0][0][0],
                                viewpoint['py'][0][0][0][0]])
        objects.append(
            {
                'azimuth': azimuth,
                'elevation': elevation,
                'distance': distance,
                'focal': focal,
                'theta': theta,
                'principal': principal
            }
        )
