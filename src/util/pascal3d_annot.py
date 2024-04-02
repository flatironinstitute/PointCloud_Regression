import numpy as np
import scipy.io

class Pascal3DAnnotations:
    def __init__(self, ann_file:str) -> None:
        ann_data = scipy.io.loadmat(ann_file)

        self.img_file = ann_data['record']['filename'][0][0][0]

        self.segmented = ann_data['record']['segmented'][0][0][0]

        self.objects = []
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
            theta = viewpoint['theta'][0][0][0][0] 
            principal = np.array([viewpoint['px'][0][0][0][0],
                                  viewpoint['py'][0][0][0][0]])
            viewport = viewpoint['viewport'][0][0][0][0]
