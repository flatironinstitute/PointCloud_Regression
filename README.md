# PointCloud_Regression
Point Cloud regression with new algebraical representation on ModelNet40 datasets (ICCV 2023)

<img src="https://github.com/EmperorAkashi/PointCloud_Regression/blob/main/docs/Figure2-v1.jpg" width="75%" >
Our representation illustrates how quaternion space in 2D must be covered by multiple alternative solutions depending on the actual 2D cosine and sine rotation parameters.

## **Related Publications**
> **Algebraically rigorous quaternion framework for the neural network pose estimation problem**  
> Chen Lin, Andrew Hanson, and Sonya Hanson  
> ICCV 2023 [[paper]](https://openaccess.thecvf.com/content/ICCV2023/html/Lin_Algebraically_Rigorous_Quaternion_Framework_for_the_Neural_Network_Pose_Estimation_ICCV_2023_paper.html) 
 
# Get started
### Use our architecture on the Jupyter Notebook 
start by creating a virtual environment, i.e.

`conda create -n env_name` 

in your terminal. Then activate the environment via:

`conda activate env_name`

In the environment, first, install our package:

`pip install -e .`

the dependencies should automatically be installed via `setup.py`; Alternatively, you could create a `docker image`.

Install the `ipykernel` via conda install ipykernel`.

Create a new kernel for your environment by running the command `python -m ipykernel install --user --name=env_name`.

Start Jupyter Notebook using the command Jupyter notebook. In the Jupyter Notebook interface, you should now see "env_name" as an option in the kernel dropdown menu.

Finally, you could import any module in our code base as a regular package:

```python
from regression.model import PointNet
import regression.metric as M
import regression.adj_util as A
from regression.dataset import SimulatedDataset, ModelNetDataset
```

Finally, one could run the test via `pytest /your/repository/path/PointCloud_Regression/tests/`

The test pipeline tested the convention of quaternion, where our homemade code uses Hamitonian convention, but the `scipy.spatial.transform` uses JPL convention


