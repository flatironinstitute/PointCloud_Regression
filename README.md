# PointCloud_Regression
Point Cloud regression on simulated and ModelNet40 datasets.

<img src="https://github.com/EmperorAkashi/PointCloud_Regression/blob/main/docs/Figure2-v1.jpg" width="75%" >
Our representation illustrates how quaternion space in 2D must be covered by multiple alternatives solutions depending on the actual 2D cosine and sine rotation parameters.
 
# Get started
### Use our architecture on the Jupyter Notebook 
start from create a virtual environment, i.e.

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


