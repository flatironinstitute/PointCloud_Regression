# PointCloud_Regression
Point Cloud regression on simulated and ModelNet40 dataset.

# Use our architecture on the Jupyter Notebook 
start from create a virtual environment, i.e.

`conda create -n env_name` 

in your terminal. Then activate the environment via:

conda activate env_name`

In the environment, first install our package:

`pip install -e .`

then install all dependenies by following the `docker` file.

Install the `ipykernel` via conda install ipykernel`.

Create a new kernel for your environment by running the command `python -m ipykernel install --user --name=env_name`.

Start Jupyter Notebook using the command jupyter notebook. In the Jupyter Notebook interface, you should now see "env_name" as an option in the kernel dropdown menu.

Finally you could import any module in our code base as a regular package:

```python
from regression.model import PointNet
import regression.metric as M
import regression.adj_util as A
from regression.dataset import SimulatedDataset, ModelNetDataset
```


