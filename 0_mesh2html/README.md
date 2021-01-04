# volume2stl
This repo helps you to transform a volume array to a .stl file and then visualize it.

## Sidenote:
kimimaro requires the following installations when used on rc cluster:
```
conda install cython
module load gcc # or module load g++
conda install numpy
pip install kimimaro
```

# PROTOTYPE A: 0_mesh2html folder

0 Preprocess segments:
-
- specify segments you want to process
- dilate slightly the segments
- create mask for dilation.
- np.unique(my_masked_id) --> select only part with biggest uc
- eliminates ouliers too disconnected/far from main structure
```
python preprocess_segments.py
```

1 Transform h5 to stl file
-
- [ ] IMPROVEMENT: RUN MARCHING CUBE WITH NEWER METHOD THAN CLASSIC 
- Download https://github.com/Rhoana/3dxp 
- Get List of matching ID pairs of mitos and dendrites and parse it: 
```
import numpy as np
idlist = np.loadtxt('seg_spiny_v2.txt')
print(":".join([str(int(i)) for i in idlist]))
```
- Run:
```
python ~/scriptsAndSoftware/repos/3dxp/PYTHON/all_stl.py --xyz /n/pfister_lab2/Lab/nils/snowproject/seg_64nm_maindendrite.h5 ./ -l 10547806:3214132:10892531:3976194:9471446:12105621:13544271:6238827:6659767:9387188:918525:15221648:5793428:1499496:12570277:2927761
```

2 Rotate stl file with PCA:
-
- Do you want to rotate mitochondria and dendrites that are matching ? The PCA rotation matrix needs to be the same for both instances of the mito-dendrite pair.
```
python PCA_mesh.py
```

3 Get access to x-server:
-
- e.g. Download .stl files and git repo to local machine

4 Run vtkplotlib to obtain .png from .stl
-
- install conda environment for py3
```
python create_figures.py
```

### TODO For the figures:
    - make background transparent
    - include mitos into cells,donglai has mapping
    - try out pyqt5 of vtkpll
        - bg transparent.
        - computation without opening possible?
        - put 2 elements in 1 plot
        - make neuron slightly transparent
        - colors for both
        
5 Visualize with html
-
- Display images in a grid:
    - Find way to iterate through images:


- Option B:
    - use plots and subplots to arrange images in a grid
    - convert master plot into html.
    

# Optional: 
### 1. Run ibexHelper on .stl volume to skeletonize a given instance.
```
git clone https://github.com/donglaiw/ibexHelper.git
```
- install ibexHelper
- test ibexHelper with seg id: 9494881

- opt=='0': # mesh -> skeleton
```
python ~/scriptsAndSoftware/repos/ibexHelper/script/demo.py 0
```
- opt=='1': # skeleton -> dense graph
```
python ~/scriptsAndSoftware/repos/ibexHelper/script/demo.py 1
```
- opt == '2': # reduced graph
```
python ~/scriptsAndSoftware/repos/ibexHelper/script/demo.py 2
```
- opt == '3': # generate h5 for visualization
```
python ~/scriptsAndSoftware/repos/ibexHelper/script/demo.py 3
```

### 2. Visualize the skeleton  point cloud in neuroglancer:
```
SCALE =[2, 2, 2]    
ibexpath = '~/snowjournal/volume2stl/results/' + '1499496/'

# point cloud
print('load ibex nodes')
node = h5py.File(ibexpath + 'node_pos.h5', 'r')
pts = np.array(node['main'])
with viewer.txn() as s:
    s.layers.append(name='1499496nodes_210', layer=neuroglancer.PointAnnotationLayer(points=SCALE*pts[:,np.array([2,1,0])]))
```

### 3. Find dendrite main axis:
```
python longest_axis.py
```    
save them with prev code of demo.py, opt == 1 method
