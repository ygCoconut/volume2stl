# volume2stl
This repo helps you to transform a volume array to a .stl file and then visualize it.


# PROTOTYPE A
### 1. Transform h5 to stl file
#### IMPROVEMENT: RUN MARCHING CUBE WITH NEWER METHOD THAN CLASSIC 
- Download https://github.com/Rhoana/3dxp 
- Run:

```
python ~/scriptsAndSoftware/repos/3dxp/PYTHON/all_stl.py --xyz /n/pfister_lab2/Lab/donglai/mito/db/30um_human/seg_64nm.h5 ./ -l listof_ids.txt
```

### 2. Rotate stl file with PCA:
```
python PCA_mesh.py
```

### 3. Get access to x-server:
- e.g. Download .stl files and git repo to local machine

### 4. Run vtkplotlib to obtain .png from .stl
- install conda environment for py3
- run stl2img.py


### 5. Visualize with html
- Display images in a grid:
    - Find way to iterate through images:
```
http://140.247.107.10/donglai/public/js-demo/demo/display_grid.htm
```


- Option B:
    - use plots and subplots to arrange images in a grid
    - convert master plot into html.
    

# PROTOTYPE B
### 1. Run ibexHelper on .stl volume to display its abstract visualization
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