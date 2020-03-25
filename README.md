# volume2stl
This repo helps you to transform a volume array to a .stl file and then visualize it.


### 1. Transform h5 to stl file

- Download https://github.com/Rhoana/3dxp 
- Run:

```
python ~/scriptsAndSoftware/repos/3dxp/PYTHON/all_stl.py --xyz /n/pfister_lab2/Lab/donglai/mito/db/30um_human/seg_64nm.h5 ./ -l listof_ids.txt
```

### 2. Rotate stl file with PCA:
```
python test_mesh.py
```

### 3. Write .json that contains path to .stl directories
```
python /n/home00/nwendt/snowjournal/make_json.py  /n/home00/nwendt/snowjournal/stl/
```

### 4. Get access to x-server:
- e.g. Download .stl files and git repo to local machine

### 5. Run vtkplotlib to obtain .png from .stl
- install conda environment for py3
- run stl2img.py


### 6. Visualize with html
- Display images in a grid:
    - Find way to iterate through images:
```
http://140.247.107.10/donglai/public/js-demo/demo/display_grid.htm
```


- Option B:
    - use plots and subplots to arrange images in a grid
    - convert master plot into html.