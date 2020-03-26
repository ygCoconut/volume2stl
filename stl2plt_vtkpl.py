# https://github.com/bwoodsend/vtkplotlib
# pip install vtkplotlib

import vtkplotlib as vpl
from stl.mesh import Mesh
import numpy as np

# vpl.is_interactive
vpl.auto_figure(False)

path = "./stl/11150538/new.stl"
path = "./new.stl"
# path = "./stl/11150538/xyz_0_0_0.stl"

# Read the STL using numpy-stl
mesh = Mesh.from_file(path)


# reshape to (n, 3, 3)
# np_mesh = np.array(mesh).reshape((-1, 3, 3))

# np_mesh.shape

# Plot the mesh
# vpl.mesh_plot(np_mesh)



#vpl.auto_figure(True)
fig = vpl.figure()
# vpl.mesh_plot(mesh, color='lightblue', fig=fig)
vpl.mesh_plot(mesh, color='pink', fig=fig)
vpl.mesh_plot(mesh, fig=fig)
# vpl.show();

# Show the figure
# vpl.save_fig('test.png')

vpl.save_fig('testfig3.png', magnification=5, off_screen=True, fig=fig)
#vpl.screenshot_fig(off_screen=True)


# If you want the brick to be blue you can replace the mesh_plot with

# vpl.mesh_plot(mesh, color="blue")
