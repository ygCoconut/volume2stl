import numpy as np
from stl import mesh
from sklearn.decomposition import PCA

file_path = 'vol5/stl/1/0_0_0.stl'
file_path='./stl/11150538/xyz_0_0_0.stl'

test_mesh = mesh.Mesh.from_file(file_path)
mesh_arr = np.asarray(test_mesh)
num_points = mesh_arr.shape[0]
mesh_stacked =  np.vstack([mesh_arr[:,0:3], mesh_arr[:,3:6], \
                  mesh_arr[:,6:9]])
pca = PCA(n_components=3)
pca.fit(mesh_stacked)
new_stack = pca.transform(mesh_stacked)
new_mesh_arr = np.hstack([new_stack[0:num_points,:], \
                new_stack[num_points:2*num_points,:], \
                new_stack[2*num_points:3*num_points,:]])

data = np.zeros(num_points, dtype=mesh.Mesh.dtype)
data['vectors'] = new_mesh_arr.reshape(num_points, 3,3)
# import pdb; pdb.set_trace()
new_stl = mesh.Mesh(data)
new_stl.save('new.stl')

