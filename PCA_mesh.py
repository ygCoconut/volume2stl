import numpy as np
from stl import mesh
from sklearn.decomposition import PCA
import json
import os
import sys
from tqdm import tqdm

def get_stl_paths(full_path, json_filename=''):
    dir_list = os.listdir(full_path)
    data = {}
    for d in dir_list:
        file_list = os.listdir(os.path.join(full_path, d))
        if 'new.stl' in file_list:
            data[d] = os.path.join(full_path, d+'/new.stl')
        else:
            data[d] = os.path.join(full_path, d+'/xyz_0_0_0.stl')
    if json_filename:
        print('Making JSON for', json_filename)
        with open(json_filename+".json", 'w') as f:
             json.dump(data, f)
    return data

def stl_PCA(path_dict):

    for k in tqdm(path_dict):
        test_mesh = mesh.Mesh.from_file(path_dict[k])
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
        new_stl = mesh.Mesh(data)
        new_stl.save( path_dict[k].replace('.stl', '_pca.stl') )


if __name__ == '__main__':
    
    print('start')
    paths_dict = get_stl_paths('/n/home00/nwendt/snowjournal/volume2stl/stl')
#     paths_dict = get_stl_paths(sys.argv[1])
    stl_PCA(paths_dict)
    print('done')