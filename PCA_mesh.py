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

def stl_PCA(path_dict, path_dict2=None, pair_mapping=None):
    
    assert ((path_dict2 is None) + (pair_mapping is None)) != 1, \
        'please specify all kwargs or none of them'

    if pair_mapping is not None:
        for k in tqdm(pair_mapping):
            test_mesh= mesh.Mesh.from_file(path_dict[k[0]])
            test_mesh2 = mesh.Mesh.from_file(path_dict2[k[1]])
            
            mesh_arr = np.asarray(test_mesh)
            mesh_arr2 = np.asarray(test_mesh2)
            num_points = mesh_arr.shape[0]
            num_points2 = mesh_arr2.shape[0]
            mesh_stacked =  np.vstack([mesh_arr[:,0:3], mesh_arr[:,3:6], \
                              mesh_arr[:,6:9]])
            mesh_stacked2 =  np.vstack([mesh_arr2[:,0:3], mesh_arr2[:,3:6], \
                              mesh_arr2[:,6:9]])

            pca = PCA(n_components=3)
            pca.fit(mesh_stacked)
            new_stack = pca.transform(mesh_stacked)            
            new_stack2 = pca.transform(mesh_stacked2)            

            new_mesh_arr = np.hstack([new_stack[0:num_points,:], \
                            new_stack[num_points:2*num_points,:], \
                            new_stack[2*num_points:3*num_points,:]])

            new_mesh_arr2 = np.hstack([new_stack2[0:num_points2,:], \
                            new_stack2[num_points2:2*num_points2,:], \
                            new_stack2[2*num_points2:3*num_points2,:]])

            data = np.zeros(num_points, dtype=mesh.Mesh.dtype)
            data2 = np.zeros(num_points2, dtype=mesh.Mesh.dtype)
            data['vectors'] = new_mesh_arr.reshape(num_points, 3,3)
            data2['vectors'] = new_mesh_arr2.reshape(num_points2, 3,3)
            new_stl = mesh.Mesh(data)
            new_stl2 = mesh.Mesh(data2)
            new_stl.save( path_dict[k[0]].replace('.stl', '_pca.stl') )
            new_stl2.save( path_dict2[k[1]].replace('.stl', '_pca.stl') )

    else:
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
    pairs = True

    if pairs == False:
        paths_dict = get_stl_paths('/n/home00/nwendt/snowjournal/volume2stl/stl')
        stl_PCA(paths_dict)
        
    else:
        dendrite_paths = get_stl_paths('/n/home00/nwendt/snowjournal/volume2stl/stl_mitos_dendrites_length_500/stl_dendrites')
        mito_paths = get_stl_paths('/n/home00/nwendt/snowjournal/volume2stl/stl_mitos_dendrites_length_500/stl_mitos')
        
        # mito-id, seg-id  -->   seg-id, mito-id
        idmap = np.loadtxt('/n/pfister_lab2/Lab/donglai/mito/db/30um_human/mito_len500_bead_pair.txt')
        idmap = idmap[:,[1,0]].astype(np.uint32).astype(str)

        stl_PCA(dendrite_paths, path_dict2=mito_paths, pair_mapping=idmap)
        
    print('done')