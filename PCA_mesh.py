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
#         for k in tqdm(pair_mapping):
#             list:
#             test_mesh= mesh.Mesh.from_file(path_dict[k[0]])
#             test_mesh2 = mesh.Mesh.from_file(path_dict2[k[1]])
            
#             dict:
        for k,values in tqdm(pair_mapping.iteritems()):
#             import pdb; pdb.set_trace()
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
            new_stl.save( path_dict[k].replace('.stl', '_pca_pair.stl') )
            
            for v in values:
                test_mesh2 = mesh.Mesh.from_file(path_dict2[str(v)])
                mesh_arr2 = np.asarray(test_mesh2)
                num_points2 = mesh_arr2.shape[0]
                mesh_stacked2 =  np.vstack([mesh_arr2[:,0:3], mesh_arr2[:,3:6], \
                                  mesh_arr2[:,6:9]])


                new_stack2 = pca.transform(mesh_stacked2)            


                new_mesh_arr2 = np.hstack([new_stack2[0:num_points2,:], \
                                new_stack2[num_points2:2*num_points2,:], \
                                new_stack2[2*num_points2:3*num_points2,:]])

                data2 = np.zeros(num_points2, dtype=mesh.Mesh.dtype)
                data2['vectors'] = new_mesh_arr2.reshape(num_points2, 3,3)
                new_stl2 = mesh.Mesh(data2)
                new_stl2.save( path_dict2[str(v)].replace('.stl', '_pca_pair.stl') )

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
    """
    Returns PCA-rotated instances as .stl files, which allows for maximization of the surface that will be rendered to a picture when the .stl file is plotted.

    There are 2 plotting scenarios considered:
    pairs == False: You just want to render individual .stl files.
    pairs == True: You want to render muliple instances with the same rotation, e.g. mitos contained in dendrites should have the same oritentation as the dendrite they belong to.

    if pairs is false, each .stl file can be calculated individually with its own pca
    if pairs is true, you need to specify an array containing the matched pairs that should be rotated according to the same pca. The first of the 2 paths specified for the pairs corresponds to the fist column of the matched pairs array and will be the object of the pca, while the same rotation will be applied to the second pair element. 
    """

    print('start')
    pairs = True

    if pairs == False:
        dendrite_paths = get_stl_paths('/n/home00/nwendt/snowjournal/volume2stl/stl_mitos_dendrites_length_500/stl_dendrites_nocrumbs')
        stl_PCA(dendrite_paths)

#         mito_paths = get_stl_paths('/n/home00/nwendt/snowjournal/volume2stl/stl_mitos_dendrites_length_500/stl_mitos')
#         stl_PCA(mito_paths)
        
    else:
        dendrite_paths = get_stl_paths('/n/home00/nwendt/snowjournal/volume2stl/stl')
        mito_paths = get_stl_paths('/n/home00/nwendt/snowjournal/volume2stl/stl_all_mitos')
        
        # mito-id, seg-id  -->   seg-id, mito-id
#         list:
#         idmap = np.loadtxt('/n/pfister_lab2/Lab/donglai/mito/db/30um_human/mito_len500_bead_pair.txt')
#         idmap = idmap[:,[1,0]].astype(np.uint32).astype(str)
        
#     dict:
        with open('data/lut_dendrite_mito_237.json') as json_file:
            idmap = json.load(json_file)
        
        stl_PCA(dendrite_paths, path_dict2=mito_paths, pair_mapping=idmap)
        
    print('done')