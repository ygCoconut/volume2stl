# https://github.com/bwoodsend/vtkplotlib
# pip install vtkplotlib

import vtkplotlib as vpl
from stl.mesh import Mesh
import numpy as np
import json
import os
import sys
from tqdm import tqdm
from PIL import Image

def get_stl_paths(full_path, stl_file='/xyz_0_0_0_pca_pair.stl' ,json_filename=''):
    dir_list = os.listdir(full_path)
    data = {}
    for d in dir_list:
        file_list = os.listdir(os.path.join(full_path, d))
        data[d] = os.path.join(full_path, d + stl_file)
    if json_filename:
        print('Making JSON for', json_filename)
        with open(json_filename+".json", 'w') as f:
             json.dump(data, f)
    return data


def _transparent_background(fig_path):
    #make black background transparent

    img = Image.open(fig_path)
    img = img.convert("RGBA")
    datas = img.getdata()

    newData = []
    for item in datas:
        if item[0] == 0 and item[1] == 0 and item[2] == 0:
            newData.append((0, 0, 0, 0)) #0 = black, last 0 is transp.
        else:
            newData.append(item)
    img.putdata(newData)
    # img.save(fig_path.replace('.png', '_transp.png'), "PNG")
    img.save(fig_path, "PNG")

def _add_normalizing_vector_point(mesh, minpt, maxpt):
    """
    This function allows you to visualize all meshes in their size relative to each other
    It is a quick simple hack: by adding 2 vector points at the same x coordinates at the
    extreme left and extreme right of the largest .stl mesh, all the meshes are displayed
    with the same scale.
    input: [mesh], minpoint coordinates, maxpoint coordinates
    output: [mesh] with 2 added coordinate points
    """
    newmesh = Mesh(np.zeros(mesh.vectors.shape[0]+2, dtype=Mesh.dtype))
    # newmesh.vectors =  np.vstack([mesh.vectors,
    #                 np.array([ [[0,maxpt,0], [0,maxpt,0], [0,maxpt,0]],
    #                            [[0,minpt,0], [0,minpt,0], [0,minpt,0]] ], float) ])
    newmesh.vectors =  np.vstack([mesh.vectors,
                    np.array([ [[0,0,maxpt], [0,0,maxpt], [0,0,maxpt]],
                               [[0,0,minpt], [0,0,minpt], [0,0,minpt]] ], float) ])

# newmesh.vectors = np.vstack([mesh.vectors,                np.array([ [[0,0,maxpt], [0,0,maxpt], [0,0,maxpt]],                          [[0,0,minpt], [0,0,minpt], [0,0,minpt]] ], float) ])

    return newmesh

def create_figure(path_dict, figure_path, path_dict2=None, pair_mapping=None, transp_backg=False):

    assert ((path_dict2 is None) + (pair_mapping is None)) != 1, \
        'please specify all kwargs or none of them'

    if pair_mapping is not None:
        for k in tqdm(pair_mapping):
            mesh= Mesh.from_file(path_dict[k[0]])
            mesh2 = Mesh.from_file(path_dict2[k[1]])

            # if debug == True:
            mesh = _add_normalizing_vector_point(mesh, 300, -300)

            fig = vpl.figure()
            fig.background_color = 'black'
            vpl.mesh_plot(mesh, color = 'pink', opacity=0.3) #make dendrite translucent
            vpl.mesh_plot(mesh2) # add second .stl to same plot

            save_path = figure_path + k[0] + '_' + k[1] + '.png'
            vpl.save_fig(save_path, magnification=5, off_screen=True, )
            if transp_backg == True: #make black background transparent
                _transparent_background(save_path)
            fig.close()

    else:
        for k in tqdm(path_dict):
            # Read the STL using numpy-stl
            mesh = Mesh.from_file(path_dict[k])

            if debug == True:
                mesh = _add_normalizing_vector_point(mesh, 300, -300)

            fig = vpl.figure()
            fig.background_color = 'black'
            vpl.mesh_plot(mesh)

            save_path = figure_path +str(k) + '.png'
            vpl.save_fig(save_path, magnification=5, off_screen=True, )
            if transp_backg == True: #make black background transparent
                _transparent_background(save_path)
            fig.close()


if __name__ == '__main__':
    # change figure path and stl path if needed
    # change background transparency if needed

    print('start')
    debug = False
    if debug == True:
        import pdb
        figure_path = '/home/youngcoconut/Documents/snowjournal/volume2stl/figures/test/'
        paths_dict = {1499496: '/home/youngcoconut/Documents/snowjournal/volume2stl/stl/1499496/xyz_0_0_0_pca.stl'}
        create_figure(paths_dict, figure_path, transp_backg=True)

    else:
        pairs = True
        if pairs == False:
            figure_path = '/home/youngcoconut/Documents/snowjournal/volume2stl/figures/test/'
            stl_path = '/home/youngcoconut/Documents/snowjournal/volume2stl/stl_mitos_dendrites_length_500/stl_dendrites/'
            paths_dict = get_stl_paths(stl_path)
    #     paths_dict = get_stl_paths(sys.argv[1])
            create_figure(paths_dict, figure_path, transp_backg=True)


        else:
            figure_path = '/home/youngcoconut/Documents/snowjournal/volume2stl/figures/pca_nocrumbs/'
            paths_dict = get_stl_paths('/home/youngcoconut/Documents/snowjournal/volume2stl/stl_mitos_dendrites_length_500/stl_dendrites_nocrumbs/')
    #     paths_dict = get_stl_paths(sys.argv[1])
            path_dict2=get_stl_paths('/home/youngcoconut/Documents/snowjournal/volume2stl/stl_mitos_dendrites_length_500/stl_mitos/')

            # mito-id, seg-id  -->   seg-id, mito-id
            idmap = np.loadtxt('mito_len500_bead_pair.txt')
            idmap = idmap[:,[1,0]].astype(np.uint32).astype(str)

            create_figure(paths_dict, figure_path, path_dict2=path_dict2, pair_mapping=idmap, transp_backg=True)

    print('done')
