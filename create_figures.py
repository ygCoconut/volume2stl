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

def get_stl_paths(full_path, json_filename=''):
    dir_list = os.listdir(full_path)
    data = {}
    for d in dir_list:
        file_list = os.listdir(os.path.join(full_path, d))
        data[d] = os.path.join(full_path, d+'/xyz_0_0_0_pca_pair.stl')
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
        # if item[0] == 255 and item[1] == 255 and item[2] == 255:
        if item[0] == 0 and item[1] == 0 and item[2] == 0:
            # newData.append((255, 255, 255, 0))
            newData.append((0, 0, 0, 0))
        else:
            # if item[0] > 150:
            #     newData.append((0, 0, 0, 255))
            # else:
            newData.append(item)
    img.putdata(newData)
    # img.save(fig_path.replace('.png', '_transp.png'), "PNG")
    img.save(fig_path, "PNG")


def create_figure(path_dict, figure_path, path_dict2=None, pair_mapping=None, transp_backg=False):

    assert ((path_dict2 is None) + (pair_mapping is None)) != 1, \
        'please specify all kwargs or none of them'


    if pair_mapping is not None:
        for k in tqdm(pair_mapping):
            mesh= Mesh.from_file(path_dict[k[0]])
            mesh2 = Mesh.from_file(path_dict2[k[1]])

            fig = vpl.figure()
            fig.background_color = 'black'

            # %% add the mitochondria to the same plot the following way:
            vpl.mesh_plot(mesh, color = 'pink', opacity=0.3) #make dendrite translucent
            vpl.mesh_plot(mesh2)

            save_path = figure_path + k[0] + k[1] + '.png'
            vpl.save_fig(save_path, magnification=5, off_screen=True, )
            # vpl.screenshot_fig(save_path, off_screen=True)

            #make black background transparent
            if transp_backg == True:
                _transparent_background(save_path)

            fig.close()

    else:
        for k in tqdm(path_dict):
            # Read the STL using numpy-stl
            mesh = Mesh.from_file(path_dict[k])

            fig = vpl.figure()
            fig.background_color = 'black'

            vpl.mesh_plot(mesh)

            save_path = figure_path +str(k) + '.png'
            vpl.save_fig(save_path, magnification=5, off_screen=True, )
            # vpl.screenshot_fig(save_path, off_screen=True)

            #make black background transparent
            if transp_backg == True:
                _transparent_background(save_path)

            fig.close()


if __name__ == '__main__':
    # change figure path and stl path if needed
    # change background transparency if needed

    print('start')
    pairs = True

    if pairs == False:
        figure_path = '/home/youngcoconut/Documents/snowjournal/volume2stl/figures/test'
        stl_path = '/home/youngcoconut/Documents/snowjournal/volume2stl/stl_mitos_dendrites_length_500/stl_dendrites/'
        paths_dict = get_stl_paths(stl_path)
#     paths_dict = get_stl_paths(sys.argv[1])
        create_figure(paths_dict, figure_path, transp_backg=True)


    else:
        figure_path = '/home/youngcoconut/Documents/snowjournal/volume2stl/figures/xyz_0_0_0_pca_mito_dendrite_pairs/'
        paths_dict = get_stl_paths('/home/youngcoconut/Documents/snowjournal/volume2stl/stl_mitos_dendrites_length_500/stl_dendrites/')
#     paths_dict = get_stl_paths(sys.argv[1])
        path_dict2=get_stl_paths('/home/youngcoconut/Documents/snowjournal/volume2stl/stl_mitos_dendrites_length_500/stl_mitos/')

        # mito-id, seg-id  -->   seg-id, mito-id
        idmap = np.loadtxt('mito_len500_bead_pair.txt')
        idmap = idmap[:,[1,0]].astype(np.uint32).astype(str)

        create_figure(paths_dict, figure_path, path_dict2=path_dict2, pair_mapping=idmap, transp_backg=True)

    print('done')
