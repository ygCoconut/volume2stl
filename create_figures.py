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
        data[d] = os.path.join(full_path, d+'/xyz_0_0_0_pca.stl')
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


def create_figure(path_dict, figure_path, transp_backg=False):

    for k in tqdm(path_dict):
        # Read the STL using numpy-stl
        mesh = Mesh.from_file(path_dict[k])

        fig = vpl.figure()
        fig.background_color = 'black'
        vpl.mesh_plot(mesh)

        # %% add the mitochondria to the same plot the following way:
        # mesh2 = Mesh.from_file('home/youngcoconut/Documents/snowjournal/volume2stl/stl/10010752/xyz_0_0_0_pca.stl')
        # vpl.mesh_plot(mesh2, color = 'pink', opacity=0.5) #make dendrite translucent

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
    figure_path = '/home/youngcoconut/Documents/snowjournal/volume2stl/figures/xyz_0_0_0_pca_transparent/'
    stl_path = '/home/youngcoconut/Documents/snowjournal/volume2stl/stl'
    paths_dict = get_stl_paths(stl_path)
#     paths_dict = get_stl_paths(sys.argv[1])
    create_figure(paths_dict, figure_path, transp_backg=True)
    print('done')
