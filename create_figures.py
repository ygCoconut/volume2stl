# https://github.com/bwoodsend/vtkplotlib
# pip install vtkplotlib

import vtkplotlib as vpl
from stl.mesh import Mesh
import numpy as np
import json
import os
import sys
from tqdm import tqdm

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


def write_figure(path_dict, figure_path):

    for k in tqdm(path_dict):
        # Read the STL using numpy-stl
        mesh = Mesh.from_file(path_dict[k])

        vpl.auto_figure(False)
        fig = vpl.figure()
        # vpl.mesh_plot(mesh, color='pink', fig=fig)
        vpl.mesh_plot(mesh, fig=fig)

        save_path = figure_path +str(k) + '.png'
        vpl.save_fig(save_path, magnification=5, off_screen=True, fig=fig)

def transparent_background(fig_path):
    pass
    # make black background transparent
    # import PIL



if __name__ == '__main__':

    print('start')
    figure_path = '/home/youngcoconut/Documents/snowjournal/volume2stl/figures/xyz_0_0_0_pca/'
    stl_path = '/home/youngcoconut/Documents/snowjournal/volume2stl/stl'
    paths_dict = get_stl_paths(stl_path)
#     paths_dict = get_stl_paths(sys.argv[1])
    write_figure(paths_dict, figure_path)
    print('done')
