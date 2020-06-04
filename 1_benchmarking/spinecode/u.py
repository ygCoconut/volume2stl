import os
import re

import numpy as np
import h5py
from scipy.spatial import KDTree  # aliasing this
import matplotlib.pyplot as plt

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]


POS_REGEXP = re.compile(r"\[([0-9]*) +([0-9]*) +([0-9]*)\]")
TUP_REGEXP = re.compile(r"\(([0-9]*), +([0-9]*), +([0-9]*)\)")


def read_cvol_around_pt(cvol, pt, bbox_width=(125, 125, 100),
                        return_bbox=False):
    cvol.bounded=False
    cvol.fill_missing=True

    bbox = make_bbox([pt], bbox_width)
    bbox = (bbox[0].astype(int), bbox[1].astype(int))

    cutout = cvol[bbox[0][0]:bbox[1][0],
                  bbox[0][1]:bbox[1][1],
                  bbox[0][2]:bbox[1][2]]

    if return_bbox:
        return cutout, bbox
    else:
        return cutout


def parse_pos(pos, regexp=POS_REGEXP):
    """Extracts a tuple from the coordinates from the DataAnalysisLink"""
    m = regexp.match(pos)
    return tuple(map(int, m.groups()))


def make_bbox(pts, bbox_margin):
    pts = np.array(pts)
    if len(pts.shape) == 1 or pts.shape[1] == 1:
        # single pt
        return pts - bbox_margin, pts + bbox_margin
    else:
        return np.min(pts, 0) - bbox_margin, np.max(pts, 0) + bbox_margin


def scale_to_nm(coord, voxel_res=[4, 4, 40]):
    return (coord[0]*voxel_res[0],
            coord[1]*voxel_res[1],
            coord[2]*voxel_res[2])


def scale_to_vx(coord, voxel_res=[4, 4, 40], asint=True):
    vx_coord = (coord[0]/voxel_res[0],
                coord[1]/voxel_res[1],
                coord[2]/voxel_res[2])

    if asint:
        vx_coord = tuple(map(int, vx_coord))

    return vx_coord


def logspace_bins(arr, n, eps=1e-10):
    return np.logspace(np.log10(arr.min())-eps,
                       np.log10(arr.max())+eps,
                       num=n)
