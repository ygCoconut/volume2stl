import os
import itertools
from collections import Counter

import h5py
from scipy.spatial import KDTree
import scipy.sparse as sp
from scipy.sparse import csgraph
import numpy as np
import zmesh

from . import skel, u


def mesh_seg(seg, voxel_offset, voxel_res, segid=None):

    if segid is not None:
        seg = seg == segid

    mesher = zmesh.Mesher((1, 1, 1))
    mesher.mesh(np.swapaxes(seg, 0, 2))
    mesh_id = mesher.ids()[0]
    mesh = mesher.get_mesh(mesh_id, normals=False,
                           simplification_factor=0,
                           max_simplification_error=0)
    mesh.vertices = (mesh.vertices + voxel_offset) * voxel_res
    mesher.erase(mesh_id)
    mesher.clear()
    return {"vertices": mesh.vertices, "faces": mesh.faces.reshape(-1, 3),
            "num_vertices": len(mesh.vertices)}


def spine_head_volume_near_pts(skeleton, mesh, pts, bbox_margin,
                               skel_kdtree=None, path_inds=None,
                               spine_labels=None, cv_skel=True,
                               verbose=True, root=None,
                               simplest_cut=False):
    """Computes the spine head volume near a list of points (often synapses)"""
    if path_inds is None:
        path_inds, _ = skel.compute_path_info(skeleton, root=root,
                                              cv_skel=cv_skel)

    if spine_labels is None:
        path_radii = [skeleton.radii[inds] for inds in path_inds]
        spine_labels = skel.label_skeleton(skeleton, path_inds=path_inds,
                                           path_radii=path_radii)

    if skel_kdtree is None:
        skel_kdtree = KDTree(skeleton.vertices)

    volumes = list()
    errors = list()
    for (i, pt) in enumerate(pts):
        if verbose:
            print(f"\r#{i+1} of {len(pts)}", end="")
        try:
            # Extract spine and mesh
            spine_skel_inds, root = skel.extract_spine_near_pts(
                                    skeleton, spine_labels,
                                    [pt], path_inds=path_inds,
                                    return_all_paths=True)[0]
            spine_skel = skel.skel_by_inds(skeleton, spine_skel_inds)
            spine_root = skel.translate_skel_ids(spine_skel, [root])[0]
            spine_mesh = isolate_spine_mesh(mesh, skeleton, spine_skel_inds,
                                            kdtree=skel_kdtree,
                                            bbox_margin=bbox_margin)
    
            # Extract head
            spine_head, _ = isolate_spine_head(spine_skel, pt,
                                               spine_mesh=spine_mesh,
                                               spine_root=spine_root,
                                               cv_skel=False,
                                               simplest_cut=simplest_cut)
            sealed = simple_mesh_seal(spine_head)
            # I'm pretty sure that this helps to avoid overflow
            centered = center_mesh(sealed)
            scaled = scale_mesh(centered, [4, 4, 40], [3.58, 3.58, 40])
            volumes.append(mesh_volume(scaled))

        except Exception as e:
            if verbose:
                print("")
            print(f"EXCEPTION with pt {pt}: {e}")
            print(f"Recording volume of -1 for this pt")
            volumes.append(-1)
            errors.append((i, pt))

    if verbose:
        print("")
    return volumes, errors


def spine_and_head_near_pt(skeleton, mesh, pt, bbox_margin,
                           skel_kdtree=None, path_inds=None,
                           spine_labels=None, cv_skel=True,
                           root=None, simplest_cut=False):
    """Separates a spine and spine head (along with meshes)"""
    if path_inds is None:
        path_inds, _ = skel.compute_path_info(skeleton, root=root,
                                              cv_skel=cv_skel)

    if spine_labels is None:
        path_radii = [skeleton.radii[inds] for inds in path_inds]
        spine_labels = skel.label_skeleton(skeleton, path_inds=path_inds,
                                           path_radii=path_radii)

    if skel_kdtree is None:
        skel_kdtree = KDTree(skeleton.vertices)

    # Extract spine and mesh
    spine_skel_inds, root = skel.extract_spine_near_pts(
                            skeleton, spine_labels,
                            [pt], path_inds=path_inds,
                            return_all_paths=True,
                            kdt=skel_kdtree)[0]
    spine_skel = skel.skel_by_inds(skeleton, spine_skel_inds)
    spine_root = skel.translate_skel_ids(spine_skel, [root])[0]
    spine_mesh = isolate_spine_mesh(mesh, skeleton, spine_skel_inds,
                                    kdtree=skel_kdtree,
                                    bbox_margin=bbox_margin)
    
    # Extract head
    head_mesh, head_inds = isolate_spine_head(spine_skel, pt,
                                              spine_mesh=spine_mesh,
                                              spine_root=spine_root,
                                              cv_skel=False,
                                              simplest_cut=simplest_cut)

    head_skel = skel.skel_by_inds(spine_skel, head_inds)

    return spine_skel, spine_mesh, head_skel, head_mesh


def isolate_spine_head(spine_skel, synapse_pt, spine_mesh=None,
                       skel_kdtree=None, spine_root=None, cv_skel=False,
                       simplest_cut=False):
    """
    Spine head extraction using a spine mesh, a spine skeleton, and a synapse
    point on the spine.
    """
    if skel_kdtree is None:
        skel_kdtree = KDTree(spine_skel.vertices)

    syn_skel_pt = skel_kdtree.query(synapse_pt)[1]

    if spine_root is None:
        spine_root, path_inds = skel.find_furthest_pt(spine_skel, syn_skel_pt)
    else:
        path_inds = skel.paths_containing_pair(
                        spine_skel, spine_root, syn_skel_pt,
                        max_len=True, cv_skel=cv_skel)

    # Ensuring that the root node is at the beginning of the path
    # for the cut fn
    if np.nonzero(path_inds == spine_root)[0] > len(path_inds) // 2:
        path_inds = path_inds[::-1]
    cut_ind, local_cut_ind = skel.medium_radius_cut_pt(
                                 spine_skel, path_inds,
                                 simplest_vers=simplest_cut)
    remaining_pts = path_inds[local_cut_ind:]

    # Restricting mesh filtering to a single path to avoid "split path" problem
    # -> this wasn't worth it, it fixed the split spine cases, but made others worse
    #    by artificially merging them
    #path_skel = skel.skel_by_inds(spine_skel, path_inds)
    #path_kdtree = KDTree(path_skel.vertices)
    #path_rem_pts = skel.translate_skel_ids(path_skel, remaining_pts)

    if spine_mesh is None:
        return remaining_pts
    else:
        return filter_mesh_by_node_prox(
                   spine_mesh, spine_skel, remaining_pts,
                   kdtree=skel_kdtree), remaining_pts


def isolate_spine_mesh(mesh, skel, spine_nodes, kdtree=None, bbox_margin=None):
    """
    Spine mesh extraction using a cell's skeleton and the nodes of that
    skeleton that describe the spine.
    """
    return filter_mesh_by_node_prox(mesh, skel, spine_nodes,
                                    kdtree=kdtree, bbox_margin=bbox_margin)


def filter_mesh_by_node_prox(mesh, skel, nodes, kdtree=None, bbox_margin=None):
    """
    Finds the subset of a mesh whose closest points on the skeletons are listed
    by the node indices within nodes
    """
    if kdtree is None:
        kdtree = KDTree(skel.vertices)

    if bbox_margin is not None:
        skel_coords = skel.vertices[nodes]
        bbox_min, bbox_max = u.make_bbox(skel_coords, bbox_margin)
        mesh = mesh_within_bbox(mesh, bbox_min, bbox_max)

    node_set = set(nodes)
    new_inds = np.array([i for (i, v) in enumerate(mesh["vertices"])
                         if kdtree.query(v)[1] in node_set])
    return mesh_by_inds(mesh, new_inds)


def mesh_within_bbox(mesh, bbox_min, bbox_max):
    valid_inds = inds_within_bbox(mesh["vertices"], bbox_min, bbox_max)
    return mesh_by_inds(mesh, valid_inds)


def mesh_by_inds(mesh, inds):
    assert len(inds) > 0, "empty inds"
    new_verts = mesh["vertices"][inds]

    ind_map = np.empty((max(inds)+1,), dtype=inds.dtype)
    ind_map[inds] = np.arange(len(inds))
    explicit_faces = mesh["faces"].reshape((-1, 3))
    face_inds = np.all(np.isin(explicit_faces, inds), axis=1)
    new_faces = ind_map[explicit_faces[face_inds]].ravel()

    return dict(vertices=new_verts, faces=new_faces,
                num_vertices=len(new_verts))


def inds_within_bbox(vertices, bbox_min, bbox_max):
    overmin = np.all(vertices >= bbox_min, axis=1)
    undermax = np.all(vertices <= bbox_max, axis=1)

    return np.nonzero(overmin & undermax)[0]


def simple_mesh_seal(mesh):
    """Seals a mesh by adding one vertex at the centroid of each boundary"""
    components = find_mesh_boundary_components(mesh)
    for comp in components:
        new_pt = np.mean(mesh["vertices"][comp], axis=0)
        mesh = attach_new_pt(mesh, new_pt, boundary=comp)

    return mesh


def find_mesh_boundary_vertices(mesh):
    boundary_edges = find_mesh_boundary_edges(mesh)
    return list(set(itertools.chain.from_iterable(boundary_edges)))


def find_mesh_boundary_components(mesh):
    boundary_edges = find_mesh_boundary_edges(mesh)

    return edge_components(boundary_edges)


def edge_components(edges):
    rows, cols = zip(*edges)

    ids = list(set(rows + cols))
    idmap = {v: i for (i, v) in enumerate(ids)}
    rows = [idmap[v] for v in rows]
    cols = [idmap[v] for v in cols]

    vals = np.ones((len(rows),), dtype=np.uint8)
    num_ids = max(max(rows), max(cols)) + 1
    g = sp.coo_matrix((vals, (rows, cols)), shape=(num_ids, num_ids)).tocsr()

    num_comps, labels = csgraph.connected_components(
                            g, directed=False, return_labels=True)
    graph_comps = [np.nonzero(labels == i)[0] for i in range(num_comps)]

    return [[ids[v] for v in comp] for comp in graph_comps]


def find_mesh_boundary_edges(mesh):
    faces = mesh["faces"].reshape((-1, 3))
    edges = [tuple(sorted(edge)) for edge in
             np.vstack((faces[:, :2],
                        faces[:, 1:],
                        faces[:, [2, 0]]))]

    counter = Counter(edges)
    return [edge for (edge, count) in counter.items() if count == 1]


def attach_new_pt(mesh, new_pt, boundary=None):
    new_verts = np.vstack((mesh["vertices"], new_pt))
    new_i = len(mesh["vertices"])

    boundary_edges = find_mesh_boundary_edges(mesh)
    if boundary is not None:
        bset = set(boundary)
        boundary_edges = [edge for edge in boundary_edges
                          if (edge[0] in bset) and (edge[1] in bset)]

    new_faces = np.array(assemble_consistent_faces(boundary_edges, new_i))
    new_faces = flip_if_mostly_inwards(new_faces, new_verts, new_pt)
    all_new_faces = np.hstack((mesh["faces"], new_faces.ravel()))

    return dict(vertices=new_verts, faces=all_new_faces,
                num_vertices=len(new_verts))


def assemble_consistent_faces(edges, new_i):
    vertex_order = dfs_order_vertices(edges)
    return [[vertex_order[i-1], vertex_order[i], new_i]
            for i in range(len(vertex_order))]


def dfs_order_vertices(edges):
    rows, cols = zip(*edges)

    ids = list(set(rows + cols))
    idmap = {v: i for (i, v) in enumerate(ids)}
    rows = [idmap[v] for v in rows]
    cols = [idmap[v] for v in cols]

    vals = np.ones((len(rows),), dtype=np.uint8)
    num_ids = max(max(rows), max(cols)) + 1
    g = sp.coo_matrix((vals, (rows, cols)), shape=(num_ids, num_ids)).tocsr()

    return [ids[i] for i in csgraph.depth_first_order(g, 0, directed=False)[0]]


def flip_if_mostly_inwards(faces, vertices, new_pt):
    bad_faces = find_bad_face_normals(faces, vertices, new_pt)
    if sum(bad_faces) > len(faces) // 2:
        faces = faces[:, [1, 0, 2]]
    return faces


def fix_bad_face_normals(faces, vertices, new_pt):
    bad_faces = find_bad_face_normals(faces, vertices, new_pt)
    faces = faces.copy()
    faces[bad_faces] = faces[bad_faces][:, [1, 0, 2]]

    return faces


def find_bad_face_normals(faces, vertices, new_pt):
    centroid = np.mean(vertices, axis=0)
    normals = compute_normals(faces, vertices)

    return (normals @ (new_pt - centroid)) < 0


def compute_normals(faces, vertices):
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    return np.cross(v1-v0, v2-v0)


def mesh_volume(mesh, scale=1000):

    vs = mesh["vertices"] / scale
    fs = mesh["faces"].reshape((-1, 3))

    v0 = vs[fs[:, 0]]
    v1 = vs[fs[:, 1]]
    v2 = vs[fs[:, 2]]

    return np.abs(np.sum(np.cross(v0, v1) * v2 / 6.))


def center_mesh(mesh):
    centroid = np.mean(mesh["vertices"], axis=0)
    return dict(vertices=mesh["vertices"]-centroid,
                faces=mesh["faces"], num_vertices=mesh["num_vertices"])


def scale_mesh(mesh, old_res, new_res):
    verts = (mesh["vertices"] / old_res) * new_res
    return dict(vertices=verts, faces=mesh["faces"],
                num_vertices=mesh["num_vertices"])


def write_mesh(mesh, filename):
    if os.path.isfile(filename):
        os.remove(filename)

    with h5py.File(filename) as f:
        f.create_dataset("vertices", data=mesh["vertices"])
        f.create_dataset("faces", data=mesh["faces"])
        f.create_dataset("num_vertices", data=mesh["num_vertices"])


def read_mesh(filename):
    assert os.path.isfile(filename)

    with h5py.File(filename) as f:
        vertices = f["vertices"][()]
        faces = f["faces"][()]
        num_vertices = f["num_vertices"][()]

    return dict(vertices=vertices, faces=faces, num_vertices=num_vertices)
