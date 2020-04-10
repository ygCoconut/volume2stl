import os,sys

# add ibexHelper path , NEEDS TO BE ABSOLUTE !!!
# sys.path.append('/home/donglai/lib/ibex_fork/ibexHelper')
sys.path.append('/n/home00/nwendt/scriptsAndSoftware/repos/ibexHelper')
from ibexHelper.skel import CreateSkeleton, ReadSkeletons
from ibexHelper.util import GetBbox, ReadH5, WriteH5
from ibexHelper.skel2graph import GetGraphFromSkeleton
from ibexHelper.graph import ShrinkGraph_v2, GetNodeList, GetEdgeList
from ibexHelper.graph2x import Graph2H5
import h5py
import numpy as np
import networkx as nx
from scipy.ndimage.morphology import distance_transform_cdt
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm

# 1. I/O
def blockPrint(): # Disable print
    sys.stdout = open(os.devnull, 'w')
def enablePrint(): # Restore print
    sys.stdout = sys.__stdout__

def write_skel_coordinates(skel, G, save_path='node_pos.h5'):
    # get array containing all selected points of the skeleton,
    # so we can display them in neuroglancer
    print('get new node positions based on skeleton')
    node_pos = np.stack(skel.get_nodes()).astype(int)

    point_cloud = np.zeros( (len(G.nodes), 3), int )
    for i,n in enumerate(G.nodes):
        point_cloud[i] = node_pos[n] #node pos allows relabel mapping trick

    WriteH5(save_path, point_cloud)


# def skel2graph(dendrite_id, dendrite_folder, seg, res, shrink=False):
def skeleton2graph(dendrite_id, dendrite_folder, seg_fn, res, shrink=False):
    print('generate graph from skeleton ..')
    skel = ReadSkeletons(dendrite_folder,
                     skeleton_algorithm='thinning',
                     downsample_resolution=res,
                     read_edges=True)[1]

    seg = ReadH5(seg_fn, 'main')
    sz = seg.shape
#     bb = GetBbox(seg>0)
    bb = GetBbox(seg==int(dendrite_id))

    seg_b = seg[bb[0]:bb[1]+1,bb[2]:bb[3]+1,bb[4]:bb[5]+1]
    dt = distance_transform_cdt(seg_b, return_distances=True)
    new_graph, wt_dict, th_dict, ph_dict = GetGraphFromSkeleton(skel,
        dt=dt, dt_bb=[bb[x] for x in [0,2,4]], modified_bfs=modified_bfs)

    edge_list = GetEdgeList(new_graph, wt_dict, th_dict, ph_dict)
    G = nx.Graph(shape=sz)
    # add edge attributes
    G.add_edges_from(edge_list)

    if shrink == True: #shrink graph size
        edgTh = [40,1] # threshold
        n0 = len(G.nodes())
        G = ShrinkGraph_v2(G, threshold=edgTh)
        n1 = len(G.nodes())
        print('#nodes: %d -> %d'%(n0,n1))
        
    return G, skel
    

# 2. Graph Operations 
def longest_axis_dijkstra(G):
#     Todo: implement with nx.dijkstra_path
    pass

def longest_axis_exhaustive(G):
    """
    Info: Search algorithm to find longest axis in a graph. 
    Tries all combinations without any heuristics.
    input: nx.Graph
    output: [list] of all paths between all nodes, [list] of length of each path
    
    """
    path_list = []
    path_length = []
    node_pairs = []
    for source in G.nodes:
        for target in G.nodes:
            sh_p = nx.shortest_path(G, source, target)
            sh_p_l = nx.shortest_path_length(G, source, target)
            path_list.append(sh_p)
            path_length.append(sh_p_l)
            node_pairs.append([source, target])
            
    nbunch = path_list[np.argmax(path_length)]
    SG = G.subgraph(nbunch)
    return SG, np.max(path_length), nbunch, node_pairs[np.argmax(path_length)]


def extract_main_axis_from_skeleton(dendrite_id, dendrite_folder, seg_fn,
                                    res, write=True, shrink=False):

    G, skel = skel2graph(dendrite_id, dendrite_folder, seg_fn, res, shrink=shrink)

    # %% get longest axis (main axis):
    SG, nodes, path_length, endnodes = longest_axis_exhaustive(G)

    if write == True:
        if shrink == True:
            write_skel_coordinates(skel, G,
                save_path= dendrite_folder+'node_pos_longaxis_shrinked2.h5')
        else:
            write_skel_coordinates(skel, G,
                save_path= dendrite_folder+'node_pos_longaxis2.h5')

    return SG, endnodes

def edge_length_and_thickness(G, node1, node2):
    length = nx.shortest_path_length(G, node1, node2, weight='weight')
    # assumption for thickness: path is unique
    thickness = nx.shortest_path_length(G, node1, node2, weight='thick')
    print(thickness, length)
    print(G.nodes)
    #length can be 0 if only 1 node
    # todo: add root node 
    try:
        thickness /= length
        return length, thickness
    except:
        return 0, 0

def get_spines(Gsp, mainaxis_nodes):
    "remove main-axis nodes from graph, return list of unconnected subgraphs"
    for node in list(mainaxis_nodes):
        Gsp.remove_node(node)

    graphs = list(nx.connected_component_subgraphs(G))
    return graphs # graph_list

if __name__=='__main__':
# if opt=='4': # longest graph path
    print('start')
    
    bfs = 'bfs'; modified_bfs=False 
    res = [60,64,64] # z,y,x resolution of skeleton
    seg_fn = '/n/pfister_lab2/Lab/donglai/mito/db/30um_human/seg_64nm.h5' # crumbs
    seg_fn = '/n/pfister_lab2/Lab/nils/snowproject/seg_64nm_maindendrite.h5' # no crumbs
    
    seg = np.array(h5py.File(seg_fn, 'r')['main'])
#     dendrite_ids = np.loadtxt('mito_len500_bead_pair.txt', int)[:,1]
    dendrite_ids = np.loadtxt('seg_spiny_v2.txt', int)
    lookuptable = np.zeros((dendrite_ids.shape[0], 5))
#     did = 1499496
    for i, did in enumerate(tqdm(dendrite_ids)):
        blockPrint()
        dendrite_folder = 'results_spines/{}/'.format(did)

        CreateSkeleton(seg==did, dendrite_folder, res, res)
        # get main axis:
#         G, nodeends = extract_main_axis_from_skeleton(did, dendrite_folder,
#                                 seg_fn, res, write=True, shrink = False)
        
        # load skeleton of given seg id, return it and its graph
        G, skel = skeleton2graph(did, dendrite_folder, seg_fn, res)
        # %% get longest axis 
        main_G, _, _, endnodes = longest_axis_exhaustive(G)
        # get average thickness and length
        length, thickness = edge_length_and_thickness(G, endnodes[0], endnodes[1])

        #get spines of the main axis dendrite:
        len_spines = []
        thick_spines = []
        spines_G_list = get_spines(G, main_G.nodes)
        for spine in spines_G_list:
            spine_mainG, _, _, endnodes_spine = longest_axis_exhaustive(spine)
            ln_sp, tk_sp = edge_length_and_thickness(spine, endnodes_spine[0], endnodes_spine[1])
#             len_spines.append(ln_sp)
#             thick_spines.append(tk_sp)
            len_spines = ln_sp
            thick_spines = tk_sp

        lookuptable[i] = [did, thickness, length, 
                          np.mean(thick_spines), np.mean(len_spines)]
        np.savetxt('lookuptable.txt', lookuptable,
            header = 'dendrite id, thickness, length, spines_avg_thickness, spines_avg_length',
                   fmt=['%d', '%f', '%f', '%f', '%f'])
        
        enablePrint()
    
    lot_s = lookuptable[np.argsort(-lookuptable[:,1])]
    np.savetxt('lookuptable.txt', lot_s,
            header = 'dendrite id, thickness, length, spines_avg_thickness, spines_avg_length',
            fmt=['%d', '%f', '%f', '%f', '%f'])
        
    print('done')

# use the following to print all the thicknesses as a list
#     ','.join([str(float(i)) for i in lot_s[:,1]])
# ','.join([str(int(i)) for i in lot_s[:,0]]) #get ids
    
    