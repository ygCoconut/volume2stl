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


def load_graph(out_folder, bfs):
    G = nx.read_gpickle(out_folder+'graph-%s.obj'%(bfs))
    n0 = len(G.nodes())
#     G = ShrinkGraph_v2(G, threshold=edgTh)
    n1 = len(G.nodes())
    print('#nodes: %d -> %d'%(n0,n1))
    return G

def draw_graph(G, pos=None, save_name=''):
    if pos==None:
        pos = nx.spring_layout(G)
    nx.draw(G, pos, cmap = plt.get_cmap('jet'))
#     nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('jet'), 
#                        node_color = values, node_size = 500)
    nx.draw_networkx_labels(G, pos)
    if save_name: 
        plt.savefig(save_name)
    plt.show()
    return pos

def longest_axis_exhaustive_old(G):
    """
    Info: Search algorithm to find longest axis in a graph. 
    Tries all combinations without any heuristics.
    input: nx.Graph
    output: [list] of all paths between all nodes, [list] of length of each path
    
    """
    path_list = []
    path_length = []
    for source in G.nodes:
        for target in G.nodes:
            sh_p = nx.shortest_path(G, source, target)
            sh_p_l = nx.shortest_path_length(G, source, target)
            path_list.append(sh_p)
            path_length.append(sh_p_l)
    import pdb;pdb.set_trace()
    # %% get longest path:
    node_list = path_list[np.argmax(path_length)]
    max_length = path_length[np.argmax(path_length)]
    SG = G.subgraph(node_list)
    return G, node_list, max_length

def longest_axis_exhaustive(G, return_extra=True):
    """
    Info: Search algorithm to find longest axis in a graph. 
    Tries all combinations without any heuristics.
    input: nx.Graph
    output: Graph
        optional: [float] length of longest path, [float] thickness of longest path
    """
    print('compute longest axis (exhaustive)')
    longest_path = []
    max_length = -1
    for source in G.nodes:
        for target in G.nodes:
            try:
                length = G[source][target]['weight']
                if length > max_length:
                    max_length = length
                    longest_path = [source, target]
            except: pass
    thickness =  G[longest_path[0]][longest_path[1]]['thick']
    import pdb;pdb.set_trace()
    if True:
        print('get new node positions based on skeleton')
        nodes = G[longest_path[0]][longest_path[1]]['path']
        node_pos = np.stack(skel.get_nodes()).astype(int)

        point_cloud = np.zeros( (len(nodes), 3), int )
        for i,n in enumerate(nodes):
            point_cloud[i] = node_pos[n] #node pos allows relabel mapping trick

        if shrink == True:
            WriteH5(dendrite_folder+'node_pos_longaxis_shrinked2.h5', point_cloud)
        else:
            WriteH5(dendrite_folder+'node_pos_longaxis2.h5', point_cloud)
            
            
#         skel = ReadSkeletons(dendrite_folder,skeleton_algorithm='thinning',downsample_resolution=res,read_edges=True)[1]
            
            
    SG = G.subgraph(G[longest_path[0]][longest_path[1]]['path'])
    if return_extra == True: return SG, max_length, thickness
    else: return SG

def extract_main_axis_from_skeleton(dendrite_id, dendrite_folder, seg_fn,
                                    res, write=True, shrink=False):
    
        skel = ReadSkeletons(dendrite_folder,
                             skeleton_algorithm='thinning',
                             downsample_resolution=res,
                             read_edges=True)[1]

        print('generate dt for edge width')
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
            
        # %% get longest axis (main axis):
        SG, max_length, thickness = longest_axis_exhaustive(G)

        if write == True:
            # get array containing all selected points of the skeleton,
            # so we can display them in neuroglancer
            print('get new node positions based on skeleton')
            node_pos = np.stack(skel.get_nodes()).astype(int)

            point_cloud = np.zeros( (len(SG.nodes), 3), int )
            for i,n in enumerate(SG.nodes):
                point_cloud[i] = node_pos[n] #node pos allows relabel mapping trick

            if shrink == True:
                WriteH5(dendrite_folder+'node_pos_longaxis_shrinked2.h5', point_cloud)
            else:
                WriteH5(dendrite_folder+'node_pos_longaxis2.h5', point_cloud)
        
        return SG, G

def edge_length_and_thickness(G, node1, node2):
    return G[node1][node2]['weight'], G[node1][node2]['thick'] 
    
if __name__=='__main__':
# if opt=='4': # longest graph path
    print('start')
    dendrite_id = 1499496
    dendrite_folder = 'results/{}/'.format(dendrite_id)
    bfs = 'bfs'; modified_bfs=False 
    res = [60,64,64] # z,y,x resolution of skeleton
    seg_fn = '/n/pfister_lab2/Lab/donglai/mito/db/30um_human/seg_64nm.h5'
    
    SG, G = extract_main_axis_from_skeleton(dendrite_id, dendrite_folder,
                            seg_fn, res, write=True, shrink = True)
    # display nx graph figures
    display = False
    if display == True:
        pos = draw_graph(G, save_name='G_short.png')
        _ = draw_graph(SG, pos=pos, save_name='G_short_cut.png')
    print('done')
