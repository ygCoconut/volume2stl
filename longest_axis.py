import os,sys

# add ibexHelper path , NEEDS TO BE ABSOLUTE !!!
# sys.path.append('/home/donglai/lib/ibex_fork/ibexHelper')
sys.path.append('/n/home00/nwendt/scriptsAndSoftware/repos/ibexHelper')
from ibexHelper.skel import CreateSkeleton, ReadSkeletons
from ibexHelper.util import GetBbox, ReadH5, WriteH5
from ibexHelper.skel2graph import GetGraphFromSkeleton
# from ibexHelper.graph import ShrinkGraph_v2, GetNodeList, GetEdgeList
from ibexHelper.graph import ShrinkGraph_v2, GetEdgeList
# from ibexHelper.graph2x import Graph2H5
import h5py
import numpy as np
import networkx as nx
from scipy.ndimage.morphology import distance_transform_cdt
import matplotlib.pyplot as plt
from tqdm import tqdm

import argparse

# 1. I/O

res = [30,32,32] # z,y,x resolution of skeleton
seg_fn = '/n/pfister_lab2/Lab/donglai/mito/db/30um_human/seg_32nm.h5'#replacement
dendrite_ids = [11, 12, 13, 16, 17, 18, 20, 24, 25, 26] #handpicked benchm. exps
output_folder = '/n/pfister_lab2/Lab/nils/snowproject/hum_segv2/benchmarks/'

def get_args():
    parser = argparse.ArgumentParser(description='Compute the skeleton + graph + metrics of segments')
    parser.add_argument('-seg', type=str, default='~/my_ndarray.h5',
                       help='path to segmentation volume')

    parser.add_argument('-res', type=str, default='30:32:32',
                       help='zyx resolution of the segmented volume')
    # either input the pre-compute prediction score
    parser.add_argument('-ids', type=str, default='11:12:13:16:17:18:20:24:25:26',
                       help='ids of the segments to process')
    # or avg input affinity/heatmap prediction
    parser.add_argument('-out', type=str, default='/n/pfister_lab2/Lab/nils/snowproject/hum_segv2/benchmarks/',
                       help='output folder to save all computations and results')
    parser.add_argument('-cs', type=int, default=0, help='create skeleton if 1')
    
    args = parser.parse_args()
    return args

def blockPrint(): # Disable print
    sys.stdout = open(os.devnull, 'w')
def enablePrint(): # Restore print
    sys.stdout = sys.__stdout__

def print_lot(lot_s):
    # use the following to print all the thicknesses and all the ids in a 
    # look_up_table.txt as a list
    print(','.join([str(int(i)) for i in lot_s[:,0]]) ) #get ids
    print(','.join([str(float(i)) for i in lot_s[:,1]]) ) #get th

def plot_Graph(G, nodepos=None, return_pos=False):
    if nodepos == None:
        nodepos = nx.kamada_kawai_layout(G)
    endnodes = [x for x in G.nodes() if G.degree(x)==1]
    endnodepos = {k:v for k,v in nodepos.items() if k in endnodes}
    # need to debug: labels dict not filtering..
#     labels = nx.get_edge_attributes(G,'weight')
#     labels_new = {k:v for k,v in labels.items() if k in G.edges(endnodes)}
#     nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)
    # only draw nodelabels for endnodes
    nx.draw_networkx_labels(G.subgraph(endnodes), nodepos)
    nx.draw(G,nodepos)
    if return_pos: 
        return pos
    
def write_skel_coordinates(skel, G, save_path='node_pos.h5'):
    # get array containing all selected points of the skeleton,
    # so we can display them in neuroglancer
    print('get new node positions based on skeleton')
    node_pos = np.stack(skel.get_nodes()).astype(int)

    point_cloud = np.zeros( (len(G.nodes), 3), int )
    for i,n in enumerate(G.nodes):
        point_cloud[i] = node_pos[n] #node pos allows relabel mapping trick

    WriteH5(save_path, point_cloud)

def skeleton2graph(dendrite_id, dendrite_folder, seg, res, shrink=False):
    print('generate graph from skeleton ..')
    skel = ReadSkeletons(dendrite_folder, skeleton_algorithm='thinning',
                         downsample_resolution=res, read_edges=True)[1]

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
def search_longest_path_efficient(G, weight='weight'):
    """
    Info: Search algorithm to find longest path in a graph. 
    Breadth first search (BFS) used twice for acyclic graphs.
    BFS over all endnodes used for cyclic graphs
    input: nx.Graph
    optional: edge paramater weight -> either 'weight' or 'thick'
    output: [list] of all paths between all nodes, [list] of length of each path
    """
    
    def _bfs_tree(G):
        # twice BFS in undirected acyclic graph (Tree) to find endnodes
        # bfs for longest path is performed in Directed Tree
        endnodes = [x for x in G.nodes() if G.degree(x)==1]
        DiTree1 = nx.traversal.bfs_tree(G, endnodes[0])
        SG1 = G.subgraph(nx.dag_longest_path(DiTree1, weight=weight))
        new_root = [x for x in SG1.nodes() if G.degree(x)==1 and x != endnodes[0]][0]
        DiTree2 = nx.traversal.bfs_tree(SG1, new_root)
        SG2 = G.subgraph(nx.dag_longest_path(DiTree2, weight=weight))
        return SG2, nx.dag_longest_path_length(DiTree2)
    
    def _exhaustive(G):
        # Finds longest shortest path by iterating over all endnodes
        path_list = []
        path_length = []
#         node_pairs = []
        endnodes = [x for x in G.nodes() if G.degree(x)==1]
        for source in endnodes:
            for target in endnodes:
                sh_p = nx.shortest_path(G, source, target, weight=weight)
                sh_p_l = nx.shortest_path_length(G, source, target, weight=weight)
                path_list.append(sh_p)
                path_length.append(sh_p_l)
#                 node_pairs.append([source, target])

        nbunch = path_list[np.argmax(path_length)]
        SG = G.subgraph(nbunch)
        return SG, np.max(path_length)#, nbunch, node_pairs[np.argmax(path_length)]

    if nx.tree.is_tree(G): #single connected acyclic graph
        return _bfs_tree(G)

    else:
        if nx.is_forest(G): #multiple disconnected trees
            #todo: loop over connected components (=trees), compare longest branches
            return _exhaustive(G)
        else: #cyclic graph
            return _exhaustive(G)

def prune_graph(G, threshold=0.15, max_depth=5):
    """
    Endnodes of the main axis can be located in spines instead of stopping at the end
    of the main axis. We try to correct for that as follows:
    
    1) We select neighbors of the endnodes of the main-axis with neighborhood<=max_depth
    2) The average thickness between the end-node and a neighbor needs to be within the 
    threshold percentage of the overall average thickness from endnode to endnode. A node that
    does not respect this rule is flagged.
    3) Each node has a deep edge (away from endnode) and a "shallow" edge (closer to endnode).
    If the average thickness of a deep edge is higher than the avg. th. of a shallow edge, 
    the concerned node is flagged.
    One of the reasons 3) is working: thickness rises faster than length as nodes are removed
    4) The deepest flagged node and all nodes that are more shallow are removed from the graph.
    
    input: Graph, opt: max_depth of the graph, threshold for thickness %
    output: pruned graph without removed nodes
    """
    en = [x for x in G.nodes() if G.degree(x)==1] # endnodes
    avg_th = nx.shortest_path_length(G, en[0], en[1], weight='thick') / \
             nx.shortest_path_length(G, en[0], en[1], weight='weight')
    th = nx.shortest_path_length(G, en[0], en[1], weight='thick')
    
    def _neighborhood(G, node, n):
    # https://stackoverflow.com/questions/22742754/finding-the-n-degree-neighborhood-of-a-node
        path_lengths = nx.single_source_dijkstra_path_length(G, node, weight=None)
        return [node for node, length in path_lengths.iteritems() if length == n]
    # 1) find neighbors
    deep_neighbors = [_neighborhood(G, en[0], max_depth)[0], 
                      _neighborhood(G, en[1], max_depth)[0]]
    en_candidates = [list(nx.shortest_simple_paths(G, en[0], deep_neighbors[0]))[0][1:],
                     list(nx.shortest_simple_paths(G, en[1], deep_neighbors[1]))[0][1:]]
    
    # compute thickness of all neighbor nodes
    paththick0 =[nx.shortest_path_length(G, en[0], p, weight='thick') for p in en_candidates[0]]
    pathlen0 =  [nx.shortest_path_length(G, en[0], p, weight='weight') for p in en_candidates[0]]
    paththick1 =[nx.shortest_path_length(G, en[1], p, weight='thick') for p in en_candidates[1]]
    pathlen1 =  [nx.shortest_path_length(G, en[1], p, weight='weight') for p in en_candidates[1]]
    avgthick0 = [paththick0[i]/pathlen0[i] for i in range(max_depth)]
    avgthick1 = [paththick1[i]/pathlen1[i] for i in range(max_depth)]
    
    # 2) add to remove list all the nodes below threshold of avg thickness
    idx_rm0 = [i for i in range(len(avgthick0)) if avgthick0[i] < avg_th*threshold ]
    idx_rm1 = [i for i in range(len(avgthick1)) if avgthick1[i] < avg_th*threshold ]
    # 3) add to remove list all the nodes that have deep edge less thick than "shallow" edge
    idx_rm0 += [i for i in range(len(avgthick0) - 1) if avgthick0[i]>avgthick0[i+1]]
    idx_rm1 += [i for i in range(len(avgthick1) - 1) if avgthick1[i]>avgthick1[i+1]]
    
    # 4) remove list of nodes that are indexed
    idx_max0 = 0 if not idx_rm0 else max(idx_rm0) #rm nothing if empty rm array
    idx_max1 = 0 if not idx_rm1 else max(idx_rm1)
    en_rm0 = ([en[0]] + en_candidates[0])[:idx_max0]
    en_rm1 = ([en[1]] + en_candidates[1])[:idx_max1]
    Grm = G.copy()
    Grm.remove_nodes_from(en_rm0 + en_rm1)
    
    return Grm

def get_spines(G, Gma, return_extra=False):
    "remove main-axis edges from graph, return list of unconnected subgraphs"
    # return G - Gma_edges
    Grm = G.copy()
    Grm.remove_edges_from(Gma.edges) #rm edges between main axis nodes
    Grm.remove_nodes_from(list(nx.isolates(Grm))) #rm single nodes

    graphs = list(nx.connected_component_subgraphs(Grm)) # get subgraphs
    graphs = [g for g in graphs if len(g)>2] # rm outliers
    
    if return_extra == True:
        nc_clean = [len(g) for g in graphs if len(g)>2] #rm outliers
#         plt.hist(ln, bins=len(ln)+1)
#         plt.hist(nc_clean, bins=len(nc_clean)-1)
#         ui, uc = np.unique(ln, return_counts=True)
        return graphs, np.mean(nc_clean)
    
    else:
        return graphs # graph_list


def edge_length_and_thickness(G, node1, node2):
    length = nx.shortest_path_length(G, node1, node2, weight='weight')
    # assumption for thickness: path is unique
    thickness = nx.shortest_path_length(G, node1, node2, weight='thick')
#     count = len(G)
#     print(thickness, length)
#     print(G.nodes)
#     print(count)
    #length can be 0 if only 1 node
    # todo: add root node
    try:
        thickness /= length
        return length, thickness#, count
    except:
        return 0, 0#, 1
    
if __name__=='__main__':
# if opt=='4': # longest graph path
    print('start')
    
    bfs = 'bfs'; modified_bfs=False 
#     res = [60,64,64] # z,y,x resolution of skeleton
#     seg_fn = '/n/pfister_lab2/Lab/donglai/mito/db/30um_human/seg_64nm.h5' # crumbs
#     seg_fn = '/n/pfister_lab2/Lab/nils/snowproject/seg_64nm_maindendrite.h5' # no crumbs
#     seg_fn = '/n/pfister_lab2/Lab/donglai/mito/db/30um_human/segv2_16nm.h5' #obsolete
# #     dendrite_ids = np.loadtxt('mito_len500_bead_pair.txt', int)[:,1]
# #     dendrite_ids = np.loadtxt('data/seg_spiny_v2.txt', int)
#     dendrite_ids = np.loadtxt('/n/pfister_lab2/Lab/nils/snowproject/stats_humsegv2/ui500.txt')
#     dendrite_ids = dendrite_ids[dendrite_ids>0]

# first experiment donglai
#     res = [30,32,32] # z,y,x resolution of skeleton
#     seg_fn = '/n/pfister_lab2/Lab/donglai/mito/db/30um_human/seg_32nm.h5'#replacement
#     dendrite_ids = [11, 12, 13, 16, 17, 18, 20, 24, 25, 26] #handpicked benchm. exps
#     output_folder = '/n/pfister_lab2/Lab/nils/snowproject/hum_segv2/benchmarks/'

    
    args = get_args() # get args
    seg_fn = args.seg
    output_folder = args.out
    res = [int(i) for i in args.res.split(':')]
    dendrite_ids = np.array([int(i) for i in args.ids.split(':')])
    
    print('Read segmentation volume..')
    seg = ReadH5(seg_fn, 'main')
    
    if args.cs == 1: # only needed if no skeleton created yet
        print("\nCreate skeletons for given ids:")
        for i, did in enumerate(tqdm(dendrite_ids)):
            blockPrint()
            dendrite_folder = '{}/skels/{}/'.format(output_folder, did)
            CreateSkeleton(seg==did, dendrite_folder, res, res)
            enablePrint()
    
    print("\nAnalyse skeletons for given ids:")
    lookuptable = np.zeros((dendrite_ids.shape[0], 8))
    for i, did in enumerate(tqdm(dendrite_ids)):
        blockPrint()
        dendrite_folder = '{}/skels/{}/'.format(output_folder, did)
        # load skeleton of given seg id, return it and its graph
        G, skel = skeleton2graph(did, dendrite_folder, seg, res)

        # %% get longest axis 
#         main_G, _, _, endnodes = search_longest_path_exhaustive(G)
#         weight = 'weight' # longest path based on edge parameter weigth
        weight = 'weight' # longest path based on edge parameter weigth
        main_G, length = search_longest_path_efficient(G, weight=weight)
        if len(main_G)<=10: #hardcoded, 10 nodes needs to be bigger than max_depth
            endnodes = [x for x in main_G.nodes() if main_G.degree(x)==1]
            length, thickness = edge_length_and_thickness(main_G, endnodes[0], endnodes[1])
            lookuptable[i] = [did, len(main_G), thickness, length, 0, 0, 0, 0]
            continue
            
        main_G_pruned = prune_graph(main_G, threshold=0.15, max_depth=5)
#write_skel_coordinates(skel, main_G, save_path='node_pos_weightweight_noprune5.h5')

        endnodes = [x for x in main_G_pruned.nodes() if main_G_pruned.degree(x)==1]
        if not endnodes: #graphs that do not have any nodes left after pruning = outliers
            endnodes = [x for x in main_G.nodes() if main_G.degree(x)==1]
            length, thickness = edge_length_and_thickness(main_G, endnodes[0], endnodes[1])
            lookuptable[i] = [did, len(main_G), thickness, length, 0, 0, 0, 0]
            continue
        # get average thickness and length
        length, thickness = edge_length_and_thickness(main_G_pruned, endnodes[0], endnodes[1])
        
        #get spines of the main axis dendrite:
        len_spines = []
        thick_spines = []
        S_list = get_spines(G, main_G)
        for S in S_list:
            #TODO: renovate this part here !!!!!!!
            S_main, _ = search_longest_path_efficient(S)
            len(S_main)
            en = [x for x in S_main.nodes() if S_main.degree(x)==1] # endnodes
            ln_sp, tk_sp = edge_length_and_thickness(S_main, en[0], en[1])
            len_spines.append(ln_sp)
            thick_spines.append(tk_sp)
        
        nc_clean = [len(g) for g in S_list if len(g)>2]
        nc_mean = 0 if not nc_clean else np.mean(nc_clean)
        lookuptable[i] = [did, len(main_G), thickness, length, 
                          np.mean(thick_spines), np.mean(len_spines), 
                          nc_mean, len(S_list)]
        # backup saving
        
        np.savetxt('{}/lookuptable.txt'.format(output_folder), lookuptable,
            header = 'dendrite id, graph_sz, thickness, length,' + \
                   ' spines_avg_thickness, spines_avg_length, spines_avg_nodes, num_spines',
            fmt=['%d', '%d', '%f', '%f', '%f', '%f', '%f', '%d'] )
        
        enablePrint()
    
    lot_s = lookuptable[np.argsort(-lookuptable[:,2])]
    np.savetxt('{}/lookuptable.txt'.format(output_folder), lot_s,
            header = 'dendrite id, graph_sz, thickness, length,' + \
                   ' spines_avg_thickness, spines_avg_length, spines_avg_nodes, num_spines',
            fmt=['%d', '%d', '%f', '%f', '%f', '%f', '%f', '%d'] )
        
    print('done')

