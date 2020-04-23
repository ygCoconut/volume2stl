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

def print_lot(lot_s):
    # use the following to print all the thicknesses and all the ids in a 
    # look_up_table.txt as a list
    print(','.join([str(int(i)) for i in lot_s[:,0]]) ) #get ids
    print(','.join([str(float(i)) for i in lot_s[:,1]]) ) #get th

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
    skel = ReadSkeletons(dendrite_folder,
                     skeleton_algorithm='thinning',
                     downsample_resolution=res,
                     read_edges=True)[1]

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

def search_longest_path_exhaustive(G):
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
    #Todo: remove main axis edges --> nodes belong to spines
    "remove main-axis nodes from graph, return list of unconnected subgraphs"
    for node in list(mainaxis_nodes):
        Gsp.remove_node(node)

    graphs = list(nx.connected_component_subgraphs(G))
    return graphs # graph_list


def prune_graph(G0, threshold=0.5, num_its=3):
    "recursively prunes endnodes of graph, if edge thickness below avg graph thickness"
    endnodes = [x for x in G.nodes() if G.degree(x)==1]
#     avg_th = nx.shortest_path_length(G, endnodes[0], endnodes[1], weight='thick')
    avg_th = nx.shortest_path_length(G, endnodes[0], endnodes[1], weight='thick') / nx.shortest_path_length(G, endnodes[0], endnodes[1], weight='weight')
    
    def _prune(G):
        endnodes = [x for x in G.nodes() if G.degree(x)==1]
        import pdb; pdb.set_trace()
#         endthick = G[endnodes[0]][nx.neighbors(G, endnodes[0]).next()]['thick']
        endthick = G[endnodes[0]][nx.neighbors(G, endnodes[0]).next()]['thick'] / G[endnodes[0]][nx.neighbors(G, endnodes[0]).next()]['weight']
        if endthick < threshold*avg_th:
            G.remove_node(endnodes[0])
#         endthick = G[endnodes[1]][nx.neighbors(G, endnodes[1]).next()]['thick']
        endthick = G[endnodes[1]][nx.neighbors(G, endnodes[1]).next()]['thick'] / G[endnodes[1]][nx.neighbors(G, endnodes[1]).next()]['weight']
        if endthick < threshold*avg_th:
            G.remove_node(endnodes[1])
        return G
    
    G1 = nx.Graph(G0) #unfrozen_graph
    for _ in range(num_its):
        G1 = _prune(G1)
    return G1

def prune_graph2(G, threshold=0.15, max_depth=3):
    en = [x for x in G.nodes() if G.degree(x)==1] # endnodes
    avg_th = nx.shortest_path_length(G, en[0], en[1], weight='thick') / \
             nx.shortest_path_length(G, en[0], en[1], weight='weight')
    th = nx.shortest_path_length(G, en[0], en[1], weight='thick')
    
    def _neighborhood(G, node, n):
    # https://stackoverflow.com/questions/22742754/finding-the-n-degree-neighborhood-of-a-node
        path_lengths = nx.single_source_dijkstra_path_length(G, node, weight=None)
        return [node for node, length in path_lengths.iteritems() if length == n]
    
    deep_neighbors = [_neighborhood(G, en[0], max_depth)[0], 
                _neighborhood(G, en[1], max_depth)[0]]
    paths = [list(nx.shortest_simple_paths(G, en[0], deep_neighbors[0]))[0][1:],
             list(nx.shortest_simple_paths(G, en[1], deep_neighbors[1]))[0][1:]]
    
    paththick0 =[nx.shortest_path_length(G, en[0], p, weight='thick') for p in paths[0]]
    pathlen0 =  [nx.shortest_path_length(G, en[0], p, weight='weight') for p in paths[0]]
    paththick1 =[nx.shortest_path_length(G, en[1], p, weight='thick') for p in paths[1]]
    pathlen1 =  [nx.shortest_path_length(G, en[1], p, weight='weight') for p in paths[1]]
    
    avgthick0 = [paththick0[i]/pathlen0[i] for i in range(max_depth)]
    avgthick1 = [paththick1[i]/pathlen1[i] for i in range(max_depth)]
    
    mtx = np.array([paththick0, paththick1,
                    pathlen0,   pathlen1,
                    avgthick0,  avgthick1 ])

    np.savetxt('paths.txt', mtx)
    for n in paths[0]:
        try:
            G.remove_node(n) if endthick < threshold*avg_th
        except: pass
    for n in paths[1]:
        try:
            G.remove_node(n) if endthick < threshold*avg_th
        except: pass
        
            
def plot_Graph(G, nodepos=None, return_pos=False):
    if nodepos == None:
        nodepos = nx.kamada_kawai_layout(G)
    endnodes = [x for x in G.nodes() if G.degree(x)==1]
    endnodepos = {k:v for k,v in nodepos.items() if k in endnodes}
    

    # need to debug: labels dict not filtering..
    labels = nx.get_edge_attributes(G,'weight')
    labels_new = {k:v for k,v in labels.items() if k in G.edges(endnodes)}
    nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)

    
    # only draw nodelabels for endnodes
    nx.draw_networkx_labels(G.subgraph(endnodes), nodepos)
    nx.draw(G,nodepos)
    
    

    G.edges(endnodes)
    S = G.edge_subgraph(G.edges(endnodes))
    
    dir(G)

    labels = nx.get_edge_attributes(G.subgraph(G.edges(endnodes)) ,'thick')
    nx.draw_networkx_edge_labels(G,nodepos,edge_labels=labels)

    
    nx.get_edge_attributes(G.edges(endnodes)['thick'])

#     nx.draw_networkx_edges(G)
    nx.draw_networkx_nodes(G, pos)
    pos[endnodes]
    labels = nx.get_edge_attributes(G,'weight')
    nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)
#     labels = nx.get_edge_attributes(G,'weight')
        
#     nx.draw_networkx_edges(G, pos, labels)
    
    if return_pos: 
        return pos
        
if __name__=='__main__':
# if opt=='4': # longest graph path
    print('start')
    
    bfs = 'bfs'; modified_bfs=False 
    res = [60,64,64] # z,y,x resolution of skeleton
    seg_fn = '/n/pfister_lab2/Lab/donglai/mito/db/30um_human/seg_64nm.h5' # crumbs
    seg_fn = '/n/pfister_lab2/Lab/nils/snowproject/seg_64nm_maindendrite.h5' # no crumbs
    
#     dendrite_ids = np.loadtxt('mito_len500_bead_pair.txt', int)[:,1]
    dendrite_ids = np.loadtxt('data/seg_spiny_v2.txt', int)
    lookuptable = np.zeros((dendrite_ids.shape[0], 5))
    did = 6659767
    did = 1499496

    seg = ReadH5(seg_fn, 'main')
#     seg = np.array(h5py.File(seg_fn, 'r')['main'])
    
    create_skel = False
    if create_skel == True: # only needed if no skeleton created yet 
        print("\nCreate skeletons for given ids:\n")
        for i, did in enumerate(tqdm(dendrite_ids)):
            blockPrint()
            dendrite_folder = 'results_spines/{}/'.format(did)
            CreateSkeleton(seg==did, dendrite_folder, res, res)
            enablePrint()
    

    for i, did in enumerate(tqdm(dendrite_ids)):
        blockPrint()
        dendrite_folder = 'results_spines/{}/'.format(did)
        # load skeleton of given seg id, return it and its graph
        G, skel = skeleton2graph(did, dendrite_folder, seg, res)

        # %% get longest axis 
#         main_G, _, _, endnodes = search_longest_path_exhaustive(G)
#         weight = 'weight' # longest path based on edge parameter weigth
        weight = 'weight' # longest path based on edge parameter weigth
        main_G, length = search_longest_path_efficient(G, weight=weight)
        main_G_pruned = prune_graph(main_G, threshold=0.9, num_its=3)
        write_skel_coordinates(skel, main_G, save_path='node_pos_weightweight_prune.h5')

    
        endnodes = [x for x in main_G.nodes() if main_G.degree(x)==1]
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
        # backup saving
        np.savetxt('lookuptable.txt', lookuptable,
            header = 'dendrite id, thickness, length,' + \
                   ' spines_avg_thickness, spines_avg_length',
            fmt=['%d', '%f', '%f', '%f', '%f'] )
        
        enablePrint()
    
    lot_s = lookuptable[np.argsort(-lookuptable[:,1])]
    np.savetxt('lookuptable.txt', lot_s,
            header = 'dendrite id, thickness, length,' + \
                   ' spines_avg_thickness, spines_avg_length',
            fmt=['%d', '%f', '%f', '%f', '%f'] )
        
    print('done')

