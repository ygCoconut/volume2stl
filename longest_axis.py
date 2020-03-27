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


def load_graph(out_folder, bfs)
    import networkx as nx
    G = nx.read_gpickle(out_folder+'graph-%s.obj'%(bfs))
    n0 = len(G.nodes())
#     G = ShrinkGraph_v2(G, threshold=edgTh)
    n1 = len(G.nodes())
    print('#nodes: %d -> %d'%(n0,n1))
    return G
    
def longest_axis_exhaustive(G):
    
    path_list = []
    path_length = []
    for source in G.nodes:
        for target in G.nodes:
            sh_p = nx.shortest_path(G, source, target)
            sh_p_l = nx.shortest_path_length(G, source, target)
            path_list.append(sh_p)
            path_length.append(sh_p_l)

    return path_list, path_length


if __name__=='__main__':
# if opt=='4': # longest graph path
    
#     opt = sys.argv[1]
#     seg_fn = '/n/pfister_lab2/Lab/donglai/mito/db/30um_human/seg_64nm.h5'
#     res = [60,64,64] # z,y,x
    out_folder = 'results/ibexHelper/'
    bfs = 'bfs'; modified_bfs=False 
    
    G = load_graph(out_folder, bfs)
    paths, length =longest_axis_exhaustive(G)

    print('longest path:\n:', paths[np.argmax(length)])
    print('path length: \t:', np.max(length))


