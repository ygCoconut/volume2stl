import pdb
import h5py
import numpy as np
import kimimaro
import matplotlib.pyplot as plt
import pickle
# from spinecode import mesh, skel

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
    parser.add_argument('-task', type=int, default=0, help='graph generation method')
    
    parser.add_argument('-ws', type=str, default='skel_points',
                        help='write skeleton with name -ws')
    args = parser.parse_args()
    return args

def loadh5py(path, vol="main"):
    return np.array(h5py.File(path, 'r')[vol]).squeeze()

def writeh5_file(file, filename):
    hf = h5py.File(filename, 'w')
    hf.create_dataset('main', data=file)
    hf.close()
    
def writeh5_file_compress(dtarray, filename):
    datasetname = 'main'
    fid=h5py.File(filename,'w')
    ds = fid.create_dataset(datasetname, dtarray.shape, compression="gzip", dtype=dtarray.dtype)
    ds[:] = dtarray
    fid.close()

# LISTING 1: Producing Skeletons from a labeled image.
def skeletonize(labels, scale, const, obj_ids):
    skels = kimimaro.skeletonize(
      labels, 
      teasar_params={
        'scale': scale,
        'const': const, # physical units
        'pdrf_exponent': 4,
        'pdrf_scale': 100000,
        'soma_detection_threshold': 1100, # physical units
        'soma_acceptance_threshold': 3500, # physical units
        'soma_invalidation_scale': 1.0,
        'soma_invalidation_const': 300, # physical units
        'max_paths': 50, # default  None
      },
      object_ids= obj_ids, # process only the specified labels
      # object_ids=[ ... ], # process only the specified labels
      # extra_targets_before=[ (27,33,100), (44,45,46) ], # target points in voxels
      # extra_targets_after=[ (27,33,100), (44,45,46) ], # target points in voxels
      dust_threshold=500, # skip connected components with fewer than this many voxels
#       anisotropy=(30,30,30), # default True
      anisotropy=(32,32,30), # default True
      fix_branching=True, # default True
      fix_borders=True, # default True
      progress=True, # default False, show progress bar
      parallel=4, # <= 0 all cpu, 1 single process, 2+ multiprocess
      parallel_chunk_size=100, # how many skeletons to process before updating progress bar
    )
    return skels

if False:
    from run_kimimaro import *

if __name__ == '__main__':
    print('start')

#     dendrite_ids = np.loadtxt('mito_len500_bead_pair.txt', int)[:,1]
#     dendrite_ids = np.loadtxt('data/seg_spiny_v2.txt', int)

    
    # dendrites; task NILS
    seg_fn = '/n/pfister_lab2/Lab/donglai/mito/db/30um_human/seg_32nm.h5'
#     obj_ids = np.loadtxt('/n/pfister_lab2/Lab/nils/snowproject/hum_segv2/ui500.txt')
#     out_f = '/n/pfister_lab2/Lab/nils/snowproject/hum_segv2/skel_16nmv2_kimi/'
#     out_f = '/n/pfister_lab2/Lab/nils/snowproject/hum_segv2/skel_32nm_kimi/'
    out_f = '/n/pfister_lab2/Lab/nils/snowproject/hum_segv2/benchmarks/'
#     oid = obj_ids[obj_ids>0]
    oid = [11, 12, 13, 16, 17, 18, 20, 24, 25, 26]

    #mito; task SILIN
#     seg_fn = '/n/pfister_lab2/Lab/donglai/mito/db/30um_rat/mito_64nm.h5'
#     obj_ids = np.loadtxt('/n/pfister_lab2/Lab/donglai/mito/db/30um_rat/mito_len500_bead.txt', int)
#     oid = obj_ids[obj_ids>0]
#     out_f = '/n/pfister_lab2/Lab/nils/snowproject/rat_segv2/skel_mito64nm_kimi/'
    
    
    # no crumbs
    print('Load segmentation from {}'.format(seg_fn))
    labels = loadh5py(seg_fn)
    
    if True:
        print('skeletonize..')
        import time
        start_time = time.time()
        skels = skeletonize(labels, 4, 500, obj_ids=list(oid))
        #save skeleton
        print('save skel at {}'.format(out_f))
        with open('{}/skeleton_job.p'.format(out_f), 'wb') as fp:
            pickle.dump(skels, fp, protocol=pickle.HIGHEST_PROTOCOL)
        print("--- %s seconds ---" % (time.time() - start_time))
    else:
        skels = np.load('{}/skeleton.p'.format(out_f), allow_pickle=True)

    # get special information:  
    if True:
        sl = [[k, len(skels[k].vertices),  skels[k].radius.mean()] for k in skels]    
        writeh5_file(sl, '{}/skel_len_rad.h5'.format(out_f))
    

    print('done')

    
    
    # Try things out in console:
#     print('run skeleton --> dendrites and spines')
#     skels.keys()
#     dir(skels[1499496])    
#     skels[1499496].viewer()    
#     skel_labels = skel.label_skeleton(skels[1499496])
#     dir(skel_labels)
#     skels[1499496].vertices *= skel_labels #might not work
        
#     skel_labels.max()    
    
#     This part here is more important:
#     rl = skels[1499496].clone()
#     rl.radius[skel_labels==1] = 0
#     rl.viewer()
    

    

