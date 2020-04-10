'''
0 Preprocess segments:
-
- specify segments you want to process
- dilate slightly the segments
- create mask for dilation.
- np.unique(my_masked_id) --> select only part with biggest uc
- eliminates ouliers too disconnected/far from main structure 
'''

import numpy as np
import h5py
from scipy.ndimage import binary_dilation, label
from tqdm import tqdm

def writeh5_file(file, filename=None):
    hf = h5py.File(filename, 'w')
    hf.create_dataset('main', data=file)
    hf.close()


if __name__=='__main__':
    print('start')

    segpath = '/n/pfister_lab2/Lab/donglai/mito/db/30um_human/seg_64nm.h5'
    savepath = '/n/pfister_lab2/Lab/nils/snowproject/seg_64nm_maindendrite.h5'
    seg = h5py.File(segpath, 'r')
    seg = np.array(seg['main'], np.uint32) # x y z
    
    dendrite_ids = np.loadtxt('seg_spiny_v2.txt', int)
    for i, did in enumerate(tqdm(dendrite_ids)):
#         dil = binary_dilation(seg==did)*did

        # find all components of the dendrite, tolerate tiny gaps
        s = np.ones((3, 3, 3), int)
        dil, nf = label((seg==did)*did, structure=s)
        # find main component
        ui, uc = np.unique(dil, return_counts=True)
        uc = uc[ui>0]; ui = ui[ui>0]
        max_id = ui[np.argmax(uc)]
        # remove non-main components from segmentation
        seg[seg==did] = 0
        seg[dil==max_id] = did

    writeh5_file(seg, savepath)
    print('start')
        
        
    
        
    
