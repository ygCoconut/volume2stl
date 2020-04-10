import numpy as np
import os, sys, json
import itertools


def _get_dendrite_mito_map(fp):
    mitodirs = os.listdir(fp)
    mapmitos = {}
    for m in mitodirs:
        dendr_id = m.replace('.txt', '')
        mitos = os.popen('cat {}{}'.format(donglaidir, m)).read().split('\n')
        mitos = filter(None, mitos)
        mapmitos[int(dendr_id)] = [int(m) for m in mitos]
    return mapmitos

def get_lut():
    donglaidir = '/n/pfister_lab2/Lab/donglai/mito/db/30um_human/neuron_mito/'
    lookuptable = _get_dendrite_mito_map(donglaidir)
    with open('lut_dendrite_mito.json', 'w') as outfile:
        json.dump(lookuptable, outfile)
        
def get_colon_sv():
    # get string with comma sep. vals containing all mitos
    with open('lut_dendrite_mito.json', 'r') as json_file:
        data = json.load(json_file)
    mito_list = []
    for k in data:
        mito_list.append(data[k])
    
    colon_sv = ':'.join(str(i) for i in list(itertools.chain.from_iterable(mito_list)))
    with open('mito300_colon_sv.txt', 'w') as outfile:
        json.dump(colon_sv, outfile)

get_colon_sv()

# ''.join(mito_list)
print(list(itertools.chain.from_iterable(mito_list)))

idlist = np.loadtxt('seg_spiny_v2.txt')

idlist = ":".join([str(int(i)) for i in idlist])


# mito-id, seg-id
idmap = np.loadtxt('/n/pfister_lab2/Lab/donglai/mito/db/30um_human/mito_len500_bead_pair.txt')
print(":".join([str(int(i)) for i in idmap[:,0]]))
print(":".join([str(int(i)) for i in idmap[:,1]]))
idmap.shape

## %% Lookup table processing
lot = np.loadtxt('lookuptable_dendsandspines.txt')
lot = lot[:,[0,1,2,4,3]]
    
lot_s = lot[np.argsort(-lot[:,1])]
np.savetxt('lookuptable.txt', lot_s,
        header = 'dend_id, thick, lg, spine_th, spine_lg',
        fmt=['%d', '%f', '%f', '%f', '%f'])