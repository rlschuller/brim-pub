import numpy as np
import os
import h5py
import matplotlib.pyplot as plt

CWD = os.getcwd()
BREAST_MRI_DIR = CWD[:CWD.find('breast_mri')+10]

import sys
sys.path.append(BREAST_MRI_DIR + '/src')
from utils.mri_class import MRI
from utils.util import enumerateWithEstimate

with h5py.File(f'{BREAST_MRI_DIR}/data/processed/hdf5/breast_segmentation.hdf5', 'r') as f:
    IDs = list(f.keys())
    IDs.sort(key=lambda x: int(x))

for _, ID in enumerateWithEstimate(IDs, 'Saving masks as PNGs'):
    mri = MRI(ID)
    img = np.rot90(mri.exam[0,:,:,100])
    mask = np.rot90(mri.seg[:,:,100])
    cmap = plt.cm.jet
    cmap.set_under(alpha=0)
    plt.imshow(img, cmap='gray')
    plt.imshow(mask, cmap=cmap, alpha=0.5, vmin=0.01)
    plt.axis('off')
    os.makedirs(f'{BREAST_MRI_DIR}/data/processed/pngs/new_breast_segmentation', exist_ok=True)
    plt.savefig(f'{BREAST_MRI_DIR}/data/processed/pngs/new_breast_segmentation/{ID}.png', bbox_inches='tight')