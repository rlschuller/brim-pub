import numpy as np
import matplotlib.pyplot as plt
import sys
import h5py
import os

CWD = os.getcwd()
BREAST_MRI_DIR = CWD[:CWD.find('breast_mri')+10]

import sys
sys.path.append(BREAST_MRI_DIR + '/src')
from utils.mri_class import MRI
from utils.util import enumerateWithEstimate

def getMRI(ID):
    return MRI(ID)

with h5py.File('./data/processed/hdf5/processed.hdf5', 'r') as f:
    IDs = list(f['birads'].keys())
    IDs.sort(key=lambda x: int(x))

number_of_tumors = []
for _,ID in enumerateWithEstimate(IDs, 'Counting number of tumors'):
    mri = getMRI(ID)
    tumors = mri.get_tumors()
    number_of_tumors.append(len(tumors))
    if len(tumors) == 0:
        print(f'No tumors in {ID}')
    if len(tumors) > 5:
        print(f'{ID} has {len(tumors)} tumors')
    
    
plt.hist(number_of_tumors, bins = range(0, 10))
plt.tight_layout
plt.title('Number of tumors per exam')
plt.xlabel('Number of tumors')
plt.ylabel('Number of exams')
plt.savefig('number_of_tumors.png')