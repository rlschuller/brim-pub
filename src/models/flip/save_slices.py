import cv2
import h5py
import os
import numpy as np

import sys
sys.path.append('./src')
from utils.util import enumerateWithEstimate

os.chdir('./data/processed/hdf5/')

f = h5py.File('processed.hdf5', 'r')
exams = f['exams']
assert isinstance(exams, h5py.Group)

ID_list = list(exams.keys())
ID_list.sort(key=lambda x: int(x))

ID_test = ID_list[::10]
ID_test.sort(key=lambda x: int(x))
ID_list = list(set(ID_list) - set(ID_test))
ID_list.sort(key=lambda x: int(x))

size = 32
slices = np.zeros((len(ID_test), size, size))
for i, ID in enumerateWithEstimate(ID_test, 'Saving slices'):
    exam = exams[ID][()]
    slc = np.max(exam, axis=(0, 3))
    slc = cv2.resize(slc, (size, size))
    slices[i] = slc

# Save the slices as npy
np.save('./src/models/flip/slices32_test.npy', slices)

f.close()