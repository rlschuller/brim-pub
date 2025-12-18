import medpy.io
from pathlib import Path
import numpy as np
from tqdm import tqdm
import h5py


all_segs = h5py.File("data/processed/hdf5/breast_segmentation.hdf5", "r")
all_birads = h5py.File("data/processed/hdf5/processed.hdf5", "r")['birads']

n = 0
for id in tqdm(list(all_birads.keys())):
    birads = np.array(all_birads[id])
    segment = np.array(all_segs[id])

    b = np.sum(birads != 0)
    sb = np.sum((segment != 0) * (birads != 0))

    if b != sb:
        print()
        print(f"{np.sum(birads != 0)=}")
        print(f"{np.sum((segment != 0) * (birads != 0))=}")
        n += 1
        print(f"{n=}")

print(f"{n=}")
