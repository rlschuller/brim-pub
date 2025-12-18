import argparse
from glob import glob

import medpy.io
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument("--path", "-p", type=str)

args = parser.parse_args()

list_of_birads_paths = glob(f"{args.path}/**/birads.nrrd", recursive=True)
list_of_birads_paths += glob(f"{args.path}/**/B.nrrd", recursive=True)
list_of_birads_paths += glob(f"{args.path}/**/C.nrrd", recursive=True)
list_of_birads_paths += glob(f"{args.path}/**/D.nrrd", recursive=True)
list_of_birads_paths += glob(f"{args.path}/**/E*.nrrd", recursive=True)

for path in list_of_birads_paths:
    data, header = medpy.io.load(path)
    number_of_nonzero_voxels = np.sum(data != 0)
    tqdm.write(f"{number_of_nonzero_voxels},{path}")
