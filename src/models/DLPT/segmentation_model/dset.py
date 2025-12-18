import glob
import os
import random
import math
import logging

import numpy as np

import torch
import torch.cuda
from torch.utils.data import Dataset
import torch.nn.functional as F

import json
import h5py

# Importing MRI class from another directory
GLOBAL_DIR = os.getcwd()
import sys
sys.path.append(GLOBAL_DIR)
from utils.mri_class import MRI

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)


INPUT_PATH = GLOBAL_DIR + '/data/processed/nrrd_segs/full_data/*'
TUMORINFO_PATH = GLOBAL_DIR + '/src/data/TumorInfo.json'
HDF_PATH = GLOBAL_DIR + '/data/processed/hdf5/'

def AugmentData(chunk, augmentation_dict):
    transform_t = torch.eye(4)
    chunk = chunk.unsqueeze(0)

    for i in range(3):
        if 'flip' in augmentation_dict:
            if random.random() > 0.5:
                transform_t[i,i] *= -1

        if 'offset' in augmentation_dict:
            offset_float = augmentation_dict['offset']
            random_float = (random.random() * 2 - 1)
            transform_t[i,3] = offset_float * random_float

        if 'scale' in augmentation_dict:
            scale_float = augmentation_dict['scale']
            random_float = (random.random() * 2 - 1)
            transform_t[i,i] *= 1.0 + scale_float * random_float


    if 'rotate' in augmentation_dict:
        angle_rad = random.random() * math.pi * 2
        s = math.sin(angle_rad)
        c = math.cos(angle_rad)

        rotation_t = torch.tensor([
            [c, -s, 0, 0],
            [s, c, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])

        transform_t @= rotation_t

    affine_t = F.affine_grid(
            transform_t[:3].unsqueeze(0).to(torch.float32),
            chunk.size(),
            align_corners=False,
        )

    augmented_chunk = F.grid_sample(
            chunk,
            affine_t,
            padding_mode='border',
            align_corners=False,
        ).to('cpu')

    if 'noise' in augmentation_dict:
        noise_t = torch.randn_like(augmented_chunk)
        noise_t *= augmentation_dict['noise']

        augmented_chunk += noise_t

    return augmented_chunk[0]



class MRI2dSegmentationDataset(Dataset):
    def __init__(self,
                 dataset_hdf,
                 contextSlices_count=3,
                 val_ratio_int=0,
                 isValSet_bool=None,
                 ratio_int=0,
                 kfold_ndx=0,
                 kfold_total=0,
            ):

        self.isValSet_bool = isValSet_bool
        self.contextSlices_count = contextSlices_count
        self.ratio_int = ratio_int

        self.slices = dataset_hdf['slices']
        keys_list = list(self.slices.keys())
        len_slices = len(self.slices)

        if kfold_total > 1 and kfold_ndx < kfold_total:
            val_list = keys_list[
                int(    kfold_ndx   * (len_slices/ kfold_total)):
                int((kfold_ndx + 1) * (len_slices/ kfold_total))
                ]
            if isValSet_bool:
                log.info("Fold {}/{}. Validation list: ({},{})".format(
                    kfold_ndx+1,
                    kfold_total,
                    int(    kfold_ndx   * (len_slices/ kfold_total)),
                    int((kfold_ndx + 1) * (len_slices/ kfold_total))
                    ))

        elif val_ratio_int > 0:
            val_len = len_slices // (val_ratio_int + 1)
            val_list = keys_list[:val_len]
            
        else:
            val_list = []
        
        if isValSet_bool:
            self.ID_list = val_list
        else:
            self.ID_list = list(set(keys_list) - set(val_list))

        self.num_slices_per_ID = [{'ID': ID, 'len': len(self.slices[ID])} for ID in self.ID_list]
        self.num_slices = np.sum([exam['len'] for exam in self.num_slices_per_ID])
        with open(TUMORINFO_PATH, 'r') as f:
            tumorinfo_list = json.load(f)
            tumorID_list = [tumor['ID'] for tumor in tumorinfo_list]
            num_nodules = len([ID for ID in tumorID_list if ID in self.ID_list])
            slices_with_tumor = []
            for ID in self.ID_list:
                slices = set()
                for tumor in tumorinfo_list:
                    if tumor['ID'] == ID:
                        slices |= set(range(int(tumor['center'][2]) - int(tumor['diameter']//2), int(tumor['center'][2]) + int(tumor['diameter']//2)))
                slices_with_tumor.append({'ID': ID, 'slices': np.array(list(slices))})

        log.info("{!r}: {} {} series, {} slices, {} nodules".format(
            self,
            len(self.slices),
            "validation" if isValSet_bool else "training",
            self.num_slices,
            num_nodules
        ))

    def shuffleSamples(self):
        random.shuffle(self.num_slices_per_ID)

    def __len__(self):
        return self.num_slices

    def __getitem__(self, ndx):
        if not self.isValSet_bool and self.ratio_int > 0:
            pos_ndx = int(ndx // (self.ratio_int + 1))

            if ndx % (self.ratio_int + 1):
                neg_ndx = ndx - 1 - pos_ndx
                neg_ndx %= len(self.neg_list)
                
            else:
                pos_ndx %= len(self.pos_list)

            if self.augmentation_dict:
                pass
                # mri_cube = AugmentData(mri_cube, self.augmentation_dict)
            
        else:
            context_slices = np.zeros(((self.contextSlices_count * 2 + 1) * 5, 1024, 1024))
            target = np.zeros((1024, 1024))
            ID_ndx = 0
            current_len = self.num_slices_per_ID[ID_ndx]['len']
            while ndx >= current_len:
                ndx -= current_len
                ID_ndx += 1
                current_len = self.num_slices_per_ID[ID_ndx]['len']
            start_ndx = ndx - self.contextSlices_count
            end_ndx = ndx + self.contextSlices_count + 1
            ID = self.num_slices_per_ID[ID_ndx]['ID']
            slices = self.slices[ID][max(start_ndx,0):min(end_ndx,current_len)]
            _, _, h, w = slices.shape
            target[:h,:w] = slices[ndx-max(start_ndx,0),-1]
            context_slices[:,:h,:w] = np.pad(slices[:,:-1], ((max(-start_ndx,0), max(end_ndx,current_len) - current_len), (0,0), (0,0), (0,0)), 'edge').reshape(-1, h, w)
            
        return context_slices, target


# Example usage
def main():
    dataset_hdf = h5py.File(HDF_PATH + 'slices.hdf5', 'r')
    ds = MRI2dSegmentationDataset(dataset_hdf, val_ratio_int=0, isValSet_bool=False)
    slice0, target = ds[0]
    dataset_hdf.close()


if __name__ == "__main__":
    main()