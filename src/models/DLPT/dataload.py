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

CWD = os.getcwd()
BREAST_MRI_DIR = CWD[:CWD.find('breast_mri')+10]

import sys
sys.path.append(BREAST_MRI_DIR + '/src')
from utils.mri_class import MRI

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)



INPUT_PATH = BREAST_MRI_DIR + '/data/processed/nrrd_segs/full_data/*'
CANDIDATEINFO_PATH = BREAST_MRI_DIR + '/src/data/TumorInfo.json'
HDF_PATH = BREAST_MRI_DIR + '/data/processed/hdf5/'
SPLIT_PATH = BREAST_MRI_DIR + '/data/processed/datasets/'

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


class MRIDataset(Dataset):
    def __init__(self,
                 dataset_hdf,
                 val_ratio_int=0,
                 split_name=None,
                 isValSet_bool=None,
                 ratio_int=0,
                 augmentation_dict=None,
            ):

        self.isValSet_bool = isValSet_bool
        self.augmentation_dict = augmentation_dict

        self.nontumors = dataset_hdf['nontumors']
        self.tumors = dataset_hdf['tumors']


        if split_name:
            split_path = SPLIT_PATH + split_name + '.json'
            with open(split_path, 'r') as f:
                split_dict = json.load(f)
                path_list = split_dict['train' if not isValSet_bool else 'val']
                key_list = [os.path.basename(path) for path in path_list]
                self.key_list = key_list
        
        else:
            key_list = list(set(self.nontumors.keys()) | set(self.tumors.keys()))
            key_list.sort(key=lambda x: int(x))
            
            if val_ratio_int > 0:
                val_key_list = key_list[::val_ratio_int + 1]
                train_key_list = list(set(key_list) - set(val_key_list))
            else:
                val_key_list = []
                train_key_list = key_list
    
            if isValSet_bool:
                self.key_list = val_key_list
            else:
                self.key_list = train_key_list

        self.pos_list = []
        self.neg_list = []

        for key in self.key_list:
            if key in self.tumors:
                new_address = [(key, i) for i in range(self.tumors[key].shape[0])]
                self.pos_list.extend(new_address)
            if key in self.nontumors:
                new_address = [(key, i) for i in range(self.nontumors[key].shape[0])]
                self.neg_list.extend(new_address)

        self.ratio_int = ratio_int

        log.info("{!r}: {} exams, {} {} samples, {} neg, {} pos, {} ratio".format(
            self,
            len(self.key_list),
            len(self.neg_list) + len(self.pos_list),
            "validation" if isValSet_bool else "training",
            len(self.neg_list),
            len(self.pos_list),
            '{}:1'.format(self.ratio_int) if self.ratio_int else 'unbalanced'
        ))


    def shuffleSamples(self):
        if self.ratio_int:
            random.shuffle(self.neg_list)
            random.shuffle(self.pos_list)

    def __len__(self):
        if self.isValSet_bool:
            return len(self.pos_list) + len(self.neg_list)
        elif self.ratio_int > 0:
            return int((len(self.neg_list) + 1) * (self.ratio_int + 1)/ self.ratio_int)
        else:
            return len(self.pos_list) + len(self.neg_list)

    def __getitem__(self, ndx):
        if not self.isValSet_bool and self.ratio_int > 0:
            pos_ndx = int(ndx // (self.ratio_int + 1))

            if ndx % (self.ratio_int + 1):
                neg_ndx = ndx - 1 - pos_ndx
                neg_ndx %= len(self.neg_list)
                ID, i = self.neg_list[neg_ndx]
                mri_cube = torch.from_numpy(self.nontumors[ID][i])
                target = torch.tensor([1, 0], dtype=torch.long)
            else:
                pos_ndx %= len(self.pos_list)
                ID, i = self.pos_list[pos_ndx]
                mri_cube = torch.from_numpy(self.tumors[ID][i])
                target = torch.tensor([0, 1], dtype=torch.long)

            if self.augmentation_dict:
                mri_cube = AugmentData(mri_cube, self.augmentation_dict)
            
        else:
            if ndx < len(self.pos_list):
                ID, i = self.pos_list[ndx]
                mri_cube = torch.from_numpy(self.tumors[ID][i])
                target = torch.tensor([0, 1], dtype=torch.long)
            else:
                ndx -= len(self.pos_list)
                ID, i = self.neg_list[ndx]
                mri_cube = torch.from_numpy(self.nontumors[ID][i])
                target = torch.tensor([1, 0], dtype=torch.long)
        

        return mri_cube, target


# # For visualization of the augmentation process
# import matplotlib.pyplot as plt
# def view_augmentation():
#     augmentation_dict={
#         'flip': True, 
#         'offset': 0.1, 
#         'scale': 0.2, 
#         'rotate': True, 
#         'noise': 25.
#     }

#     dataset_hdf = h5py.File(HDF_PATH + 'complete_dataset32.hdf5', 'r')
#     train_ds = MRIDataset(dataset_hdf, val_ratio_int=5, isValSet_bool=False, ratio_int=1)
#     block, target = train_ds[0]

#     if target[1] == 1:
#         fig, axs = plt.subplots(2, 3)
#         block_before = block.clone()
#         axs[0, 0].imshow(block_before[2, 16], cmap='gray')
#         axs[0, 1].imshow(block_before[2, : ,16], cmap='gray')
#         axs[0, 2].imshow(block_before[2, :, :, 16], cmap='gray')
#         axs[0, 1].set_title('Before Augmentation')
#         block = AugmentData(block_before, augmentation_dict)
#         axs[1, 0].imshow(block[2, 16], cmap='gray')
#         axs[1, 1].imshow(block[2, : ,16], cmap='gray')
#         axs[1, 2].imshow(block[2, :, :, 16], cmap='gray')
#         axs[1, 0].set_title('After Augmentation')
#         plt.show()
    
#     dataset_hdf.close()



# Example usage
def main():
    dataset_hdf = h5py.File(HDF_PATH + 'cubes32stride16.hdf5', 'r')
    train_ds = MRIDataset(dataset_hdf, isValSet_bool=False, ratio_int=1, split_name='crimson_duck-843')
    val_ds = MRIDataset(dataset_hdf, isValSet_bool=True, ratio_int=1, split_name='crimson_duck-843')
    print(f"{train_ds[0][0].shape = }")
    print(f"{train_ds[0][1] = }")
    print(f"{val_ds[0][0].shape = }")
    print(f"{len(train_ds) = }")
    print(f"{len(val_ds) = }")
    dataset_hdf.close()


if __name__ == "__main__":
    main()