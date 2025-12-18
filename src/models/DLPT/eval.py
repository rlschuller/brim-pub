import numpy as np
import torch
from model import MRIModel
import functools
import json
import itertools
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import nrrd
import os

from scipy.ndimage import zoom

import argparse


CWD = os.getcwd()
BREAST_MRI_DIR = './'

import sys
sys.path.append(BREAST_MRI_DIR + '/src')
from utils.mri_class import MRI, cubify, uncubify
from utils.util import enumerateWithEstimate

TUMORINFO_PATH = BREAST_MRI_DIR + '/src/data/TumorInfoComplete.json'
DATA_PATH = BREAST_MRI_DIR + '/data/processed/nrrd_segs/full_data/'
HDF_PATH = BREAST_MRI_DIR + '/data/processed/hdf5/'
SPLIT_PATH = BREAST_MRI_DIR + '/data/processed/datasets/'

@functools.lru_cache(1)
def getMRI(ID: str) -> MRI:
    return MRI(ID)

def init_model(model_path: str) -> MRIModel:
    model = MRIModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model = model.to(torch.device("cuda" if use_cuda else "cpu"))
    return model

def InBreast(ID: str, c: tuple, size: int) -> bool:
    mri = getMRI(ID)
    InMask = np.mean(mri.get_seg()[c[0]:c[0] + size, c[1]:c[1] + size, c[2]:c[2] + size]) > 0.4
    return InMask

class eval_dataset(Dataset):
    def __init__(self, ID: str, size: int=32, stride: int=8):
        self.ID = ID
        self.size = size
        self.stride = stride
        self.mri = getMRI(ID)
        self.exam = self.mri.exam
        self.exam_shape = self.exam.shape
        self.pred_cubes = np.zeros((self.exam_shape[1]//stride, self.exam_shape[2]//stride, self.exam_shape[3]//stride))
        self.tumors = np.zeros((2,) + self.mri.birads.shape)
        self.cubes_coords = list(itertools.product(range(0, self.exam_shape[1] - size + 1, stride), range(0, self.exam_shape[2] - size + 1, stride), range(0, self.exam_shape[3] - size + 1, stride)))
        self.cubes_coords = [c for c in self.cubes_coords if InBreast(ID, c, size)]
        self.cubes_coords = np.array(self.cubes_coords)
    
    def __len__(self):
        return len(self.cubes_coords)
    
    def __getitem__(self, idx):
        c = self.cubes_coords[idx]
        i1, i2, i3 = c[0]//self.stride, c[1]//self.stride, c[2]//self.stride
        cube = self.exam[:, c[0]:c[0] + self.size, c[1]:c[1] + self.size, c[2]:c[2] + self.size]
        cube = torch.tensor(cube)
        return cube, (i1, i2, i3)
    
    def update(self, idx, pred, batch_size):
        idx_list = list(range(idx * batch_size, min((idx+1) * batch_size, len(self.cubes_coords))))
        for i, idx in enumerate(idx_list):
            c = self.cubes_coords[idx]
            self.tumors[0,c[0]:c[0] + self.size, c[1]:c[1] + self.size, c[2]:c[2] + self.size] += pred[1][i,0].item()
            self.tumors[1,c[0]:c[0] + self.size, c[1]:c[1] + self.size, c[2]:c[2] + self.size] += pred[1][i,1].item()


def eval(model_path: str, ID: str, size: int=32, stride: int=8):
    """
    Evaluate the model on the MRI data.

    Parameters:
        model_path (str): The path to the saved model.
        ID (str): The ID of the MRI data.
        size (int): The size of the cubes to be evaluated.
        stride (int): The stride of the cubes to be evaluated.
    """
    with torch.no_grad():
        batch_size = 64
        model = init_model(model_path)
        dataset = eval_dataset(ID, size, stride)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        # for i, cube in enumerateWithEstimate(loader, f"Eval {ID}"):
        for i, cube in enumerate(loader):
            pred = model(cube[0].to(torch.device("cuda")))
            dataset.update(i, pred, batch_size)
    return dataset.tumors


def plot_pred(ID, tumors):
    mri = getMRI(ID)
    with open(TUMORINFO_PATH, 'r') as f:
        tumors_info = json.load(f)
    tumor_coords = np.array([t['center'] for t in tumors_info if t['ID'] == ID])
    cmap_pred = plt.cm.jet
    cmap_pred.set_under(alpha=0.)
    cmap_birads = plt.cm.viridis
    cmap_birads.set_under(alpha=0.)
    for i, c in enumerate(tumor_coords):
        z_coord_tumor = c[2]
        slc = mri.exam[0, :, :, z_coord_tumor]
        tumor_slc = tumors[1, :, :, z_coord_tumor] / np.max(tumors[1])
        birads_slc = mri.birads[:, :, z_coord_tumor]


        # rotate
        slc = np.rot90(slc)
        tumor_slc = np.rot90(tumor_slc)
        birads_slc = np.rot90(birads_slc)

        plt.figure(frameon=False)
        plt.imshow(slc, cmap='gray')
        plt.imshow(tumor_slc, cmap=cmap_pred, alpha=0.3, vmin=0.3, vmax=1.)
        plt.imshow(birads_slc, cmap=cmap_birads, alpha=0.8, vmin=0.1)
        plt.axis('off')

        plt.savefig(images_path + f'{ID}_tumor_{i}.png', dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()
        #plt.show()
    

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', 
                    type=str, 
                    default='2024-09-03_06.06.13-flip_lr1e-4',
                    help='Model name')

parser.add_argument('--epoch',
                    type=int,
                    default=3,
                    help='Epoch')

parser.add_argument('--cube-size'
    # WORK IN PROGRESS
)

args = parser.parse_args()

MODEL_NAME = args.model_name
EPOCH = args.epoch
CUBE_SIZE = 32
STRIDE = 8



model_path = BREAST_MRI_DIR + f'/models/DLPT/split/{MODEL_NAME}/epoch{EPOCH}.pt'
images_path = BREAST_MRI_DIR + f'/data/processed/DLPT_preds/split/images/{MODEL_NAME}-epoch{EPOCH}/'
preds_path = BREAST_MRI_DIR + f'/data/processed/DLPT_preds/split/{MODEL_NAME}-epoch{EPOCH}/'
os.makedirs(images_path, exist_ok=True)
os.makedirs(preds_path, exist_ok=True)


def main():
    split_filename = 'crimson_duck-843'
    with open(SPLIT_PATH + split_filename + '.json', 'r') as f:
        split = json.load(f)
    valpathList = split['val']
    testpathList = split['test']
    valList = [os.path.basename(p) for p in valpathList]
    testList = [os.path.basename(p) for p in testpathList]

    # Run the evaluation on the validation set
    for _, ID in enumerateWithEstimate(valList + testList, f'Evaluating {MODEL_NAME} epoch {EPOCH}'):
        tumors = eval(model_path, ID, stride=STRIDE, size=CUBE_SIZE)
        plot_pred(ID, tumors)

        np.savez_compressed(BREAST_MRI_DIR + f'/data/processed/DLPT_preds/split/{MODEL_NAME}-epoch{EPOCH}/{ID}.npz', tumors=tumors[1])
    # runEval('03')

# This is necessary to avoid a bug in Qt
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")

if __name__ == '__main__': 
    main()