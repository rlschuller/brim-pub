import argparse
import numpy as np
import torch
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os

from src.models.cancer_seg.CNN import CNN

from sys import getsizeof
import argparse
import pathlib
import nrrd
import matplotlib.pyplot as plt
import random
import tqdm



parser = argparse.ArgumentParser()

parser.add_argument(
    "-model",
    default = '3dCNN',
    type=str
)

parser.add_argument(
    "-version",
    default = 'best_so_far',
    type=str
)

parser.add_argument(
    "-data",
    default = 'test',
    type=str
)

parser.add_argument(
    "-threshold",
    default = 0.0,
    type = float
)

args = parser.parse_args()

CONTINUE = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


print("Loading model")

if args.model == '3dCNN':
    from src.models.cancer_seg.CNN import CNN
    print("Loading model version")

    if args.version == 'best_so_far':
        model = CNN().to(device=device)
        model.load_state_dict(torch.load('models/cancer_seg/best_so_far.pth'))
        #model.double()
        model.eval()

    else:
        print('Version label not found')
        CONTINUE = False

else:
    print('Model label not found')
    CONTINUE = False



print('Loading data')
if CONTINUE:
    class MyDataset(Dataset):

        def __init__(self, csv_path, vol_dir, transform = None):
            
            df = pd.read_csv(csv_path)#, dtype={'Class Label':int, 'File Name': str})
            df = df[df['Class Label'] != 'Class Label']
            df['Class Label'] = df['Class Label'].astype('float') 
            df['File Name'] = df['File Name'].astype('str')
            df = df.drop(columns = df.columns[0])
            
            self.vol_dir = vol_dir
            self.vol_names = df['File Name']
            self.y = df['Class Label']
            self.transform = transform

        def __getitem__(self, index):
            vol = np.load(os.path.join(self.vol_dir, self.vol_names[index]))#.astype(np.double)
            if self.transform is not None:
                vol = self.transform(vol)

            # label = self.y[index]
            if self.y[index] == 0.0:
                label = np.array([0.])
            else:
                label = np.array([1.])
            return vol, label
        
        def __len__(self):
            return self.y.shape[0]

    if args.data == 'train':
        print('Data: train')
        dataset = MyDataset(csv_path = 'slices/T_16/train.csv',
                                vol_dir = 'slices/T_16/train',
                                transform = None)

        loader = DataLoader(dataset = dataset,
                                batch_size = 64,
                                drop_last = True,
                                shuffle = True,
                                num_workers = 16)

    elif args.data == 'test':
        dataset = MyDataset(csv_path = 'slices/T_16/test.csv',
                                vol_dir = 'slices/T_16/test',
                                transform = None)

        loader = DataLoader(dataset = dataset,
                                batch_size = 64,
                                drop_last = True,
                                shuffle = True,
                                num_workers = 16)

    else:
        print('Dataset label not found')
        CONTINUE = False


if CONTINUE:

    if args.threshold == 0.0:
        print('Testing thresholds')
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        for threshold in thresholds:
            
            normals = 0
            tumors = 0
            errors1 = 0
            errors2 = 0
            
            model.eval()

            losses = []
            total_number = int(len(dataset)/16)

            for batch_idx, (x,y) in enumerate(loader):
                x = x.to(device)
                y = y.to(device)
                
                normals += (y == 0).sum()
                tumors += (y == 1).sum()
                scores = model(x)
                decisions = 1*(scores > threshold)

                err = y - decisions
                errors1 += len(err[err == 1])
                errors2 += len(err[err == -1])
            
            print( f'For {threshold}: \n 0as0 = {normals - errors2}, 0as1 = {errors2}, \n 1as0 = {errors1}, 1as1 = {tumors - errors1} \n ---------------- \n' )


    else:
        print('...', end = '\r')
        normals = 0
        tumors = 0
        errors1 = 0
        errors2 = 0

        model.eval()

        losses = []
        total_number = int(len(dataset)/16)

        for batch_idx, (x,y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)
            
            normals += (y == 0).sum()
            tumors += (y == 1).sum()
            
            scores = model(x)
            decisions = 1*(scores > args.threshold)

            err = y - decisions
            errors1 += len(err[err == 1])
            errors2 += len(err[err == -1])
        
        print( f'For {args.threshold}: \n 0as0 = {normals - errors2}, 0as1 = {errors2}, \n 1as0 = {errors1}, 1as1 = {tumors - errors1}' )