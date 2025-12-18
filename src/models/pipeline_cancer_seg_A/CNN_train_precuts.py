import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os

from CNN import CNN_test

from sys import getsizeof
import argparse
import pathlib
import nrrd
import matplotlib.pyplot as plt
import random
import tqdm



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
        vol = np.load(os.path.join(self.vol_dir, self.vol_names[index]))
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




#arguments
parser = argparse.ArgumentParser()

parser.add_argument(
    "-o",
    default = './models/mass_seg/',
    type=str
)

parser.add_argument(
    "-m",
    default = 'model.pth',
    type=str
)

parser.add_argument(
    "-ep",
    default = 64,
    type=int
)

parser.add_argument(
    "-lr",
    default = 1e-2,
    type=float
)

parser.add_argument(
    "-bs",
    default = 32,
    type=int
)



args = parser.parse_args()

output_path = args.o
model_name = args.m
learning_rate = args.lr
batch_size = args.bs
num_epochs = args.ep


#device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


train_dataset = MyDataset(csv_path = 'data/processed/pre_cuts/16_16_16_8/train.csv',
                          vol_dir = 'data/processed/pre_cuts/16_16_16_8/train',
                          transform = None)

train_loader = DataLoader(dataset = train_dataset,
                          batch_size = 16,
                          drop_last = True,
                          shuffle = True,
                          num_workers = 20)


# test_dataset = MyDataset(csv_path = 'data/processed/pre_cuts/16_16_16_8/test.csv',
#                           vol_dir = 'data/processed/pre_cuts/16_16_16_8/test',
#                           transform = None)

# test_loader = DataLoader(dataset = test_dataset,
#                           batch_size = 1,
#                           drop_last = True,
#                           shuffle = True,
#                           num_workers = 6)

#iniciar o modelo
model = CNN_test().to(device)
model.double()

#função de perda


# weights = torch.tensor([1.0, 5.0]).to(device=device)#, 40000.0]).to(device=device)
# criterion = nn.CrossEntropyLoss(weight = weights)
criterion = nn.MSELoss()
# optimizer = optim.SGD(model.parameters(), lr=learning_rate)
optimizer = optim.Adam(model.parameters(), lr=1e-5)
#optimizer = optim.SGD(model.parameters(), lr=learning_rate)

#treinar network
model.train()

bmsf = 0

losses = []
total_number = int(len(train_dataset)/16)

j = 1

for epoch in range(num_epochs):
    print(epoch)
    for batch_idx, (x,y) in enumerate(train_loader):
        if (torch.tensor([1.]) in y):
            x = x.to(device)
            y = y.to(device)


            scores = model(x)
            loss = criterion(scores, y)

            #backward
            optimizer.zero_grad()
            loss.backward()

            #gradient descent or adam step
            optimizer.step()
        


            scores
            decisions = 1*(scores>=0.5)

            err = decisions - y
            errors1 = len(err[err == 1])
            errors2 = len(err[err == -1])

            print("Epoch:", epoch+1, end = '')
            print(' | Batch:', batch_idx, '/', total_number, end = '')
            print(' |Batch size:', y.size()[0], end = '')
            print(' |Number of tumors:', (y!=0.).sum().item(), end = '')
            print(f' |Errors: {errors2} 1as0,  {errors1} 0as1', end = '')
            print(' |Batch error:', loss.item())

            del x
            del y
            del loss
            del scores

        
    torch.save(model.state_dict(), output_path + f'last_trained_' + model_name)
    torch.save(model.state_dict(), output_path + f'epoch_{epoch}_' + model_name)


#criterion = nn.MSELoss()
    # model.eval()
    # for batch_idx, (x,y) in enumerate(test_loader):

    #     x = x.to(device)
    #     y = y.to(device)

    #     scores = model(x)
    #     if scores.item() >= 0.5:
    #         scores = 1
    #     else:
    #         scores = 0
        

    # model.train()