import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os

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
        # if self.transform is not None:
        #     vol = self.transform(vol)
        
        label = self.y[index]
        return vol, label
    
    def __len__(self):
        return self.y.shape[0]


train_dataset = MyDataset(csv_path = 'data/processed/pre_cuts/16_16_16_4/train.csv',
                          vol_dir = 'data/processed/pre_cuts/16_16_16_4/train',
                          transform = None)

train_loader = DataLoader(dataset = train_dataset,
                          batch_size = 32,
                          drop_last = True,
                          shuffle = True,
                          num_workers = 6)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_epochs = 2

for epoch in range(num_epochs):
    print(epoch)
    for batch_idx, (x,y) in enumerate(train_loader):

        print("Epoch:", epoch+1, end = '')
        print(' | Batch index:', batch_idx, end = '')
        print(' |Batch size:', y.size()[0])

        x = x.to(device)
        y = y.to(device)