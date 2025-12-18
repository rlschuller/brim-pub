import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np

from unet3d import UNET

import argparse
import pathlib
import nrrd
import matplotlib.pyplot as plt

#arguments
parser = argparse.ArgumentParser()

parser.add_argument(
    "-i",
    default = './data/processed/nrrd_segs/full_data/',
    type=str
)


parser.add_argument(
    "-o",
    default = './models/cancer_seg/',
    type=str
)

parser.add_argument(
    "-m",
    default = 'model.pth',
    type=str
)

parser.add_argument(
    "-ep",
    default = 16,
    type=int
)

parser.add_argument(
    "-lr",
    default = 0.000001,
    type=float
)

parser.add_argument(
    "-bs",
    default = 16,
    type=int
)

parser.add_argument(
    "-tr",
    default = 160,
    type=int
)



args = parser.parse_args()

input_path = args.i
output_path = args.o
model_name = args.m
learning_rate = args.lr
batch_size = args.bs
num_epochs = args.ep
num_train = args.tr


#device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

#load data
folder = pathlib.Path(input_path)
exams = [exam for exam in folder.glob('*')]
names = [e.name for e in exams]
names = sorted(names, key = lambda x: int(x))
names = names[:num_train]
del exams


def load_exam(exam_path, seg_path):
    exam, _ = nrrd.read(exam_path)
    seg , _ = nrrd.read(seg_path)

    stride_k = 8
    
    X = torch.tensor([])
    Y = torch.tensor([])

    for k in range(0, exam.shape[0] - stride_k, stride_k):

        x = exam[k : k + stride_k, :, :]
        x = x.astype(float)
        x = torch.tensor(x)
        x = torch.reshape(x, [1, 1, stride_k, exam.shape[1], exam.shape[2]])

        y = torch.from_numpy(seg[k : k + stride_k, :, :]).reshape([1, 1, stride_k, exam.shape[1], exam.shape[2]])

        X = torch.cat((X,x))
        Y = torch.cat((Y,y))

    return X,Y


#iniciar o modelo
model = UNET().to(device)
model.load_state_dict(torch.load('./models/breast_seg/best_model_sub26.pth'))
model.double()

#função de perda
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#treinar network
model.train()

all_time_low = 10000
all_time_low_T = 10000

bmsf = 0
bmsf_T = 0

losses = []
losses_T = []
hummm = []

for epoch in range(num_epochs):
    for name in names:
        print(name)
        
        
        data, target = load_exam(exam_path = input_path + f'/{name}/A1.nrrd' , seg_path = input_path + f'/{name}/breast_mask.seg.nrrd')
        target = target.type(torch.DoubleTensor)
        sample_size = data.shape[0]
        number_of_batches = (sample_size//batch_size)
        
        
        for batch_round in range(3):
            T = 0
            for batch_idx in range(number_of_batches):
                #jogar pro cuda (Se possível)
                batch_data = data[batch_idx:batch_idx+batch_size].to(device=device)
                target_data = target[batch_idx:batch_idx+batch_size].to(device=device)

                #forward
                scores = model(batch_data)
                loss = criterion(scores, target_data)

                #backward
                optimizer.zero_grad()
                loss.backward()

                #gradient descent or adam step
                optimizer.step()
                losses.append(loss.item())
                T+=loss.item()
                del batch_data
                del target_data
                del loss
                del scores
            
            plt.plot(range(len(losses)), losses, 'o', color = 'b')
            plt.plot(range(len(losses)), losses, '-', color = 'b', alpha = 0.2)
            plt.title('Train')
            plt.savefig('./images/train_losses.png')
            plt.close()

            if T > all_time_low_T * 2:
                if name not in hummm:
                    hummm.append(name)
                    print(hummm)


            losses_T.append(T)
            if T < all_time_low_T:
                all_time_low_T = T
                torch.save(model.state_dict(), output_path + f'best_' + model_name)
                bmsf_T += 1
        
        
        print(T)
        plt.plot(range(len(losses_T)), losses_T, 'o', color = (0,0,1))
        plt.title('Train')
        plt.savefig('./images/train_T_losses.png')
        plt.close()


    torch.save(model.state_dict(), output_path + f'{epoch}_' + model_name)

torch.save(model.state_dict(), output_path + 'final_' + model_name)