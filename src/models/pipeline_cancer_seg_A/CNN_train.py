import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np

from CNN import CNN_test

from sys import getsizeof
import argparse
import pathlib
import nrrd
import matplotlib.pyplot as plt
import random

random.seed(2024)

#arguments
parser = argparse.ArgumentParser()

parser.add_argument(
    "-i",
    default = './data/processed/nrrd_segs/full_data/',
    type=str
)

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
    default = 32,
    type=int
)

parser.add_argument(
    "-lr",
    default = 1e-6,
    type=float
)

parser.add_argument(
    "-bs",
    default = 16,
    type=int
)

parser.add_argument(
    "-tr",
    default = 360,
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
names = names[::-1]
del exams


def label_this(y, full = True):
    if full:
        if 4 in y:
            return [0,0,1]
        elif 3 in y:
            return [0,1,0]
        else:
            return [1,0,0]
    else:
        if 4 in y:
            return [0,1]
        elif 3 in y:
            return [0,1]
        else:
            return [1,0]


def load_exam(exam_path, cube_size = 32, full = True):

    A1, _ = nrrd.read(f'{exam_path}/A1.nrrd')
    A2, _ = nrrd.read(f'{exam_path}/A2.nrrd')
    A3, _ = nrrd.read(f'{exam_path}/A3.nrrd')
    A4, _ = nrrd.read(f'{exam_path}/A4.nrrd')
    A5, _ = nrrd.read(f'{exam_path}/A5.nrrd')

    seg , _ = nrrd.read(f'{exam_path}/s4_pred.nrrd')

    y, _ = nrrd.read(f'{exam_path}/birads.nrrd')
    y = 1 * (y>0.1)

    x = np.array([A1, A2, A3, A4, A5, y, seg])
    x = x.astype(float)
    x = torch.from_numpy(x)

    temp1 = []
    for x1 in x.split(cube_size, dim = 1):
        temp1.append(x1)
    
    temp2 = []
    for x1 in temp1:
        for x2 in x1.split(cube_size, dim = 2):
            temp2.append(x2)

    temp3 = []
    for x2 in temp2:
        for x3 in x2.split(cube_size, dim = 3):
            if x3.shape == torch.Size([7, cube_size, cube_size, cube_size]):
                temp3.append(x3)
    

    #using breast_seg mask
    temp3 = np.array(temp3)
    mask = [1 in x for x in temp3[:, -1:, :, :]]
    temp3 = temp3[mask, :-1, :, :, :]
    

    X = torch.from_numpy(temp3[:, :-1, :, :, :])

    Y = temp3[:, -1, :, :, :]
    Y = np.array([label_this(y, full = full) for y in Y])
    Y = torch.from_numpy(Y)
    #Y = torch.tensor([1 in y for y in Y])
    if full:
        Y = torch.reshape(Y, [Y.shape[0], 3])
    else:
        Y = torch.reshape(Y, [Y.shape[0], 2])
    return X, Y


#iniciar o modelo
model = CNN_test().to(device)
model.double()

#função de perda
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)

weights = torch.tensor([1.0, 1e5]).to(device=device)#, 40000.0]).to(device=device)
criterion = nn.CrossEntropyLoss(weight = weights)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
#optimizer = optim.SGD(model.parameters(), lr=learning_rate)

#treinar network
model.train()

all_time_low = 10000
all_time_low_T = 10000

bmsf = 0

losses = []

for epoch in range(num_epochs):

    for name in names:
        print(f'{name}', end = '\r')
        
        
        data, target = load_exam(exam_path = input_path + f'/{name}', full = False)
        target = target.type(torch.DoubleTensor)
        sample_size = data.shape[0]
        number_of_batches = (sample_size//batch_size) + 1
        

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
            T+=loss.item()
            
            del batch_data
            del target_data
            del loss
            del scores
            
        T /= number_of_batches
        
        losses.append(T)
        print(f'{name}-{T}', end = '\n')
        
        # torch.save(model.state_dict(), output_path + f'last_trained_' + model_name)

        # if T < all_time_low_T:
        #     all_time_low_T = T
        #     torch.save(model.state_dict(), output_path + f'{bmsf}th_best_' + model_name)
        #     bmsf += 1

        plt.plot(range(len(losses)), losses, 'o', color = (0,0,1))
        plt.title('Train')
        plt.ylim(-0, 0.00501 )
        # plt.savefig('./images/train_losses_zoom.png')
        plt.close()

        plt.plot(range(len(losses)), losses, 'o', color = (0,0,1))
        plt.title('Train')
        # plt.savefig('./images/train_losses.png')
        plt.close()

    # torch.save(model.state_dict(), output_path + f'{epoch}_' + model_name)

# torch.save(model.state_dict(), output_path + 'final_' + model_name)