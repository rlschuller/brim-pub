import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np

from src.models.pipeline_breast_seg.unet3d import UNET

import argparse
import pathlib
import nrrd
import matplotlib.pyplot as plt

#arguments
parser = argparse.ArgumentParser()

parser.add_argument(
    "-i",
    default = './data/processed/nrrd_segs/full_data',
    type=str
)

parser.add_argument(
    "-o",
    default = './models/nrrd_segs/testes',
    type=str
)

parser.add_argument(
    "-m",
    default = './model/breast_seg/best_model_temp.pth',
    type=str
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

num_train = args.tr


#device
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print(device)
device = torch.device('cpu')

#load data
folder = pathlib.Path(input_path)
exams = [exam for exam in folder.glob('*')]
names = [e.name for e in exams]
names = sorted(names, key = lambda x: int(x))
names_0 = names[:num_train]
names_1 = names[num_train:]
del exams



#iniciar o modelo
model = UNET().to(device)
model.load_state_dict(torch.load('./models/breast_seg/best_model_sub26.pth'))
model.double()
model.eval()



for name in names:
    print(name)

    exam, _ = nrrd.read(input_path + f'/{name}/A1.nrrd')
    birads , _ = nrrd.read(input_path + f'/{name}/birads.nrrd')
    seg = np.zeros(exam.shape)

    for k in range(0, exam.shape[0] - 8, 4):

        x = exam[k : k + 8, :, :]
        x = x.astype(float)
        x = torch.tensor(x)
        x = torch.reshape(x, [1, 1, 8, exam.shape[1], exam.shape[2]])

        y = model(x)
        y = y.view(y.shape[2:]).detach().numpy()
        y = 1*(y > 0.5)

        seg[k : k + 8, :, :] += y
        seg[k : k + 8, :, :] = (seg[k : k + 8, :, :] >= 1) * 1

    breakpoint()
