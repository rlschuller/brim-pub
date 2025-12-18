import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np

from CNN import CNN

from sys import getsizeof
import argparse
import pathlib
import nrrd
import matplotlib.pyplot as plt
import random
import tqdm

random.seed(2024)

#arguments
parser = argparse.ArgumentParser()


parser.add_argument(
    "-i",
    type=str
)

parser.add_argument(
    "-output_path",
    default = './data/processed/nrrd_segs/testes2',
    type=str
)

parser.add_argument(
    "-model_dict",
    default = './models/mass_seg/epoch_0_model.pth',
    type=str
)



args = parser.parse_args()
out = args.output_path
model_path = args.model_dict
input_exam = args.i


#device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)



def load_exam(exam_path):

    A1, meta = nrrd.read(f'{exam_path}/A1.nrrd')
    A2, _ = nrrd.read(f'{exam_path}/A2.nrrd')
    A3, _ = nrrd.read(f'{exam_path}/A3.nrrd')
    A4, _ = nrrd.read(f'{exam_path}/A4.nrrd')
    A5, _ = nrrd.read(f'{exam_path}/A5.nrrd')

    flesh = (A1 > 8)
    seg , _ = nrrd.read(f'{exam_path}/s4_pred.nrrd')
    seg = 1 * np.logical_and(seg == 1, flesh)
        
    birads , _ = nrrd.read(f'{exam_path}/birads.nrrd')

    x = np.array([A1, A2, A3, A4, A5])
    x = x.astype(float)
    x = torch.from_numpy(x)
    x = torch.reshape(x,[1,*x.shape])
    return x, seg, birads, meta, A1


#iniciar o modelo
model = CNN().to(device=device)
model.load_state_dict(torch.load(model_path))
model.double()
model.eval()

def apply_model_in(exam_path, fold, output_path, cube_size = 16):
    data, seg, birads, meta, A1 = load_exam(exam_path = exam_path)
    cancers = np.zeros(birads.shape)

    for i in range(0, data.shape[2]-cube_size, cube_size//2):
        for j in range(0, data.shape[3]-cube_size, cube_size//2):
            xx = data[:, :, i:i+cube_size, j:j+cube_size, :].split(cube_size, dim = -1)
            xx  = torch.stack(list(xx), dim=0)
            xx = xx[:, 0, :, :, :, :].to(device=device)
            score = model(xx)
            for s in range(len(score)):
                cancers[i:i+cube_size, j:j+cube_size, (s)*cube_size:(s+1)*cube_size] += (score[s]).item()

            # for k in range(0, data.shape[4]-cube_size, cube_size//2):
            #     if 1 in seg[i:i+cube_size, j:j+cube_size, k:k+cube_size]:
            #             breakpoint()
            #             score = model(data[:, :, i:i+cube_size, j:j+cube_size, k:k+cube_size].to(device=device))
            #             cancers[i:i+cube_size, j:j+cube_size, k:k+cube_size] += (score[0]).item() #[1].item() - score[0][0]).item()


    m = np.min(cancers)
    M = np.max(cancers)
    
    cancers -= m

    data = data[0][0]

    nrrd.write(f'{output_path}/{fold}_data.nrrd', data.numpy(), meta)
    nrrd.write(f'{output_path}/{fold}_target.seg.nrrd', birads, meta)
    #np.save(f'{output_path}/{fold}_scores.npy', cancers)
    nrrd.write(f'{output_path}/{fold}_cancers.seg.nrrd', 1*(cancers > 6), meta)
    nrrd.write(f'{output_path}/{fold}_scores.seg.nrrd', cancers, meta)
    
    return cancers, meta

if input_exam == 'test':
    import pandas as pd
    print('A')
    input_exams = []
    input_csv = pd.read_csv('data/processed/pre_cuts/16_16_16_8/test.csv')
    input_csv = input_csv['File Name']
    icsv = [a.split('_')[0] for a in input_csv]
    icsv = list(set(icsv))
    print('B')

    for input_ex in icsv:
        print(input_ex)
        cancers, meta = apply_model_in(exam_path = f'./data/processed/nrrd_segs/full_data/{input_ex}', fold = input_ex, output_path = out)

else:
     cancers, meta = apply_model_in(exam_path = f'./data/processed/nrrd_segs/full_data/{input_exam}', fold = input_exam, output_path = out)

        #print('max: ', max(cancers), end = '')
        #print('| min: ', min(cancers))


# idx = 0
# for k in [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9]:
#     idx += 1
#     K = (cancers >= k) * 1
#     nrrd.write(f'{out}/{input_exam}_cancers{idx}.seg.nrrd', K, meta)


    

