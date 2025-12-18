from classifier import model

import torch

from tqdm import tqdm
import random
import nrrd
import numpy as np
import matplotlib.pyplot as plt

import pathlib
import matplotlib.pyplot as plt

#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
print(f'device = {device}')

#build dataset
ex_paths = './personal/classifier/data/'
lote_path = pathlib.Path(ex_paths)
lotes = [path for path in lote_path.glob("*")]

# lotes_treino = lotes[:10]
lotes_teste = lotes[10:]

#define X_Y_loader
def X_Y_loader(exame, balanced = False):
    
    X = []
    Y = []

    paths = exame.glob("*")
    mask_not_0 = []

    for path in paths:
        label = str(path)[-6]
        
        ex = (nrrd.read(str(path)))[0].astype(dtype=np.float16)
        max_in_ex = np.max(ex)
        
        if max_in_ex == 0:
            continue

        else:
            if label == '0':
                Y.append(torch.tensor([1,0]))
                mask_not_0.append(False)
            else:
                Y.append(torch.tensor([0,1]))
                mask_not_0.append(True)

            ex /= max_in_ex
            X.append(torch.from_numpy(ex))
    
    
    if balanced:
        not_zeros = sum(mask_not_0)
        if not_zeros == 0:
            return X, Y, False

        Xb = []
        Yb = []

        while sum(mask_not_0) < 2*not_zeros:
            mask_not_0[random.randint(0, len(mask_not_0)-1)] = True

        for i in range(len(mask_not_0)):
            if mask_not_0[i] == True:
                Xb.append(X[i])
                Yb.append(Y[i])

        return Xb, Yb, True

    return X, Y, True



for mod in ['cnn_E4_L10','cnn_E3_L10','cnn_E2_L10','cnn_E1_L10']:
    print(mod)
    print(f'loading model...')
    model_ = model().to(device)
    model_.load_state_dict(torch.load(f'./personal/classifier/model/{mod}.pth'))
    model_.eval()
    print('OK')


    scores = dict()

    scores['T0'] = 0
    scores['F0'] = 0

    scores['T1'] = 0
    scores['F1'] = 0

    for lote in lotes_teste:
        for exame in lote.glob("*"):
            print(scores, end = '\r')
            X, Y, s = X_Y_loader(exame)

            for i in range(len(X)):

                Xi = torch.stack(X[i:i+1]).to(device, dtype=torch.float)
                labels = torch.stack(Y[i:i+1]).to(device, dtype=torch.float)
                predicted_labels = model_(Xi)

                if predicted_labels[0][0] > 0.5:
                    if all(labels[0] == torch.tensor([1,0]).to(device)):
                        scores['T0'] += 1
                    else:
                        scores['F0'] += 1
                
                else:
                    if all(labels[0] == torch.tensor([0,1]).to(device)):
                        scores['T1'] += 1
                    else:
                        scores['F1'] += 1

            
    print('\n')
    print(mod, scores)

breakpoint()