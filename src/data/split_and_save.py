import numpy as np
import argparse
import pathlib
import nrrd
import tqdm
import torch
import matplotlib.pyplot as plt
import csv
import pandas as pd


parser = argparse.ArgumentParser()

parser.add_argument(
    "-i",
    default = './data/processed/nrrd_segs/full_data/',
    type=str
)

parser.add_argument(
    "-o",
    default = './data/processed/pre_cuts/',
    type=str
)

parser.add_argument(
    "-d",
    default = (16,16,16),
    type=tuple
)

parser.add_argument(
    "-s",
    default = 8,
    type=tuple
)

args = parser.parse_args()

input_path = args.i
output_parent_folder = args.o
dimensions = args.d
stride = args.s


#criando pastas novas
output_folder = output_parent_folder + f'{dimensions[0]}_{dimensions[1]}_{dimensions[2]}_{stride}_t4/'

train_output_folder = output_folder + 'train/'
test_output_folder = output_folder + 'test/'

output_folder  = pathlib.Path(output_folder)

if not output_folder.exists():
    # pathlib.Path(train_output_folder).mkdir(parents = True)
    pathlib.Path(test_output_folder).mkdir(parents = True)
    pd.DataFrame(columns = ['Class Label', 'File Name']).to_csv(output_parent_folder + f'{dimensions[0]}_{dimensions[1]}_{dimensions[2]}_{stride}/' + 'train.csv' )
    pd.DataFrame(columns = ['Class Label', 'File Name']).to_csv(output_parent_folder + f'{dimensions[0]}_{dimensions[1]}_{dimensions[2]}_{stride}/' + 'test.csv')

    #load and split exam function
    def cut_exam(exam_path, name, pth, dimensions = dimensions, stride = stride):
        print(f'loading: {name}')
        A1, _ = nrrd.read(f'{exam_path}/A1.nrrd')
        A2, _ = nrrd.read(f'{exam_path}/A2.nrrd')
        A3, _ = nrrd.read(f'{exam_path}/A3.nrrd')
        A4, _ = nrrd.read(f'{exam_path}/A4.nrrd')
        A5, _ = nrrd.read(f'{exam_path}/A5.nrrd')
        
        data_csv = pd.DataFrame(columns = ['Class Label', 'File Name'])
        
        print(f'cutting: {name}')

        seg1 , _ = nrrd.read(f'{exam_path}/s4_pred.nrrd')
        seg2 = np.zeros(A1.shape)
        for i in range(0, A1.shape[0], 8):
            for j in range(0, A1.shape[1], 8):
                for k in range(0, A1.shape[2], 8):
                    if A1[i:i+8, j:j+8, k:k+8].mean() > 10:
                        seg2[i:i+8, j:j+8, k:k+8] = 1
        
        seg = np.logical_and(seg1,seg2) * 1
        
        del seg1
        del seg2

        y, _ = nrrd.read(f'{exam_path}/birads.nrrd')
        

        x = np.array([A1, A2, A3, A4, A5, y, seg])
        x = x.astype(float)
        
        temp3 = []
        for i in range(0, A1.shape[0] - dimensions[0], stride):
            for j in range(0, A1.shape[1] - dimensions[1], stride):
                for k in range(0, A1.shape[2] - dimensions[2], stride):
                    if A1[i:i+dimensions[0], j:j+dimensions[1], k:k+dimensions[2]].mean() > 10:
                        temp3.append(x[:, i:i+dimensions[0], j:j+dimensions[1], k:k+dimensions[2]])

        #using breast_seg mask
        temp3 = np.array(temp3)
        mask = [1 in x for x in temp3[:, -1:, :, :, :]]
        temp3 = temp3[mask, :-1, :, :, :]
        
        j = -1

        print(f'labeling: {name}')

        for block in temp3:
            if np.count_nonzero(block[-1] != 0) >= 4: 
                cube_label = 1

                np.save(train_output_folder + f'{key}_{j}.npy', block[:-1])
                data_csv = data_csv._append({"Class Label":cube_label, "File Name": f'{key}_{j}.npy'}, ignore_index=True)

            elif 0 < np.count_nonzero(block[-1] != 0) < 4: 
                continue
            
            else:
                cube_label = 0

                np.save(train_output_folder + f'{key}_{j}.npy', block[:-1])
                data_csv = data_csv._append({"Class Label":cube_label, "File Name": f'{key}_{j}.npy'}, ignore_index=True)


        return data_csv

    #load data
    folder = pathlib.Path(input_path)
    exams = [exam for exam in folder.glob('*')]
    names = [e.name for e in exams]
    names = sorted(names, key = lambda x: int(x))
    del exams

    tsh = int(len(names) * (4/5))
    from multiprocessing import Pool


    for name in tqdm.tqdm(names[:tsh]):
        data_csv = cut_exam(exam_path = input_path + f'/{name}', name = name, pth = train_output_folder)
        data_csv.to_csv(output_parent_folder + f'{dimensions[0]}_{dimensions[1]}_{dimensions[2]}_{stride}/train.csv', mode='a')
    
    names = names[tsh:]
    
    def f(name):
        data_csv = cut_exam(exam_path = input_path + f'/{name}', name = name, pth = test_output_folder)
        data_csv.to_csv(output_parent_folder + f'{dimensions[0]}_{dimensions[1]}_{dimensions[2]}_{stride}/train.csv', mode='a')

    with Pool(5) as p:
        p.map(f, names[tsh:])

    
    def f(name):
        data_csv = cut_exam(exam_path = input_path + f'/{name}', name = name, pth = test_output_folder)
        data_csv.to_csv(output_parent_folder + f'{dimensions[0]}_{dimensions[1]}_{dimensions[2]}_{stride}/test.csv', mode='a')

    with Pool(5) as p:
        p.map(f, names[tsh:])

    # for name in tqdm.tqdm(names):#[tsh:]):
    #     data_csv = cut_exam(exam_path = input_path + f'/{name}', name = name, pth = test_output_folder)
    #     data_csv.to_csv(output_parent_folder + f'{dimensions[0]}_{dimensions[1]}_{dimensions[2]}_{stride}/test2.csv', mode='a')


else:
    print('Já existe esse corte. Caso queira fazer mesmo assim, delete a pasta primeiro e tente novamente.')

