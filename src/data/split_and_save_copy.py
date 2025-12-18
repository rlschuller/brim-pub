import numpy as np
import argparse
import pathlib
import h5py
import pandas as pd


parser = argparse.ArgumentParser()

parser.add_argument(
    "-i",
    default = './hdf5/',
    type=str
)

parser.add_argument(
    "-o",
    default = './data/processed/pre_cuts/',
    type=str
)

parser.add_argument(
    "-side",
    default = 16,
    type=tuple
)

CONTINUE = True

args = parser.parse_args()

input_path = args.i
output_parent_folder = args.o
side = args.side

#criando pastas novas
output_folder = output_parent_folder + f'C_{side}/'

train_output_folder = output_folder + 'train/'
test_output_folder = output_folder + 'test/'

if not pathlib.Path(output_folder).exists():

    pathlib.Path(train_output_folder).mkdir(parents = True)
    pathlib.Path(test_output_folder).mkdir(parents = True)
    
    # pd.DataFrame(columns = ['Class Label', 'File Name']).to_csv(output_folder + f'train.csv' )
    # pd.DataFrame(columns = ['Class Label', 'File Name']).to_csv(output_folder + f'test.csv')

else:
    print('Já existe esse corte. Deletar existente e continuar? (s/n).')
    aws = input()
    if (aws == 's') or (aws == 'y') or (aws == 'S') or (aws == 'Y'):
        def delete_folder(pth):
            for sub in pth.iterdir():
                if sub.is_dir():
                    delete_folder(sub)
                else:
                    sub.unlink()
            pth.rmdir()

        delete_folder(pathlib.Path(output_folder))

        pathlib.Path(train_output_folder).mkdir(parents = True)
        pathlib.Path(test_output_folder).mkdir(parents = True)

    else:
        CONTINUE = False




if CONTINUE:
    f = h5py.File('./data/processed/hdf5/processed.hdf5')
    g = h5py.File('./data/processed/hdf5/breast_segmentation.hdf5')

    keys = sorted(f['exams'].keys(), key = lambda x: int(x))

    print('Train folder')
    data_csv = pd.DataFrame(columns = ['Class Label', 'File Name'])
    for key in keys[:676]:
        print(key)
        
        exam = np.array(f['exams'][key])
        breast = 1*np.array(g[key])
        
        exam[0] = exam[0] * breast
        exam[1] = exam[1] * breast
        exam[2] = exam[2] * breast
        exam[3] = exam[3] * breast
        exam[4] = exam[4] * breast

        tumors = np.array(f['birads'][key])
        tumors = np.resize(tumors, [1, *tumors.shape])
        
        full = np.append(exam, tumors, axis = 0)
        
        temp3 = []

        for i in range(0, full.shape[1] - side, side):
            for j in range(0, full.shape[2] - side, side):
                for k in range(0, full.shape[3] - side, side):
                    if breast[i:i+side, j:j+side, k:k+side].any():

                        temp3.append(full[:, i:i+side, j:j+side, k:k+side])
        
        j = 0

        for block in temp3:
            j+=1
            cube_label = 0
            #if (3 in block[-1]) or (4 in block[-1]): #MUDAR AQUI PARA QUANDO FOR PARA DIFERENCIAR ENTRE 3 E 4
            if np.count_nonzero(block[-1] != 0) >= 4: 
                cube_label = 1

                np.save(train_output_folder + f'{key}_{j}.npy', block[:-1])
                data_csv = data_csv._append({"Class Label":cube_label, "File Name": f'{key}_{j}.npy'}, ignore_index=True)

            elif 0 < np.count_nonzero(block[-1] != 0) < 4: 
                continue

            else:
                np.save(train_output_folder + f'{key}_{j}.npy', block[:-1])
                data_csv = data_csv._append({"Class Label":cube_label, "File Name": f'{key}_{j}.npy'}, ignore_index=True)
        
    data_csv.to_csv(output_folder + 'train.csv')


    print('Test folder')
    data_csv = pd.DataFrame(columns = ['Class Label', 'File Name'])
    for key in keys[676:]:
        print(key)

        exam = np.array(f['exams'][key])
        breast = 1*np.array(g[key])
        
        exam[0] = exam[0] * breast
        exam[1] = exam[1] * breast
        exam[2] = exam[2] * breast
        exam[3] = exam[3] * breast
        exam[4] = exam[4] * breast

        tumors = np.array(f['birads'][key])
        tumors = np.resize(tumors, [1, *tumors.shape])
        
        full = np.append(exam, tumors, axis = 0)
        
        temp3 = []

        for i in range(0, full.shape[1] - side, side):
            for j in range(0, full.shape[2] - side, side):
                for k in range(0, full.shape[3] - side, side):
                    if full[:, i:i+side, j:j+side, k:k+side].any():

                        temp3.append(full[:, i:i+side, j:j+side, k:k+side])
        
        j = 0

        for block in temp3:
            j+=1
            cube_label = 0
            #if (3 in block[-1]) or (4 in block[-1]): #MUDAR AQUI PARA QUANDO FOR PARA DIFERENCIAR ENTRE 3 E 4
            # if np.count_nonzero(block[-1] != 0) >= 4: 
            #     cube_label = 1

            # np.save(output_folder + f'test/{key}_{j}.npy', block[:-1])
            # data_csv = data_csv._append({"Class Label":cube_label, "File Name": f'{key}_{j}.npy'}, ignore_index=True)

            if np.count_nonzero(block[-1] != 0) >= 4: 
                cube_label = 1

                np.save(test_output_folder + f'{key}_{j}.npy', block[:-1])
                data_csv = data_csv._append({"Class Label":cube_label, "File Name": f'{key}_{j}.npy'}, ignore_index=True)

            elif 0 < np.count_nonzero(block[-1] != 0) < 4: 
                continue

            else:
                np.save(test_output_folder + f'{key}_{j}.npy', block[:-1])
                data_csv = data_csv._append({"Class Label":cube_label, "File Name": f'{key}_{j}.npy'}, ignore_index=True)
        
    data_csv.to_csv(output_folder + 'test.csv')


