import nrrd
import numpy as np
import torch
import pathlib
from multiprocessing import Pool

folder = './data/processed/nrrd'
paths = pathlib.Path(folder)
exam_paths = [path for path in paths.glob("*/*")]

side = 32
stride = 16
limit = 4

def create(exam_path):
    if (pathlib.Path(str(exam_path)+'/birads.nrrd').exists()):
        # loading data
        volA1, _ = nrrd.read(str(exam_path)+'/A1.nrrd')
        volA2, _ = nrrd.read(str(exam_path)+'/A2.nrrd')
        volA3, _ = nrrd.read(str(exam_path)+'/A3.nrrd')
        volA4, _ = nrrd.read(str(exam_path)+'/A4.nrrd')
        volA5, _ = nrrd.read(str(exam_path)+'/A5.nrrd')

        vol = np.array((volA1,volA2,volA3,volA4,volA5))

        del volA1
        del volA2
        del volA3
        del volA4
        del volA5

        volBIRADS, _ = nrrd.read(str(exam_path)+'/birads.nrrd')
        
        # verificar se o caminho de output existe. Caso exista limpa ele, caso o contrário cria o caminho
        OutputPath = pathlib.Path(f'./personal/classifier/data/'+ str(exam_path).split('/')[-2]+'/'+str(exam_path).split('/')[-1])
        
        if OutputPath.exists():
            print('-----------------------> already exsists')
            # paths_to_delete = [path for path in OutputPath.glob("*")]
            # for path in paths_to_delete:
            #     path.unlink()

        else:
            OutputPath.mkdir(parents=True, exist_ok=True)
            
            # criando os novos dados
            D1, D2, D3 = volBIRADS.shape[0], volBIRADS.shape[1], volBIRADS.shape[2]
            

            n0, n3, n4 = 0, 0, 0

            for i in range(0, D1-side, stride):
                for j in range(0, D2-side, stride):
                    for k in range(0, D3-side, stride):
                        print(f'....{i}_{j}_{k}....', end = '\r')
                        new_vol = vol[:, i : i+side, j : j+side, k : k+side]
                        new_volBIRADS = volBIRADS[i : i+side, j : j+side, k : k+side]
                        
                        R0 = (new_volBIRADS == 0).sum()
                        R3 = (new_volBIRADS == 3).sum()
                        R4 = (new_volBIRADS == 4).sum()
                        R = R0 + R3 + R4


                        if R4 > limit: #pq 4?? Slá... só achei que seria bom não ser 1. E sla... 4 é um cubo, né?
                            label = 4
                            n4 += 1

                        elif R3 > limit:
                            label = 3
                            n3 += 1
                        else:
                            label = 0
                            n0 += 1

                        filename = str(OutputPath) + f'/{i}_{j}_{k}_{label}.nrrd'

                        if new_vol.shape == torch.Size([5,side,side,side]):
                            nrrd.write(filename, new_vol)




with Pool(6) as p:
    stats = p.map(create, exam_paths)
    print('fim')