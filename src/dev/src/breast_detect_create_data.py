import nrrd
import numpy as np
import pathlib
import tqdm

folder = './data/processed/nrrd'
paths = pathlib.Path(folder)
exam_paths = [path for path in paths.glob("*/*")]


def create(exam_path):
    if (pathlib.Path(str(exam_path)+'/A1.nrrd').exists()):
        # loading data
        print(str(exam_path))
        
        vol, _ = nrrd.read(str(exam_path)+'/A1.nrrd')

        seg, _ = nrrd.read(str(exam_path)[:19]+'_breast'+str(exam_path)[19:]+'/Segmentation.seg.nrrd')
        


        OutputPath = pathlib.Path(f'./data/processed/breast_detection'+ str(exam_path)[19:])

            
        # criando os novos dados
        D1, D2, D3 = vol.shape[0], vol.shape[1], vol.shape[2]
        for i in range(1, D1-1):

            X = vol[i-1:i+2, :, :]
            Y = seg[i]

            to_write = np.array([X[0],X[1],X[2],Y])
            name = str(OutputPath) + f'_{i}.nrrd'
            
            nrrd.write(name, to_write)
    
            
               
for exame in tqdm.tqdm(exam_paths):
    create(exame)
