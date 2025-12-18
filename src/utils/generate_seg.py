import argparse
import numpy as np
import nrrd
import torch
import pathlib
import time
import tqdm

parser = argparse.ArgumentParser()

parser.add_argument(
    '-input',
    default = './data/processed/nrrd/',
    type = str
)

parser.add_argument(
    '-output',
    default = './data/processed/nrrd_segs/',
    type = str
)

parser.add_argument(
    '-model',
    default = './models/breast_seg/20_model.pth',
    type = str
)


parser.add_argument(
    '-mode',
    default = 'folder',
    type = str
)

parser.add_argument(
    '-lower_limit',
    default = 0.0,
    type = float
)

parser.add_argument(
    '-nick',
    default = 'full_data',
    type = str
)

args = parser.parse_args()

input_path =  args.input
model_path = args.model
mode = args.mode
lower_limit = args.lower_limit
nick = args.nick

output_path = args.output + f'{nick}'

print('Create folder')
P = pathlib.Path(output_path)
P.mkdir(parents=True, exist_ok=True)


print('Load names')
folder = pathlib.Path(input_path)
exams = [exam for exam in folder.glob('*/*/A1.nrrd')][::-1]


examsA1 = [exam for exam in folder.glob('*/*/A1.nrrd')][::-1]
examsA2 = [exam for exam in folder.glob('*/*/A2.nrrd')][::-1]
examsA3 = [exam for exam in folder.glob('*/*/A3.nrrd')][::-1]
examsA4 = [exam for exam in folder.glob('*/*/A4.nrrd')][::-1]
examsA5 = [exam for exam in folder.glob('*/*/A5.nrrd')][::-1]
examsB = [exam for exam in folder.glob('*/*/birads.nrrd')][::-1]

for name in tqdm.tqdm(examsA1):

    exam, meta = nrrd.read(name)
    
    P = pathlib.Path(output_path + f'/{name.parent.name}/')
    P.mkdir(parents=True, exist_ok=True)
    nrrd.write(output_path + f'/{name.parent.name}/A1.nrrd', exam, meta)


for name in tqdm.tqdm(examsA2):

    exam, meta = nrrd.read(name)
    
    P = pathlib.Path(output_path + f'/{name.parent.name}/')
    nrrd.write(output_path + f'/{name.parent.name}/A2.nrrd', exam, meta)


for name in tqdm.tqdm(examsA3):

    exam, meta = nrrd.read(name)
    
    P = pathlib.Path(output_path + f'/{name.parent.name}/')
    nrrd.write(output_path + f'/{name.parent.name}/A3.nrrd', exam, meta)


for name in tqdm.tqdm(examsA4):

    exam, meta = nrrd.read(name)
    
    P = pathlib.Path(output_path + f'/{name.parent.name}/')
    nrrd.write(output_path + f'/{name.parent.name}/A4.nrrd', exam, meta)


for name in tqdm.tqdm(examsA5):

    exam, meta = nrrd.read(name)
    
    P = pathlib.Path(output_path + f'/{name.parent.name}/')
    nrrd.write(output_path + f'/{name.parent.name}/A5.nrrd', exam, meta)


for name in tqdm.tqdm(examsB):

    exam, meta = nrrd.read(name)
    
    P = pathlib.Path(output_path + f'/{name.parent.name}/')
    nrrd.write(output_path + f'/{name.parent.name}/birads.nrrd', exam, meta)



print('Load model')
import src.models.breast_seg.model as M
model = M.UNET()
model.load_state_dict(torch.load(model_path))
model.eval()
model.double()

strides = [16, 12, 8, 4] 

print('Create segs')
for name in tqdm.tqdm(exams):

    exam, meta = nrrd.read(name)
    
    # P = pathlib.Path(output_path + f'/{name.parent.name}/')
    # P.mkdir(parents=True, exist_ok=True)
    # nrrd.write(output_path + f'/{name.parent.name}/A1.nrrd', exam, meta)

    for k in range(0, exam.shape[2]):

        x = exam[:, :, k]
        x = x.astype(float)
        # x = torch.tensor(x)
        # x = torch.reshape(x, [1,1,exam.shape[0],exam.shape[1]])

        exam[:, :, k] = model(torch.reshape(torch.tensor(x), [1,1,exam.shape[0],exam.shape[1]])).detach().numpy()

    exam = (exam > 0)*1
    nrrd.write(output_path + f'/{name.parent.name}/base_pred.nrrd', exam, meta)

    for stride in strides:
        seg = torch.zeros(size = exam.shape)
        for i in range(0, seg.shape[0]-stride+1, 1):
            for j in range(0, seg.shape[1]-stride+1, 1):
                for k in range(0, seg.shape[2]- stride+1, 1):
                    
                    if lower_limit == 0:
                        if 1 in exam[i:i+stride, j:j+stride, k:k+stride]:
                            seg[i:i+stride, j:j+stride, k:k+stride] = torch.ones(size = seg[i:i+stride, j:j+stride, k:k+stride].shape)

                    else:
                        x = sum(sum(sum(exam[i:i+stride, j:j+stride, k:k+stride])))
                        if x >= (stride**3) * (lower_limit):
                            seg[i:i+stride, j:j+stride, k:k+stride] = torch.ones(size = seg[i:i+stride, j:j+stride, k:k+stride].shape)

        nrrd.write(output_path + f'/{name.parent.name}/s{stride}_pred.nrrd', seg.numpy(), meta)
    
