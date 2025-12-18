import argparse
import numpy as np
import nrrd
import torch
import pathlib

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
    default = './models/breast_seg/best_model_temp.pth',
    type = str
)

parser.add_argument(
    '-model_dim',
    default = '3d',
    type = str
)

parser.add_argument(
    '-mode',
    default = 'folder',
    type = str
)

parser.add_argument(
    '-lower_limit',
    default = 0.5,
    type = float
)

parser.add_argument(
    '-nick',
    default = 'testes',
    type = str
)

args = parser.parse_args()

input_path =  args.input
model_path = args.model
mode = args.mode
model_dim = args.model_dim
lower_limit = args.lower_limit
nick = args.nick

output_path = args.output + f'{nick}_{model_dim}'

P = pathlib.Path(output_path)
P.mkdir(parents=True, exist_ok=True)



folder = pathlib.Path(input_path)
exams = [exam for exam in folder.glob('*/*/A1.nrrd')][::-1]




if model_dim == '3d':

    import src.models.breast_detection_CNN_3d.model as M
    model = M.CNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.double()


    stride = 32

    for name in exams:
        exam, meta = nrrd.read(name)
        seg = torch.zeros(size = exam.shape)
        seg_2 = torch.zeros(size = exam.shape)

        P = pathlib.Path(output_path + f'/{name.parent.name}/')
        P.mkdir(parents=True, exist_ok=True)

        for i in range(0, exam.shape[0]-stride+1, stride):
            for j in range(0, exam.shape[1]-stride+1, stride):
                for k in range(0, exam.shape[2]-stride+1, stride):
                    
                    x = exam[i:i+stride, j:j+stride, k:k+stride]
                    x = x.astype(float)
                    x = torch.tensor(x)
                    x = torch.reshape(x, [1,1,32,32,32])

                    y = model(x)
                    if y[0][0] > y[0][1]:
                        seg[i:i+stride, j:j+stride, k:k+stride] = torch.ones(size = seg[i:i+stride, j:j+stride, k:k+stride].shape)
        
        nrrd.write(output_path + f'/{name.parent.name}/pred0.nrrd', seg.numpy(), meta)

        for i in range(0, exam.shape[0]-stride+1, int(stride/4)):
            for j in range(0, exam.shape[1]-stride+1, int(stride/4)):
                for k in range(0, exam.shape[2]- stride+1, int(stride/4)):

                    x = sum(sum(sum(seg[i:i+stride, j:j+stride, k:k+stride])))
                    if x >= (stride**3) * (lower_limit):
                        seg_2[i:i+stride, j:j+stride, k:k+stride] = torch.ones(size = seg_2[i:i+stride, j:j+stride, k:k+stride].shape)

        nrrd.write(output_path + f'/{name.parent.name}/pred.nrrd', seg_2.numpy(), meta)
        nrrd.write(output_path + f'/{name.parent.name}/exam.nrrd', exam, meta)
    



elif model_dim == '2d':
    import src.models.breast_detection_CNN_2d.model as M
    model = M.CNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.double()

    stride = 32

    for name in exams:
        exam, meta = nrrd.read(name)
        seg = torch.zeros(size = exam.shape)
        seg_2 = torch.zeros(size = exam.shape)

        P = pathlib.Path(output_path + f'/{name.parent.name}/')
        P.mkdir(parents=True, exist_ok=True)

        for i in range(0, exam.shape[0]-stride+1, stride):
            for j in range(0, exam.shape[1]-stride+1, stride):
                for k in range(0, exam.shape[2]):

                    x = exam[i:i+stride, j:j+stride, k]
                    x = x.astype(float)
                    x = torch.tensor(x)
                    x = torch.reshape(x, [1,1,stride,stride])

                    y = model(x)
                    if y[0][0] > y[0][1]:
                        seg[i:i+stride, j:j+stride, k] += 1
        
        nrrd.write(output_path + f'/{name.parent.name}/pred0.nrrd', seg.numpy(), meta)

        for i in range(0, exam.shape[0]-stride+1, int(stride/4)):
            for j in range(0, exam.shape[1]-stride+1, int(stride/4)):
                for k in range(0, exam.shape[2]- stride+1, int(stride/4)):

                    x = sum(sum(sum(seg[i:i+stride, j:j+stride, k:k+stride])))
                    if x >= (stride**3) * (lower_limit):
                        seg_2[i:i+stride, j:j+stride, k:k+stride] = torch.ones(size = seg_2[i:i+stride, j:j+stride, k:k+stride].shape)

        nrrd.write(output_path + f'/{name.parent.name}/pred.nrrd', seg_2.numpy(), meta)
        nrrd.write(output_path + f'/{name.parent.name}/exam.nrrd', exam, meta)