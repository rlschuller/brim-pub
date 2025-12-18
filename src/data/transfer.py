import numpy as np
import argparse
import pathlib
import nrrd
import tqdm
import torch
import matplotlib.pyplot as plt
import csv
import pandas as pd
import numpy as np
from scipy.ndimage import zoom


input_path = './data/processed/nrrd/'


#load and split exam function
def cut_exam(exam_path):
    print(f'{exam_path.name}')
    A1, meta1 = nrrd.read(f'{str(exam_path)}/A1.nrrd')
    A2, meta2 = nrrd.read(f'{str(exam_path)}/A2.nrrd')
    A3, meta3 = nrrd.read(f'{str(exam_path)}/A3.nrrd')
    A4, meta4 = nrrd.read(f'{str(exam_path)}/A4.nrrd')
    A5, meta5 = nrrd.read(f'{str(exam_path)}/A5.nrrd')
    Y, metaY = nrrd.read(f'{str(exam_path)}/birads.nrrd')

    
    pathlib.Path(f'data/move/{exam_path.name}').mkdir(parents = True)
    
    nrrd.write(f'data/move/{exam_path.name}/A1.nrrd', A1, meta1)
    nrrd.write(f'data/move/{exam_path.name}/A2.nrrd', A2, meta2)
    nrrd.write(f'data/move/{exam_path.name}/A3.nrrd', A3, meta3)
    nrrd.write(f'data/move/{exam_path.name}/A4.nrrd', A4, meta4)
    nrrd.write(f'data/move/{exam_path.name}/A5.nrrd', A5, meta5)
    nrrd.write(f'data/move/{exam_path.name}/birads.nrrd', Y, metaY)



folder = pathlib.Path(input_path)
exams = [exam for exam in folder.glob('*/*')]

for exam in exams:
    cut_exam(exam)
