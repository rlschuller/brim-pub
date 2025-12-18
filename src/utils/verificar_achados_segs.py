import argparse
import numpy as np
import nrrd
import torch
import pathlib
import tqdm

parser = argparse.ArgumentParser()

parser.add_argument(
    '-ex',
    default = './data/processed/nrrd/',
    type = str
)

parser.add_argument(
    '-seg',
    default = './data/processed/nrrd_segs/CNN_075_2d/',
    type = str
)

parser.add_argument(
    '-ex_l',
    default = 2,
    type = int
)

parser.add_argument(
    '-seg_l',
    default = 1,
    type = int
)

args = parser.parse_args()

exams_folder = args.ex
segs_folder = args.seg
exam_depth = args.ex_l * '*/'
segs_depth = args.seg_l* '*/'


folder = pathlib.Path(exams_folder)
exams = [exam for exam in folder.glob(f'{exam_depth}birads.nrrd')]
exams.sort()

folder = pathlib.Path(segs_folder)
segs = [seg for seg in folder.glob(f'{segs_depth}pred.nrrd')]
segs.sort()

names = [s.parent.name for s in segs]

wrongs = 0
for i in range(len(segs)):
    exam, _ = nrrd.read(exams[i])
    exam = 1*(exam != 0)
    
    seg, _ = nrrd.read(segs_folder + exams[i].parent.name + '/pred.nrrd')

    final = seg - exam
    
    values, counts = np.unique(final, return_counts = True)

    if -1 in values:
        wrongs += 1
        print(exams[i], segs_folder + exams[i].parent.name + '/pred.nrrd')
        print(wrongs, len(segs), wrongs/len(segs))
        
    


    
    