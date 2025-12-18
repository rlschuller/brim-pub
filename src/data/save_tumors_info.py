import os
import json
import glob
import functools
import h5py
import numpy as np


BREAST_MRI_DIR = './'
HDF_PATH = BREAST_MRI_DIR + '/data/processed/hdf5/'

import sys
sys.path.append(BREAST_MRI_DIR + '/src')
from utils.mri_class import MRI
from utils.util import enumerateWithEstimate

INPUT_PATH = BREAST_MRI_DIR + '/data/processed/nrrd_segs/full_data/*'
TUMORINFO_PATH = BREAST_MRI_DIR + '/src/data/'


def saveCandidateInfo(filename):
    with h5py.File(HDF_PATH + 'processed.hdf5', 'r') as f:
        exams = f['exams']
        assert isinstance(exams, h5py.Group)
        ID_list = list(exams.keys())
    ID_list.sort(key=lambda x: int(x))

    filename = TUMORINFO_PATH + filename + '.json'

    while os.path.exists(filename):
        filename = filename.replace('.json', '_new.json')

    f = open(filename, 'w')
    f.write('[')
    for _, ID in enumerateWithEstimate(ID_list, 'Processing MRI'):
        mri = MRI(ID)
        tumors_list = mri.get_tumors()
        for tumor in tumors_list:
            center_list = list(tumor.centroid)
            dict = {
                'birads_type': int(tumor.birads_type), 
                'volume': tumor.area, 
                'ID': ID, 
                'center': [int(x) for x in center_list],
                'diameter': tumor.feret_diameter_max,
                'bbox': tumor.bbox
                }
            json.dump(dict, f, indent=2)
            f.write(',')
    
    f.seek(f.tell() - 1, os.SEEK_SET)

    f.write(']')
    f.close()

@functools.lru_cache(1)
def getMRI(ID, dataset='processed'):
    return MRI(ID, dataset=dataset)


def save_tumors(filename):
    # Get the ID list
    with h5py.File(HDF_PATH + 'processed.hdf5', 'r') as f:
        exams = f['exams']
        assert isinstance(exams, h5py.Group)
        ID_list = list(exams.keys())
        ID_list.sort(key=lambda x: int(x))

    # Create the file
    filename = HDF_PATH + filename + '.hdf5'
    f = h5py.File(filename, 'w')
    birads3 = f.create_group('birads3')
    birads4 = f.create_group('birads4')

    dataset = h5py.File(HDF_PATH + 'processed.hdf5', 'r')

    for _, ID in enumerateWithEstimate(ID_list, 'Saving tumors'):
        mri = getMRI(ID, dataset=dataset)
        tumors = mri.get_tumors()
        for t in tumors:
            bbox = t.bbox
            tumor_box = np.zeros((6, bbox[3]-bbox[0], bbox[4]-bbox[1], bbox[5]-bbox[2]))
            tumor_box[:5] = mri.exam[:, bbox[0]:bbox[3], bbox[1]:bbox[4], bbox[2]:bbox[5]]
            tumor_box[5] = t.image

            if int(t.birads_type) == 3:
                birads3.create_dataset(
                    f'ID{ID}_center_{t.centroid[0]:.0f}_{t.centroid[1]:.0f}_{t.centroid[2]:.0f}',
                    data=tumor_box,
                    chunks=tumor_box.shape,
                    compression='gzip'
                )

            elif int(t.birads_type) == 4:
                birads4.create_dataset(
                    f'ID{ID}_center_{t.centroid[0]:.0f}_{t.centroid[1]:.0f}_{t.centroid[2]:.0f}',
                    data=mri.get_box(t.bbox),
                    chunks=tumor_box.shape,
                    compression='gzip'
                )
            
            else:
                print(f'ID{ID} has a tumor with birads type {t.birads_type}')
        

    f.attrs['description'] = 'Tumor data for birads 3 and 4. Each dataset is the tumor cropped from the MRI data by the bounding box. The first five channels are the MRI data, the last channel is the mask.'

    dataset.close()
    f.close()

def saveFlipBool(filename):
    with h5py.File(HDF_PATH + 'processed.hdf5', 'r') as f:
        exams = f['exams']
        assert isinstance(exams, h5py.Group)
        ID_list = list(exams.keys())
    ID_list.sort(key=lambda x: int(x))

    flip_list = []
    for _, ID in enumerateWithEstimate(ID_list, 'Processing MRI'):
        mri = MRI(ID, dataset='raw')
        if mri.fix_flip():
            flip_list.append(ID)


    filename = TUMORINFO_PATH + filename + '.json'
    while os.path.exists(filename):
        filename = filename.replace('.json', '_new.json')

    with open(filename, 'w') as f:
        json.dump(flip_list, f, indent=2)


# read the json file
def read_json(filename):
    with open(filename) as f:
        data = json.load(f)
    return data


if __name__ == '__main__':
    # saveCandidateInfo('TumorInfoComplete')
    # save_tumors('tumors_and_masks')
    saveFlipBool('flip_bool')
    pass 