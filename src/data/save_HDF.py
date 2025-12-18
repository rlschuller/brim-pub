import functools
import glob
import os
from matplotlib import pyplot as plt
import numpy as np

import h5py
import json
import numpy as np
from scipy.ndimage import zoom
import nrrd

import argparse



BREAST_MRI_DIR = './'

import sys
sys.path.append('src')
from utils.mri_class import MRI
from utils.util import enumerateWithEstimate


INPUT_PATH = BREAST_MRI_DIR + '/data/processed/nrrd_segs/full_data/*'
TUMORINFO_PATH = BREAST_MRI_DIR + '/src/data/TumorInfo.json'
HDF_PATH = BREAST_MRI_DIR + '/data/processed/hdf5/'

NEW_VOXEL_SIZE = (1.0, 1.0, 1.0)


@functools.lru_cache(1)
def getMRI(ID, dataset='raw', segmentation_path=None):
    return MRI(ID, dataset=HDF_PATH + dataset + ".hdf5", segmentation_path=segmentation_path)


def nrrdToHDF(filename: str, folderpath: str) -> None:
    exam_path_list = glob.glob(folderpath)

    filename = HDF_PATH + filename + '.hdf5'
    while os.path.exists(filename):
        filename = filename.replace('.hdf5', '_new.hdf5')

    f = h5py.File(filename, 'w')
    exams_group = f.create_group('exams')
    birads_group = f.create_group('birads')

    for _, exam_path in enumerateWithEstimate(exam_path_list, 'Saving nrrd as HDF5'):
        A1, metadata = nrrd.read(os.path.join(exam_path, "A1.nrrd"))
        A2, _ = nrrd.read(os.path.join(exam_path, "A2.nrrd"))
        A3, _ = nrrd.read(os.path.join(exam_path, "A3.nrrd"))
        A4, _ = nrrd.read(os.path.join(exam_path, "A4.nrrd"))
        A5, _ = nrrd.read(os.path.join(exam_path, "A5.nrrd"))
        birads, _ = nrrd.read(os.path.join(exam_path, "birads.nrrd"))
        exam = np.stack([A1, A2, A3, A4, A5], axis=0)

        ID = os.path.split(exam_path)[-1]
        exam_dataset = exams_group.create_dataset(
            ID,
            data=exam,
            dtype='f4',
            chunks=exam.shape,
            compression='gzip'
            )
        birads_group.create_dataset(
            ID,
            data=birads,
            dtype='f4',
            chunks=birads.shape,
            compression='gzip'
            )

        for key,value in metadata.items():
            exam_dataset.attrs[key] = value

    f.close()


def saveExamsHDF(
        filename: str,
        resize_bool: bool=False,
        fix_flip_bool: bool=False,
        normalize_bool: bool=False,
        dataset: str='raw'
        ) -> None:

    with h5py.File(HDF_PATH + dataset + '.hdf5', 'r') as f:
        exams = f['exams']
        assert isinstance(exams, h5py.Group)
        ID_set = set(exams.keys())

    filename = HDF_PATH + filename + '.hdf5'
    while os.path.exists(filename):
        filename = filename.replace('.hdf5', '_new.hdf5')

    f = h5py.File(filename, 'w')
    exams = f.create_group('exams')
    birads = f.create_group('birads')

    for _, ID in enumerateWithEstimate(ID_set, 'Saving exams'):
        mri = getMRI(ID, dataset=dataset)

        if resize_bool:
            mri.resize(NEW_VOXEL_SIZE)

        if fix_flip_bool:
            mri.fix_flip()

        if normalize_bool:
            mri.normalize()

        new_exam = exams.create_dataset(
            ID,
            data=mri.exam,
            dtype='f4',
            chunks= mri.exam.shape,
            compression='gzip'
            )
        new_birads = birads.create_dataset(
            ID,
            data= mri.birads,
            dtype='f4',
            chunks= mri.birads.shape,
            compression='gzip'
            )
        new_exam.attrs['space directions'] = np.array([[NEW_VOXEL_SIZE[0], 0, 0], [0, NEW_VOXEL_SIZE[1], 0], [0, 0, NEW_VOXEL_SIZE[2]]])
        new_birads.attrs['space directions'] = np.array([[NEW_VOXEL_SIZE[0], 0, 0], [0, NEW_VOXEL_SIZE[1], 0], [0, 0, NEW_VOXEL_SIZE[2]]])


    resize_str = f' Resized voxel to ({NEW_VOXEL_SIZE[0]}, {NEW_VOXEL_SIZE[1]}, {NEW_VOXEL_SIZE[2]}).' if resize_bool else ' No resize.'
    flip_str = ' Flipped MRI.' if fix_flip_bool else ' No flip.'
    exams.attrs['description'] = 'Breast MRI data. No breast segmentation applied.' + resize_str + flip_str
    birads.attrs['description'] = 'Birads segmentation.' + resize_str + flip_str

    f.close()


def saveCubesHDF(
    size: int=32,
    stride: int=8,
    filename: str='data/processed/hdf5/cubes.hdf5',
    dataset='data/processed/hdf5/processed.hdf5',
    segmentation_path='data/processed/hdf5/breast_segmentation.hdf5'
) -> None:
    # Get the list of IDs
    with h5py.File(dataset, 'r') as g:
        exams = g['exams']
        assert isinstance(exams, h5py.Group)
        ID_list = list(exams.keys())
        ID_list.sort(key=lambda x: int(x))
        # remove exam 45 because its too noisy
        # ID_list.remove('45')

    # Separate train and test
    test_ID_list = ID_list[::10]

    # TODO: ADD ID LIST HERE #
    ID_list = list(set(ID_list) - set(test_ID_list))

    # Create the file
    while os.path.exists(filename):
        filename = filename.replace('.hdf5', '_new.hdf5')
    f = h5py.File(filename, 'w')

    nontumors = f.create_group('nontumors',)
    tumors = f.create_group('tumors',)

    for _, ID in enumerateWithEstimate(ID_list, f'Saving exam as cubes of size {size} and stride {stride}'):
        mri = MRI(ID, dataset=dataset, segmentation_path=segmentation_path)
        X, Y = mri.get_split_in_cubes(size = size, stride = stride)

        new_nontumor_cubes = X[Y[:,0] == 1]
        new_tumor_cubes = X[Y[:,0] == 0]

        if new_nontumor_cubes.shape[0] > 0:
            nontumors.create_dataset(
                ID,
                data=new_nontumor_cubes,
                dtype='f4',
                chunks=(1, 5, size, size, size),
                compression='lzf'
                )
        else:
            print(f'No nontumor cubes for {ID}')

        if new_tumor_cubes.shape[0] > 0:
            tumors.create_dataset(
                ID,
                data=new_tumor_cubes,
                dtype='f4',
                chunks=(1, 5, size, size, size),
                compression='lzf'
                )
        else:
            print(f'No tumor cubes for {ID}')

    f.close()


def saveSlicesHDF(filename: str) -> None:
    MRI_list = glob.glob(INPUT_PATH)
    ID_set = {os.path.split(p)[-1] for p in MRI_list}

    filename = HDF_PATH + filename + '.hdf5'
    while os.path.exists(filename):
        filename = filename.replace('.hdf5', '_new.hdf5')
    f = h5py.File(filename, 'w')

    group = f.create_group('slices')
    for _, ID in enumerateWithEstimate(ID_set, 'Saving slices'):
        mri = getMRI(ID)
        data = mri.get_slices()
        group.create_dataset(
            ID,
            data=data,
            dtype='f4',
            chunks=(1,) + data.shape[1:],
            compression='lzf'
            )

    group.attrs['description'] = 'Slices from breast MRI. No breast segmentation applied. No resize. First 5 channels are the MRI data, the last channel is the birads segmentation'

    f.close()


def saveSlicesWithTumor(filename: str, size: tuple = (96, 96), pad: int = 5):
    with open(TUMORINFO_PATH, 'r') as f:
        tumors_info = json.load(f)

    filename = HDF_PATH + filename + f'{size[0]}_{size[1]}.hdf5'
    while os.path.exists(filename):
        filename = filename.replace('.hdf5', '_new.hdf5')

    f = h5py.File(filename, 'w')
    g = h5py.File(HDF_PATH + 'slices_without_mask.hdf5', 'r')

    slices = g['slices']
    assert isinstance(slices, h5py.Group)
    group = f.create_group('slices')

    for _, tumor in enumerateWithEstimate(tumors_info, 'Saving slices with tumor'):
        ID = tumor['ID']
        curr_slices = slices[ID]
        assert isinstance(curr_slices, h5py.Dataset)

        c0, c1, c2 = tumor['center']
        s0, s1 = max(0, c2 - int(tumor['diameter']//2) - pad), min(curr_slices.shape[0], c2 + int(tumor['diameter']//2) + pad)
        slices_with_tumor = curr_slices[s0:s1]

        w0, w1 = max(0, c0 - int(size[0]//2)), min(slices_with_tumor.shape[2], c0 + int(size[0]//2))
        h0, h1 = max(0, c1 - int(size[1]//2)), min(slices_with_tumor.shape[3], c1 + int(size[1]//2))
        slices_with_tumor = slices_with_tumor[:, :, w0:w1, h0:h1]

        data = np.zeros((slices_with_tumor.shape[0], 6, size[0], size[1]),)
        data[:, :, :slices_with_tumor.shape[2], :slices_with_tumor.shape[3]] = slices_with_tumor
        group.create_dataset(
            f'ID{ID}_center{c0}_{c1}_{c2}',
            data=data,
            dtype='f4',
            chunks=data.shape,
            compression='lzf'
            )

    group.attrs['description'] = f'Slices from breast MRI of size {size} centered in tumors. No breast segmentation applied. First 5 channels are the MRI data, the last channel is the birads segmentation. The center of the tumor is at the center of the slice. The slice is padded with zeros if the tumor is too close to the edge. There is a pad of {pad} slices without tumor above and below the tumor.'

    f.close()
    g.close()

def resize_segmentation(filename: str = 'breast_segmentation', dataset: str='raw'):
    datasetHDF = h5py.File(HDF_PATH + dataset + '.hdf5', 'r')

    filename = HDF_PATH + filename + f'_dset_{dataset}.hdf5'
    while os.path.exists(filename):
        filename = filename.replace('.hdf5', '_new.hdf5')

    new_segmentation = h5py.File(filename, 'w')
    ID_list = list(datasetHDF['exams'].keys())
    ID_list.sort(key=lambda x: int(x))

    for _, ID in enumerateWithEstimate(ID_list, 'Resizing segmentation'):
        mri = getMRI(ID, dataset=datasetHDF)
        old_seg = mri.get_seg()
        # Resize the segmentation
        ratio = [b/a for a,b in zip(mri.birads.shape, old_seg.shape)]
        new_seg = zoom(old_seg, ratio, order=0)

        # TODO - Check if the resizing is correct
        print(f'ID: {ID}, old shape: {old_seg.shape}, new shape: {new_seg.shape}')
        print(f'Ratio: {ratio}')

        new_seg_dataset = new_segmentation.create_dataset(
            ID,
            data= mri.birads,
            dtype='f4',
            chunks= mri.birads.shape,
            compression='gzip'
            )
        break


    new_segmentation.close()
    datasetHDF.close()

# This is necessary to avoid a bug in Qt
# os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")

def get_parser():
    parser = argparse.ArgumentParser(description='Save MRI data to HDF5 format.')
    subparsers = parser.add_subparsers(dest='command')
    subparsers.required = True
    parser_nrrd_to_hdf = subparsers.add_parser(
        'nrrd-to-hdf',
        help='Convert NRRD files to HDF5 format.'
    )
    parser_nrrd_to_hdf.add_argument(
        '--filename',
        default='raw',
        type=str,
        help='Name of the output HDF5 file.'
    )
    parser_nrrd_to_hdf.add_argument(
        '--folderpath',
        default='data/processed/nrrd/*/*',
        type=str,
        help='Glob pattern for NRRD files to convert.'
    )

    parser_save_exams = subparsers.add_parser(
        'save-exams',
        help='Save MRI exams to HDF5 format.'
    )
    parser_save_exams.add_argument(
        '--filename',
        type=str,
        default='processed',
        help='Name of the output HDF5 file.'
    )
    parser_save_exams.add_argument(
        '--resize',
        default=True,
        action='store_true',
        help='Resize the MRI exams.'
    )
    parser_save_exams.add_argument(
        '--fix_flip',
        action='store_true',
        default=True,
        help='Fix the flip in MRI exams.'
    )
    parser_save_exams.add_argument(
        '--normalize',
        action='store_true',
        default=True,
        help='Normalize the MRI exams.'
    )
    parser_save_exams.add_argument(
        '--dataset',
        default='raw',
        type=str,
        help='Dataset with raw MRI data to save. Default is "raw".'
    )

    parser_save_cubes = subparsers.add_parser(
        'save-cubes',
        help='Save MRI data as cubes of a given size and stride.'
    )
    parser_save_cubes.add_argument(
        '--filename',
        type=str,
        default='cubes',
        help='Name of the output HDF5 file.'
    )
    parser_save_cubes.add_argument(
        '--size',
        type=int,
        default=32,
        help='Size of the cubes to save.'
    )
    parser_save_cubes.add_argument(
        '--stride',
        type=int,
        default=8,
        help='Stride for the cubes.'
    )
    parser_save_cubes.add_argument(
        '--dataset',
        type=str,
        default='data/processed/hdf5/processed.hdf5',
        help='Path to the dataset HDF5 file.'
    )
    parser_save_cubes.add_argument(
        '--segmentation-path',
        type=str,
        default='data/processed/hdf5/breast_segmentation.hdf5',
        help='Path to the breast segmentation HDF5 file.'
    )

    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    args = vars(args)

    if args['command'] == 'nrrd-to-hdf':
        nrrdToHDF(args['filename'], args['folderpath'])

    elif args['command'] == 'save-exams':
        saveExamsHDF(
            args['filename'],
            resize_bool=args['resize'],
            fix_flip_bool=args['fix_flip'],
            normalize_bool=args['normalize'],
            dataset=args['dataset']
        )

    elif args['command'] == 'save-cubes':
        saveCubesHDF(
            size=args['size'],
            stride=args['stride'],
            filename=args['filename'],
            dataset=args['dataset'],
            segmentation_path=args['segmentation_path']
        )

    else:
        parser.print_help()

    # saveSlicesHDF('slices_without_mask')
    # saveSlicesWithTumor('slices_with_tumor', size=(96, 96), pad=5)
    # nrrdToHDF('complete_plus_new_data', BREAST_MRI_DIR + '/data/processed/nrrd/*/*')
    # saveExamsHDF('resized', resize_bool=True, fix_flip_bool=True)
    # saveCubesHDF('cubes', size=16, stride=8)
    # resize_segmentation()

    pass

if __name__ == '__main__':
    main()
