import os
import h5py
import numpy as np
from scipy import ndimage as ndi
from skimage.measure import label,regionprops
from scipy.ndimage import binary_dilation
import cv2
import glob
from skimage.filters import threshold_multiotsu

import argparse

CWD = os.getcwd()
BREAST_MRI_DIR = CWD[:CWD.find('breast_mri')+10]

import sys
sys.path.append(BREAST_MRI_DIR + '/src')
from utils.mri_class import MRI
from utils.util import enumerateWithEstimate

MASKS_DIR = BREAST_MRI_DIR + '/src/models/segmentation/masks'


def keep_top(slc, n=3, min_area_ratio=None):
    slc_labeled = label(slc)
    new_slc = np.zeros_like(slc_labeled)
    rps = regionprops(slc_labeled)
    if min_area_ratio is not None:
        rps = [r for r in rps if r.area > min_area_ratio * slc.size]
    areas = [r.area for r in rps]
    idxs = np.argsort(areas)[::-1]
    if n is None:
        n = len(areas)
    for i in idxs[:n]:
        new_slc[tuple(rps[i].coords.T)] = i+1
    return 1 * (new_slc > 0)


def fill_holes(img):
    img = ndi.binary_fill_holes(img)    
    img = np.vectorize(ndi.binary_fill_holes, signature='(m,n)->(m,n)')(img)
    img = np.transpose(
        np.vectorize(ndi.binary_fill_holes, signature='(m,n)->(m,n)')(np.transpose(img, (2, 1, 0))),
          (2, 1, 0))
    return img

def get_largest_connected_component(img, n=1):
    img = np.transpose(
        np.vectorize(lambda x: keep_top(x, n=n), signature='(m,n)->(m,n)')(np.transpose(img, (2, 1, 0))),
          (2, 1, 0))
    return img


def getNotAir(ID, exam_path):
    img = np.load(f'{exam_path}/{ID}.npy')
    threshold = threshold_multiotsu(img, classes=3)[0]

    smoothness = 10

    mask = cv2.threshold(img, threshold, 1, cv2.THRESH_BINARY)[1]
    mask = np.transpose(np.vectorize(lambda x: keep_top(x, n=None, min_area_ratio=0.001), signature='(m,n)->(m,n)')(np.transpose(mask, (2, 0, 1))), (1, 2, 0))
    mask1 = ndi.binary_erosion(mask, iterations=5)
    mask = ndi.binary_propagation(mask1, mask=mask)
    mask = ndi.binary_dilation(mask, iterations=smoothness)
    mask = fill_holes(mask)
    mask = ndi.binary_erosion(mask, iterations=smoothness)
    mask = get_largest_connected_component(mask, n=1)

    return mask


def processMask(ID, exam_path, masks_dir):
    NotAir = getNotAir(ID, exam_path)
    mask = np.load(f'{masks_dir}/{ID}.npy')
    mask = (mask > 0.5)
    mask = get_largest_connected_component(mask, n=1)
    mask = (mask > 0) & (NotAir > 0)
    mask = binary_dilation(mask, iterations=10, mask=NotAir)

    mean = np.mean(mask, axis=(1,2))
    arg = np.argmax(mean)
    mask[:arg+ mask.shape[0]//20,:,:] = NotAir[:arg+ mask.shape[0]//20,:,:]
    mask[arg + mask.shape[0]//10:,:,:] = 0

    mask = np.rot90(mask, -1)
    return mask


def get_args():
    parser = argparse.ArgumentParser(description='Save Breast Segmentation Masks')
    parser.add_argument(
        # '-o', '--output-path', metavar='D', type=str, default='data/processed/hdf5/breast_segmentation.hdf5',
        '-o', '--output-path', metavar='D', type=str, default='data/processed/hdf5/teste.hdf5',
        help='Directory to save processed masks HDF file', dest='output_path'
    )
    parser.add_argument(
        '-m', '--masks-dir', metavar='M', type=str, default="data/processed/pre_segmentation_masks",
        help='Directory of input masks', dest='masks_dir'
    )
    parser.add_argument(
        '-e', '--exam-dir', metavar='E', type=str, default='data/processed/pre_segmentation_data',
        help='Directory of exams', dest='exam_dir'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    f = h5py.File(args.output_path, 'w')

    ID_list = glob.glob(args.masks_dir + '/*.npy')
    ID_list = [ID.split('/')[-1].split('.')[0] for ID in ID_list]
    ID_list.sort(key=lambda x: int(x))
    for _,ID in enumerateWithEstimate(ID_list, 'Saving Breast Segmentation Masks'):
        processed_mask = processMask(ID, args.exam_dir, args.masks_dir)
        dataset = f.create_dataset(
            ID,
            data=processed_mask,
            chunks=processed_mask.shape,
            compression='gzip'
            )
    f.close()
