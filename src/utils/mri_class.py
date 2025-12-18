import os
from typing import Tuple
import numpy as np
import h5py
from scipy.ndimage import zoom
from skimage.measure import label,regionprops
from itertools import product 
import joblib
import cv2


BREAST_MRI_DIR = './'

DATA_PATH = BREAST_MRI_DIR + '/data/processed/nrrd_segs/full_data/'
HDF_PATH = BREAST_MRI_DIR + '/data/processed/hdf5/'

# Cubify function from https://stackoverflow.com/questions/42297115/numpy-split-cube-into-cubes
def cubify(arr: np.ndarray, newshape: tuple):
    oldshape = np.array(arr.shape)
    repeats = (oldshape / newshape).astype(int)
    tmpshape = np.column_stack([repeats, newshape]).ravel()
    order = np.arange(len(tmpshape))
    order = np.concatenate([order[::2], order[1::2]])
    # newshape must divide oldshape evenly or else ValueError will be raised
    return arr.reshape(tmpshape).transpose(order).reshape(-1, *newshape)

def uncubify(arr, oldshape):
    N, newshape = arr.shape[0], arr.shape[1:]
    oldshape = np.array(oldshape)    
    repeats = (oldshape / newshape).astype(int)
    tmpshape = np.concatenate([repeats, newshape])
    order = np.arange(len(tmpshape)).reshape(2, -1).ravel(order='F')
    return arr.reshape(tmpshape).transpose(order).reshape(oldshape)



class MRI:
    """
    Class for handling MRI data.
    """
    def __init__(self, 
                 ID: str,
                 dataset: str | h5py.File = 'data/processed/hdf5/processed.hdf5',
                 segmentation_path = None,
            ) -> None:
        """
        Initialize MRI object.

        Parameters:
            ID (str): The ID of the MRI data.
            dataset (str): The name of the dataset where the MRI data is stored. Default is 'complete_dataset_flipped_resized'. Must be a key in the HDF_PATH dictionary. 
        """
        self.ID = ID
        if isinstance(dataset, h5py.File):
            f = dataset

        elif isinstance(dataset, str):
            f = h5py.File(dataset, "r")

        exams = f["exams"]
        assert isinstance(exams, h5py.Group)
        curr_exam = exams[ID]
        assert isinstance(curr_exam, h5py.Dataset)
        self.exam = curr_exam

        space_directions = curr_exam.attrs["space directions"]
        assert isinstance(space_directions, np.ndarray)
        self.voxel_size = (space_directions[0,0], space_directions[1,1], space_directions[2,2])

        birads = f["birads"]
        assert isinstance(birads, h5py.Group)
        curr_birads = birads[ID]
        assert isinstance(curr_birads, h5py.Dataset)
        self.birads = curr_birads

        if isinstance(dataset, str):
            self.exam = self.exam[()]
            self.birads = self.birads[()]
            f.close()

        self.segmentation_path = segmentation_path

    def get_seg(self) -> np.ndarray:
        if not hasattr(self, 'seg'):
            with h5py.File(self.segmentation_path, "r") as segFile:
                curr_seg = segFile[self.ID]
                assert isinstance(curr_seg, h5py.Dataset)
                self.seg = curr_seg[()]
        return self.seg
        
        

    def __repr__(self) -> str:
        return f"MRI(ID={self.ID})"


    def get_split_in_cubes(
            self, 
            size: int = 32, 
            stride: int | None = 16, 
            min_birads_volume: int = 35,
            min_cube_volume_ratio: float = 0.5
            ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Splits the MRI data into cubes of specified size.

        Parameters:
            size (int): The size of the cubes. Default is 32.
            stride (int): The stride of the cubes. Default is 16. Must divide size.
            min_birads_volume (int): The minimum volume (number of voxels) of a tumor. Cubes with a tumor volume less than this value will be discarded. Default is 35.
            min_cube_volume_ratio (float): The minimum ratio of cube volume intersecting breast tissue to cube volume. Cubes with a ratio less than this value will be discarded. Default is 0.5.

        Returns:
            Tuple containing:
                X (np.ndarray): The input data in cubes.
                Y (np.ndarray): The labels.
        """
        if stride is None:
            stride = size

        if size % stride != 0:
            raise ValueError("Stride must divide size.")
        
        X = np.empty((0, 5, size, size, size))
        Y = np.empty((0, 2))

        if self.segmentation_path is None:
            data_array = np.concatenate((self.exam, self.birads[None]))
        else:
            data_array = np.concatenate((self.exam, self.birads[None], self.get_seg()[None]))
        
        for border in product(range(0, size, stride), repeat=3):
            data_array_clipped = data_array[:, border[0]:, border[1]:, border[2]:]
            _, x, y, z = data_array_clipped.shape
            data_array_clipped = data_array_clipped[:, :(x//size)*size, :(y//size)*size, :(z//size)*size]

            split_chunks = cubify(data_array_clipped, (7, size, size, size))

            birads_volume = np.sum(split_chunks[:,5], axis=(1, 2, 3))
            cube_volume_ratio = np.mean(split_chunks[:,6], axis=(1, 2, 3))
            mask = (cube_volume_ratio > min_cube_volume_ratio) & ((birads_volume == 0) | (birads_volume > min_birads_volume))
            masked_chunks = split_chunks[mask, :-1]
            masked_birads = masked_chunks[:, -1]

            X = np.concatenate((X, masked_chunks[:,:-1]), axis=0)
            birads_mask = 1 * (np.max(masked_birads, axis=(1, 2, 3)) > 0)
            Y = np.concatenate((Y, np.column_stack((1 - birads_mask, birads_mask)).astype(float)), axis=0)

        return X, Y

    def get_tumors(self, min_area: float = 5.0):
        """
        Returns a list of tumor objects, each representing a tumor in the MRI data.

        Parameters:
            min_area (float): The minimum area of a tumor. Default is 5.0.

        Returns:
            RegionProperties object.
        """
        birads_labeled = label(self.birads)
        rps = regionprops(birads_labeled, intensity_image=self.exam[0])
        rps = [r for r in rps if r.area > min_area]
        for r in rps:
            bbox = r.bbox
            r.birads_type = np.max(self.birads[bbox[0]:bbox[3], bbox[1]:bbox[4], bbox[2]:bbox[5]])

        return rps
        
    
    def get_cube(self, center: tuple, size: int | float) -> np.ndarray:
        """
        Returns a cube of MRI data centered on the specified coordinates.

        Parameters:
            center (tuple): The coordinates of the center of the cube.
            size (int | float): The size of the cube.

        Returns:
            Numpy array of the cube.
        """
        c1, c2, c3 = center
        r = int(size/2)
        cube = self.exam[:, c1-r:c1+r, c2-r:c2+r, c3-r:c3+r].astype(np.float16)
        
        return cube
    
    def get_box(self, bbox: np.ndarray | list, birads_mask_bool: bool=True) -> np.ndarray:
        """
        Returns a box of MRI data specified by the bounding box.

        Parameters:
            bbox (np.ndarray): The bounding box of the box.

        Returns:
            Numpy array of the box.
        """
        box = self.exam[:, bbox[0]:bbox[3], bbox[1]:bbox[4], bbox[2]:bbox[5]].astype(np.float16)
        birads_mask = self.birads[bbox[0]:bbox[3], bbox[1]:bbox[4], bbox[2]:bbox[5]]

        return np.concatenate((box, birads_mask[None])) if birads_mask_bool else box
    
    def get_breast_mask(self):
        """
        Returns the breast mask of the MRI data as a RegionProperties object.
        """
        rps = regionprops(label(self.get_seg()), intensity_image=self.exam[0])
        return rps[np.argmax([r.area for r in rps])]
    
    def get_slices(self):
        return np.transpose(np.concatenate((self.exam, self.birads[None])), (3,0,1,2))
    
    def fix_flip(self) -> bool:
        min_cutoff = 0.001
        max_cutoff = 0.001
        
        sorted_array = np.sort(self.exam.flatten())

        min_index = int(len(sorted_array) * min_cutoff)
        min_intensity = sorted_array[min_index]

        max_index = int(len(sorted_array) * max_cutoff) * -1
        max_intensity = sorted_array[max_index]

        # Normalize image and cutoff values
        exam_normalized = (self.exam - min_intensity) / \
            (max_intensity - min_intensity)
        exam_normalized[exam_normalized < 0.0] = 0.0
        exam_normalized[exam_normalized > 1.0] = 1.0

        slc = np.max(exam_normalized, axis=(0, 3))
        slc_resized = cv2.resize(slc, (32, 32))
        model = joblib.load('./models/flip/flip.pkl')
        flip = model.predict(slc_resized.flatten().reshape(1, -1))[0]
        if flip:
            self.exam = np.flip(self.exam, axis=2)
            self.birads = np.flip(self.birads, axis=1)
            if self.segmentation_path is not None:
                self.seg = np.flip(self.get_seg(), axis=1)
        return flip


    def resize(self, newVoxelSize: Tuple[float, float, float]):
        currentVoxelSize = self.voxel_size
        resizeFactor = (currentVoxelSize[0] / newVoxelSize[0], currentVoxelSize[1] / newVoxelSize[1], currentVoxelSize[2] / newVoxelSize[2])
        exam_shape = self.exam.shape
        new_shape = (int(exam_shape[1] * resizeFactor[0]), int(exam_shape[2] * resizeFactor[1]), int(exam_shape[3] * resizeFactor[2]))
        resized_exam = np.zeros((5,) + new_shape, dtype=np.float32)
        resized_birads = np.zeros(new_shape, dtype=np.float32)
        resized_seg = np.zeros(new_shape, dtype=np.float32)

        for i in range(exam_shape[0]):
            resized_exam[i] = zoom(self.exam[i], resizeFactor, order=3)[:new_shape[0], :new_shape[1], :new_shape[2]]
        resized_birads = zoom(self.birads, resizeFactor, order=0)[:new_shape[0], :new_shape[1], :new_shape[2]] # order=0 is nearest neighbor interpolation
        if self.segmentation_path is not None:
            resized_seg = zoom(self.get_seg(), resizeFactor, order=0)[:new_shape[0], :new_shape[1], :new_shape[2]]

        self.exam = resized_exam
        self.birads = resized_birads
        if self.segmentation_path is not None:
            self.seg = resized_seg

        self.voxel_size = newVoxelSize

    # code from https://github.com/mazurowski-lab/3D-Breast-FGT-and-Blood-Vessel-Segmentation
    def normalize(self, min_cutoff = 0.001, max_cutoff = 0.001):
        """
        Normalize the intensity of an image array by cutting off min and max values 
        to a certain percentile and set all values above/below that percentile to 
        the new max/min. 

        Parameters
        ----------
        image_array: np.array
            3D numpy array constructed from dicom files
        min_cutoff: float
            Minimum percentile of image to keep. (0.1% = 0.001)
        max_cutoff: float
            Maximum percentile of image to keep. (0.1% = 0.001)

        Returns
        -------
        np.array
            Normalized image

        """
        
        # Sort image values
        sorted_array = np.sort(self.exam.flatten())

        # Find %ile index and get values
        min_index = int(len(sorted_array) * min_cutoff)
        min_intensity = sorted_array[min_index]

        max_index = int(len(sorted_array) * max_cutoff) * -1
        max_intensity = sorted_array[max_index]

        # Normalize image and cutoff values
        self.exam = (self.exam - min_intensity) / \
            (max_intensity - min_intensity)
        self.exam[self.exam < 0.0] = 0.0
        self.exam[self.exam > 1.0] = 1.0




# This is necessary to avoid a bug in Qt
#os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")
import matplotlib.pyplot as plt

# Example usage
def main():
    not_obvious = [
        ("132", 0),
        ("143", 0),
        ("146", 0),
        ("165", 0),
        ("175", 0),
        ("196", 3),
        ("325", 0),
        ("133", 0),
        ("390", 0),
        ("481", 2),
        ("509", 0),
        ("538", 0),
        ("549", 0),
        ("612", 0),
         ]
    
    obvious = [
        ("05", 0),
        ("43", 0),
        ("61", 0),
        ("67", 0),
        ("93", 0),
        ("279", 0),
    ]

    not_tumor = [
        
    ]


         
    for ID, n in obvious:
        break
        print(ID)
        folder = './data/obvious_tumors/'
        mri = MRI(ID)
        # mri.fix_flip()
        # X, Y = mri.get_split_in_cubes(size = 32, stride = 8)
        # mask = mri.get_breast_mask()
        # slices = mri.get_slices()
        # tumors = mri.get_tumors()
        # cube = mri.get_cube((100, 100, 100), 32)
        tumors = mri.get_tumors()
        c = tumors[n].centroid
        plt.figure()
        plt.imshow(mri.exam[3, :, :, int(c[2])], cmap='gray')
        cmap = plt.cm.jet
        cmap.set_under(alpha=0)

        plt.axis('off')
        plt.tight_layout()

        os.makedirs(folder, exist_ok=True)
        
        plt.savefig(f'{folder}{ID}_{n}.png',bbox_inches='tight', dpi=500)
        plt.imshow(mri.birads[:, :, int(c[2])], cmap=cmap, alpha=0.7, vmin=0.01)
        plt.savefig(f'{folder}{ID}_{n}_segmented.png', bbox_inches='tight', dpi=500)
        plt.close()

    
    
    



if __name__ == "__main__":
    main()
