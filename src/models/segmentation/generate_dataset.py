import os
import h5py
import numpy as np
import argparse



def normalize_image(image_array, min_cutoff = 0.001, max_cutoff = 0.001):
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
    sorted_array = np.sort(image_array.flatten())

    # Find file index and get values
    min_index = int(len(sorted_array) * min_cutoff)
    min_intensity = sorted_array[min_index]

    max_index = int(len(sorted_array) * max_cutoff) * -1
    max_intensity = sorted_array[max_index]

    # Normalize image and cutoff values
    image_array = (image_array - min_intensity) / \
        (max_intensity - min_intensity)
    image_array[image_array < 0.0] = 0.0
    image_array[image_array > 1.0] = 1.0

    return image_array


def save_exams(args):
    """
    Save exams from HDF5 file to individual .npy files after normalization and rotation.
    """
    # Load the HDF5 file
    f = h5py.File(args['hdf_file'], 'r')

    # Ensure the data directory exists
    exams = f['exams']
    assert isinstance(exams, h5py.Group), "Expected 'exams' to be a h5py Group."

    for ID in exams.keys():
        # Check if the file already exists
        if os.path.exists(f'{args["data_dir"]}/{ID}.npy'):
            continue

        # Load the image data
        img = exams[ID]
        assert isinstance(img, h5py.Dataset), "Expected 'img' to be a h5py Dataset."

        # Convert the first channel to numpy array and normalize
        img = img[0]
        img = normalize_image(img)
        img = np.rot90(img, 1)

        # Save the processed image to a .npy file
        np.save(os.path.join(args['data_dir'], f'{ID}.npy'), img)

    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate dataset for segmentation from HDF5 file.')
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/processed/pre_segmentation/',
        help='Directory to save the processed data.'
    )
    parser.add_argument(
        '--hdf-file',
        type=str,
        default='data/processed/hdf5/processed.hdf5'
    )
    args = parser.parse_args()
    args = vars(args)

    print(args)
    save_exams(args)
