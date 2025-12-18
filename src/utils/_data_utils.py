import os

# import sys
from glob import glob

import nrrd
import numpy as np

# from tqdm import tqdm

# def fix_segmentation_translation(
#    exam_data: np.ndarray, exam_header: OrderedDict, seg_data: np.ndarray, seg_header: OrderedDict
# ) -> Tuple[np.ndarray, OrderedDict]:
#    """Given that the headers are correct, return a new segment with correct dimensions and translation.
#    This fix does not check for out-of-bounds errors.
#    """
#
#    matrix_exam = exam_header["space directions"].T
#    exam_origin = exam_header["space origin"]
#    inv_matrix_exam = np.linalg.inv(matrix_exam)
#    seg_origin = seg_header["space origin"]
#
#    # computes origin and endpoint in voxel exam space
#    origin = np.rint((inv_matrix_exam @ (seg_origin - exam_origin))).astype(int)
#    endpoint = origin + seg_data.shape
#
#    new_seg_data = np.zeros(exam_data.shape)
#    new_seg_data[origin[0] : endpoint[0], origin[1] : endpoint[1], origin[2] : endpoint[2]] = seg_data
#
#    return new_seg_data, exam_header


class RawIterator:
    def __init__(self, n=None):
        self.n = n
        pass

    def __iter__(self):
        k = 0
        for subject_dir in sorted(glob("data/raw/*/*")):

            if self.n and k >= self.n:
                break

            if len(glob(f"{subject_dir}/A*.nrrd")) > 6:
                continue

            exams = []
            birad_3 = []
            birad_4 = []

            try:
                for label in ["A1", "A2", "A3", "A4", "A5"]:
                    data, header = nrrd.read(os.path.join(subject_dir, label + ".nrrd"))
                    exams.append([data, header])

                is_4d = [len(e[0].shape) > 3 for e in exams]

                if any(is_4d):
                    if sum(is_4d) > 1:
                        print(f"{subject_dir}: more than one 4d file!")
                        continue
                    else:
                        i_4d = np.argmax(is_4d)
                        exams[i_4d] = [exams[i_4d][0][0, :, :], exams[i_4d][1]]

                yield exams, subject_dir
                k += 1
            except FileNotFoundError as err:

                continue
                # print(f"{subject_dir}: {err}", file=sys.stderr)
