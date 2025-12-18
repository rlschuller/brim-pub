import os
import sys
from glob import glob

import nrrd
import numpy as np

blacklisted_folders = {}
exam_labels = ["A1", "A2", "A3", "A4", "A5", "A6"]

for subject_dir in sorted(glob("data/raw/*/*")):
    if subject_dir in blacklisted_folders:
        continue

    exams = []

    if len(glob(f"{subject_dir}/A*.nrrd")) > 6:
        print(f"{subject_dir}: more than 6 files of type 'A*.nrrd'.", file=sys.stderr)

    try:
        for label in exam_labels:
            data, header = nrrd.read(os.path.join(subject_dir, label + ".nrrd"))
            exams.append([data, header])

        is_4d = [len(e[0].shape) > 3 for e in exams]

        if any(is_4d):
            if sum(is_4d) > 1:
                print(f"{subject_dir}: more than one 4d file!", file=sys.stderr)
            else:
                i_4d = np.argmax(is_4d)

                exam_4d = exams[i_4d]

                correspondence_vector = ["na"] * exam_4d[0].shape[0]

                for i in range(len(exams)):
                    if i != i_4d:
                        for j in range(exam_4d[0].shape[0]):
                            if np.array_equal(exam_4d[0][j, :, :, :], exams[i][0]):
                                correspondence_vector[j] = exam_labels[i]
                print(
                    (
                        f"{subject_dir}: 4d_file='{exam_labels[i_4d]}.nrrd"
                        f"',  correspondence_vector={correspondence_vector}"
                    ),
                    file=sys.stderr,
                )

    except FileNotFoundError as err:
        print(f"{subject_dir}: {err}", file=sys.stderr)
