import os
import pathlib

import nrrd
import numpy as np
from numpy.linalg import inv as inverse
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

from src.utils.data_utils import RawIterator

out_folder = pathlib.Path("tmp")
out_folder.mkdir(exist_ok=True, parents=True)

for exams, subject in RawIterator():

    shapes = {e[0].shape for e in exams}
    assert len(shapes) == 1

    data = np.array([e[0] for e in exams])

    x = np.array([[a * a, a, 1] for a in range(5)])

    m = inverse(x.transpose() @ x) @ x.transpose()

    out = np.zeros(data.shape[1:])
    print(out_folder / f"{os.path.basename(subject)}.nrrd")

    for i in tqdm(range(data.shape[1])):
        for j in range(data.shape[2]):
            for k in range(data.shape[3]):
                y = data[:5, i, j, k]
                c = m @ y
                out[i, j, k] = max(-c[0], 0) * (max(y) - min(y))

    nrrd.write(str(out_folder / f"{os.path.basename(subject)}.nrrd"), out)
