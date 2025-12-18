import argparse
import json
import os
import re
from glob import glob

import medpy.io
import numpy as np
import scipy
from scipy.ndimage import label
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument("-p", "--path_to_nrrd_folder", type=str)

args = parser.parse_args()

subject_folders = sorted(glob(f"{args.path_to_nrrd_folder}/*/*"))

OUTPUT_FOLDER = re.sub(r"/*$", "", args.path_to_nrrd_folder) + "-metrics"
JSONL_PATH = f"{OUTPUT_FOLDER}/all.jsonl"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

if os.path.exists(JSONL_PATH):
    os.unlink(JSONL_PATH)

print(OUTPUT_FOLDER)

assert len(subject_folders) >= 1

kernel = np.ones((2, 2, 2), dtype=np.int64)


for folder in tqdm(subject_folders):
    lot = os.path.basename(os.path.dirname(folder))
    number = os.path.basename(folder)
    subject_id = f"{lot}/{number}"

    exam_paths = sorted(glob(f"{folder}/A*.nrrd"))
    assert len(exam_paths) == 5

    exams = [np.array(medpy.io.load(e)[0]) for e in exam_paths]
    birads, header = medpy.io.load(f"{folder}/birads.nrrd")
    birads = np.array(birads)

    components, n_components = label(birads)
    metrics = {}
    metrics["id"] = subject_id
    metrics["shape"] = birads.shape
    metrics["voxel_spacing"] = header.get_voxel_spacing()
    metrics["voxel_volume"] = np.prod(header.get_voxel_spacing()) / 10**3
    metrics["voxels"] = int(np.prod(birads.shape))
    metrics["voxels_with_birads_3"] = int(np.sum(birads == 3))
    metrics["voxels_with_birads_4"] = int(np.sum(birads == 4))

    metrics["volume_with_birads_3"] = (
        metrics["voxels_with_birads_3"] * metrics["voxel_volume"]
    )
    metrics["volume_with_birads_4"] = (
        metrics["voxels_with_birads_4"] * metrics["voxel_volume"]
    )

    metrics["number_of_connected_components"] = n_components

    dilated_birads = scipy.signal.convolve(birads, kernel) != 0

    dilated_components, dilated_n_components = label(dilated_birads)
    metrics[
        "number_of_connected_components-after_dilation_of_1_voxel"
    ] = dilated_n_components

    dilated_birads = scipy.signal.convolve(dilated_birads, kernel) != 0
    dilated_components, dilated_n_components = label(dilated_birads)
    metrics[
        "number_of_connected_components-after_dilation_of_2_voxels"
    ] = dilated_n_components

    # print(json.dumps(metrics))

    metrics[
        "input_max"
    ] = max([float(np.max(e)) for e in  exams])
    metrics[
        "input_min"
    ] = min([float(np.min(e)) for e in  exams])
    metrics[
        "input_range"
    ] = metrics["input_max"] - metrics["input_min"]
    metrics[
        "input_mean"
    ] = float(np.mean([float(np.mean(e)) for e in  exams]))


    metrics[
        "vector_input_max"
    ] = [float(np.max(e)) for e in  exams]
    metrics[
        "vector_input_min"
    ] = [float(np.min(e)) for e in  exams]
    metrics[
        "vector_input_range"
    ] = [float(np.max(e))-float(np.min(e)) for e in  exams]
    metrics[
        "vector_input_mean"
    ] = [float(np.mean(e)) for e in  exams]


    with open(JSONL_PATH, "a") as f:
        f.write(json.dumps(metrics) + "\n")
