import argparse
import itertools
import json
import random

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.model import CachedDataset, SingleClassHdfDataset2d, SplitImages

parser = argparse.ArgumentParser()

parser.add_argument(
    "-is",
    "--image_size",
    type=int,
)

parser.add_argument(
    "-ss",
    "--split_size",
    type=int,
)

parser.add_argument("--only_birads_4", action="store_true")

parser.add_argument(
    "-d",
    "--dataset_path",
    type=str,
)

args = parser.parse_args()
model_hparams = vars(args)

subset_names = ["train"]

for subset_name in subset_names:

    sc_dataset = SingleClassHdfDataset2d(
        model_hparams["dataset_path"],
        subset_name,
        image_size=model_hparams["image_size"],
        only_birads_4=model_hparams["only_birads_4"],
    )

    if model_hparams["split_size"] > 0:
        dataset = SplitImages(sc_dataset, model_hparams["split_size"])
    else:
        dataset = sc_dataset

    hdf_path = "test.hdf"
    # hdf_file = h5py.File(hdf_path, "w")

    # hdf_file.create_dataset("length", data=len(dataset))

    # torch.save(len(dataset), f"test/length.ts")
    # exit(0)

    lst = list(range(len(dataset)))

    list_of_subindexes = []
    for s in range(0, len(lst), 256):
        subset_indexes = lst[s : s + 256]
        list_of_subindexes.append(subset_indexes)

    random.shuffle(list_of_subindexes)

    for k in range(0, len(list_of_subindexes), 32):
        subset_indexes = list(
            itertools.chain.from_iterable(list_of_subindexes[k : k + 32])
        )

        subset = torch.utils.data.Subset(dataset, subset_indexes)
        dataloader = DataLoader(subset, num_workers=6, shuffle=True)

        for i, data in enumerate(tqdm(dataloader, desc=subset_name)):

            # read the dataset
            inputs, labels = data
            info = dataset.get_info(i)

            # torch.save((inputs, labels, info), f"test/{i}.ts")

            # data_i = hdf_file.create_group(f"{i}")
            # data_i.create_dataset("inputs", data=inputs)
            # data_i.create_dataset("labels", data=labels)
            # data_i.create_dataset("info", data=json.dumps(info))
            ##info_group.create_dataset(f"{i}", data=info)
