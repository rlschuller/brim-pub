# flake8: noqa
import os

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from src.utils.model import *

OUTPUT_FOLDER = "test/utils/hdf_dataset"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

dataset_2d = SingleClassHdfDataset2d(
    "data/processed/datasets/toy.json",
    "train",
    image_size=256,
    only_birads_4=True,
    balance=True,
)

generator = torch.Generator()
generator.manual_seed(0)
dataset_2d.shuffle(generator)

print(len(dataset_2d))

dataloader = DataLoader(
    dataset_2d,
    batch_size=1,
    num_workers=8,
    shuffle=False,
)

i = 0
for X, y in tqdm(dataloader):
    s = torch.sum(y)
    print(X.shape)
    print(y.shape)
    save_image(X[0][0], f"{OUTPUT_FOLDER}/{i:010}-X.png")
    save_image(y[0][0], f"{OUTPUT_FOLDER}/{i:010}-y.png")
    i += 1
