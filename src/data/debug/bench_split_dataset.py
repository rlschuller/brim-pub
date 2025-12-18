import argparse

from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.model import (
    CachedDataset,
    CachedFolderDataset,
    SingleClassHdfDataset2d,
    SplitImages,
)

parser = argparse.ArgumentParser()

parser.add_argument(
    "path",
    type=str,
)

parser.add_argument(
    "--type",
    type=str,
)


args = parser.parse_args()


# hdf_file = h5py.File(args.path, "r")
if args.type == "hdf":
    dataset = CachedDataset(args.path)
elif args.type == "folder":
    dataset = CachedFolderDataset(args.path)
else:
    raise NotImplementedError(f"args.type == {args.type}")


dataloader = DataLoader(dataset, num_workers=6)
subset_name = "test"
for i, data in enumerate(tqdm(dataloader, desc=subset_name)):
    inputs, labels = data
