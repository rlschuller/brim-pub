import medpy.io
from pathlib import Path
import json
import numpy as np
import h5py
#from tqdm import tqdm
#from torch.utils.data import DataLoader

import torch
import torch.utils.data

class NrrdDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_path,
        subset='train',
        normalization_quantile=None,
        breast_mask=False,
    ):


        dataset = json.load(Path(dataset_path).open("r"))
        self.dataset = dataset

        self.ids = dataset[subset]
        self.folder = dataset["md5_path"].replace(".md5", "")
        self.normalization_quantile = normalization_quantile

        self.breast_mask = breast_mask

        with Path("data/processed/flip_IDs.json").open("r") as f:
            self.breast_mask_flip_ids = set(json.load(f))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):

        exams = []
        for filename in ["A1", "A2", "A3", "A4", "A5"]:
            data = medpy.io.load(f"{self.folder}/{self.ids[i]}/{filename}.nrrd")[0]
            exams.append(data)

        birads = medpy.io.load(f"{self.folder}/{self.ids[i]}/birads.nrrd")[0]



        X = torch.tensor(np.stack(exams, dtype=np.float32))
        y = torch.tensor(np.stack([birads == 3, birads==4], dtype=np.float32))

        metadata = {}

        metadata["id"] = self.ids[i]
        metadata["i"] = i
        metadata["folder"] = str(Path(f"{self.folder}/{self.ids[i]}"))

        if self.normalization_quantile:
            X = X / torch.quantile(
                X[:,::10,::10,::10],
                self.normalization_quantile,
                interpolation='linear'
            )


        if self.breast_mask:
            masks = h5py.File("data/processed/hdf5/breast_segmentation.hdf5")
            mask_id = self.ids[i].split("/")[-1]

            mask = np.array(masks[mask_id])


            if mask_id in self.breast_mask_flip_ids:
                mask = np.flip(mask, 1).copy()

            mask = torch.tensor(mask,dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            mask = torch.nn.functional.interpolate(mask,size=X.shape[1:]).squeeze(0).squeeze(0)

            X = {"exam": X, "mask":mask} #torch.cat((X, mask), dim=0)

        return X, y, metadata

#dataset = json.load(Path("data/processed/datasets/crimson_duck-844.json").open("r"))
#
#it = NrrdDataset(dataset)
#dl = DataLoader(it, 1, num_workers=1)
#
#for X,y,_ in tqdm(dl):
#    pass
