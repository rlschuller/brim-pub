import argparse
import json
import os
import shutil
from glob import glob
from pathlib import Path
import numpy as np

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.general import get_deterministic_name
from src.utils.data import NrrdDataset

parser = argparse.ArgumentParser()

parser.add_argument(
    "-p",
    "--path",
    type=str,
)
parser.add_argument(
    "-nw",
    "--num_workers",
    default=16,
    type=int,
)

parser.add_argument("--device", default="cuda", type=str)

args = parser.parse_args()
device = args.device

with open(f"{args.path}/hparams.json", "r") as f:
    hparams = json.load(f)

base_folder = "personal" if hparams["personal_folder"] else "."
predictions_folder = f"{hparams['output_folder']}/predictions"
checkpoints_folder = f"{hparams['output_folder']}/checkpoints"
print(f"{predictions_folder=}")
os.makedirs(predictions_folder, exist_ok=True)

print(json.dumps(hparams, indent=4))
channels = hparams["computed_hparams"]["channels"]

# load Model from the corresponding file
Model = getattr(__import__(f"src.models.{hparams['model']}.model", fromlist=["Model"]), "Model")
device = hparams["device"]
print(f"{device=}")
model = Model(
    hparams=hparams,
    in_channels=channels,
)


if hparams["pos_weight"]:
    pos_weight = torch.tensor([float(hparams["pos_weight"])], dtype=torch.float32).to(device)
else:
    pos_weight = None

if hparams["loss"] == "bce":
    loss_func = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)
else:
    raise NotImplementedError


min_loss = 9999999
min_loss_path = ""
for checkpoint_path in glob(f"{checkpoints_folder}/*.ckp"):
    loss = float(checkpoint_path.split("_")[-1].replace(".ckp", ""))
    if loss < min_loss:
        min_loss = loss
        min_loss_path = checkpoint_path
print(min_loss)
model.load_state_dict(torch.load(min_loss_path))

model.to(device)
model.eval()
with torch.no_grad():

    for subset in ["test", "val"]:

        dataset = NrrdDataset(
            hparams["dataset_path"],
            subset,
            normalization_quantile=hparams["normalization_quantile"],
            breast_mask=hparams["breast_mask"],
        )

        dataloader = DataLoader(dataset, 1, shuffle=False, num_workers=args.num_workers)


        acc_loss = 0
        n = 0

        correct_preds = 0
        total_preds = 0
        positive_y = 0
        true_positive = 0


        total = len(dataloader)
        for X, y, metadata in tqdm(dataloader, desc=subset, ncols=60):
            pass
            X = X#.to(device)#.unsqueeze(0)

            y = y#.to(device)#.unsqueeze(0)


            exam = X["exam"]
            mask = X["mask"].unsqueeze(0)


            exam = exam.permute(0,2,3,4,1)
            y = y.permute(0,2,3,4,1)

            mask = mask.permute(0,2,3,4,1)
            mask = mask.repeat(1,1,1,1,2) > .5

            pred = torch.zeros(y.shape)


            for i in range(exam.shape[-2]):

                _X = exam[:,:,:,i,:].to(device)
                _y = y[:,:,:,i,:].to(device)
                _pred_mask = mask[:,:,:,i,:].to(device)

                _pred = model(_X)
                loss = loss_func(_pred[_pred_mask], _y[_pred_mask])
                if hparams["loss"] == "bce":
                    _pred = torch.sigmoid(_pred)


                _pred = _pred * _pred_mask

                pred[:,:,:,i,:] = _pred.to('cpu')



                acc_loss += loss.item()
                n += 1


            pred = pred.permute(0,4,1,2,3)

            assert len(metadata["id"]) == 1

            npz_folder = Path(f"{predictions_folder}/{subset}/{'/'.join((metadata['id'][0].split('/')[:-1]))}")
            npz_folder.mkdir(exist_ok=True, parents=True)
            npz_path = npz_folder / (metadata["id"][0].split("/")[-1] + ".npz")

            pred_out = pred.squeeze(0).detach().numpy()
            np.savez_compressed(npz_path, pred_out)

        loss = acc_loss / n

        print(f"{loss=}")
        print()

        #f.close()
