import argparse
import json
import os
import shutil
from glob import glob

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.general import get_deterministic_name
from src.utils.model import SingleClassHdfDataset2d

parser = argparse.ArgumentParser()

parser.add_argument(
    "-f",
    "--run_folder",
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

with open(f"{args.run_folder}/hparams.json", "r") as f:
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
    n_channels=channels,
)

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

        f = open(f"{predictions_folder}/{subset}.jsonl", "w")
        #dataset = TiffDataset(
        #    hparams["dataset_path"], subset=subset, image_size=hparams["image_size"], channels=channels
        #)
        dataset = SingleClassHdfDataset2d(
            hparams["dataset_path"],
            subset,
            image_size=hparams["image_size"],
            only_birads_4=hparams["only_birads_4"],
            normalize_inputs=hparams["normalize_inputs"],
            binary_classification=True,
        )
        dataloader = DataLoader(dataset, 1, shuffle=True, num_workers=args.num_workers)

        correct = 0
        correct_preds = 0
        total = len(dataloader)
        for i, (X, y) in enumerate(tqdm(dataset, desc=subset)):
            X = X.to(device).unsqueeze(0)

            y = y.to(device).unsqueeze(0)
            pred = model(X)
            info = dataset.get_info(i)

            if hparams["loss"] == "bce":
                pred = torch.sigmoid(pred)
            correct_preds += (pred > 0.5) == y
            y = y.to("cpu")
            pred = pred.to("cpu")

            y = float(y[0][0])
            pred = float(pred[0][0])
            correct += y == (pred > 0.5)

            f.write(json.dumps({"y": y, "pred": pred, "info": info}) + "\n")
        accuracy = correct / total
        print(f"{accuracy=}")
        accuracy = correct_preds / total
        print(f"{accuracy=}")

        f.close()
