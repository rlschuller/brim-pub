import argparse
import json
import os
import shutil
from glob import glob
from pathlib import Path

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.general import get_deterministic_name
from src.utils.model import SingleClassHdfDataset2d

parser = argparse.ArgumentParser()

parser.add_argument(
    "-es",
    "--early_stop",
    default=True,
    type=bool,
)

parser.add_argument(
    "-nw",
    "--num_workers",
    default=32,
    type=int,
)

parser.add_argument(
    "-l",
    "--loss",
    default="bce",
    type=str,
)

parser.add_argument(
    "-o",
    "--optimizer",
    default="adam",
    type=str,
)

parser.add_argument("-pw", "--pos_weight", default=None, type=float)

parser.add_argument("-lr", "--learning_rate", default=1e-2, type=float)

parser.add_argument("-me", "--max_epochs", default=100, type=int)

models = glob("src/models/*/model.py")
models = [m for m in models if m not in glob("src/models/_*/model.py")]
models = [m for m in models if "DLPT" not in m] # blacklist DLPT model, since it is a different format

subparsers = parser.add_subparsers(dest="model", metavar="MODEL", required=True)

for model_path in models:
    model_name = os.path.basename(os.path.dirname(model_path))
    add_subparser = getattr(__import__(f"src.models.{model_name}.model", fromlist=["add_subparser"]), "add_subparser")
    add_subparser(subparsers)


parser_dataset = parser.add_argument_group("dataset")
parser_dataset.add_argument(
    "-d",
    "--dataset_path",
    default="data/processed/datasets/1000-toy.json",
    type=str,
)

parser_dataset.add_argument(
    "--only_birads_4",
    default=True,
    type=bool,
)

parser_dataset.add_argument(
    "-s",
    "--image_size",
    default=512,
    type=int,
    help="side of the square images. if image_size==None, use original resolution",
)

parser_dataset.add_argument(
    "-da",
    "--data_augmentation",
    default=0,
    type=int,
    help="use data augmentation",
)

parser_dataset.add_argument(
    "-ni",
    "--normalize_inputs",
    default=True,
    type=bool,
)

parser_system = parser.add_argument_group("system")
parser_system.add_argument(
    "--device",
    default="cuda",
    type=str,
)
parser_system.add_argument("-bs", "--batch_size", default=8, type=int)
parser_system.add_argument(
    "-pf", "--personal_folder", action="store_true", help="use personal folder instead of shared folder"
)

args = parser.parse_args()

# process hparams
hparams = vars(args)
shared_hparam_keys = [action.dest for action in parser._actions if action.dest != "help"]

shared_hparams = {k: hparams[k] for k in shared_hparam_keys}
model_hparams = {k: hparams[k] for k in hparams if k not in shared_hparam_keys}

hparams = shared_hparams
hparams["model_hparams"] = model_hparams


# create folders and save hparams
run_name = get_deterministic_name(hparams)

base_folder = "personal" if hparams["personal_folder"] else "."
dataset_name = Path(args.dataset_path).stem

output_folder = f"{base_folder}/models/{hparams['model']}/{dataset_name}/{run_name}"

assert not os.path.exists(output_folder)

print(f"{output_folder=}")
checkpoints_folder = f"{output_folder}/checkpoints"
loss_folder = f"{output_folder}/loss"

os.makedirs(checkpoints_folder, exist_ok=True)
os.makedirs(loss_folder, exist_ok=True)


assert args.only_birads_4

n_channels = 5

hparams["computed_hparams"] = {}
hparams["computed_hparams"]["channels"] = n_channels


with open(output_folder + "/hparams.json", "w") as f:
    hparams["output_folder"] = output_folder
    json.dump(hparams, f, indent=4)


train_dataset = SingleClassHdfDataset2d(
    args.dataset_path,
    "train",
    image_size=args.image_size,
    only_birads_4=args.only_birads_4,
    balance=args.data_augmentation,
    normalize=args.normalize_inputs,
    binary_classification=True,
)

validation_dataset = SingleClassHdfDataset2d(
    args.dataset_path,
    "val",
    image_size=args.image_size,
    only_birads_4=args.only_birads_4,
    balance=args.data_augmentation,
    normalize=args.normalize_inputs,
    binary_classification=True,
)

train_dataloader = DataLoader(train_dataset, hparams["batch_size"], shuffle=True, num_workers=hparams["num_workers"])
validation_dataloader = DataLoader(
    validation_dataset, hparams["batch_size"], shuffle=False, num_workers=hparams["num_workers"]
)


# load Model from the corresponding file
Model = getattr(__import__(f"src.models.{hparams['model']}.model", fromlist=["Model"]), "Model")
device = hparams["device"]
print(f"{device=}")
model = Model(
    hparams=hparams,
    n_channels=n_channels,
)
model.to(device)

if hparams["optimizer"] == "adam":
    optimizer = optim.Adam(model.parameters(), lr=hparams["learning_rate"])
else:
    raise NotImplementedError

if hparams["pos_weight"]:
    pos_weight = torch.tensor([float(hparams["pos_weight"])], dtype=torch.float32).to(device)
else:
    pos_weight = None

if hparams["loss"] == "bce":
    loss_func = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)
else:
    raise NotImplementedError

for epoch in range(hparams["max_epochs"]):
    print(f"{epoch=}")
    acc_loss = 0
    model.train()
    for X, y in tqdm(train_dataloader):
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        pred = model(X)
        loss = loss_func(pred, y)

        loss.backward()
        optimizer.step()

        acc_loss += loss.item()
    train_loss = acc_loss / len(train_dataloader)
    print(f"{train_loss=}")
    with open(f"{loss_folder}/train-loss.jsonl", "a") as f:
        f.write(str(train_loss) + "\n")

    acc_loss = 0
    model.eval()

    correct_preds = 0
    with torch.no_grad():
        for X, y in tqdm(validation_dataloader):
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            loss = loss_func(pred, y)
            if hparams["loss"] == "bce":
                pred = torch.sigmoid(pred)
            correct_preds += torch.sum((pred > 0.5) == y).item()
            acc_loss += loss.item()

    validation_loss = acc_loss / len(validation_dataloader)
    validation_accuracy = float(correct_preds) / len(validation_dataset)
    print(f"{validation_loss=}")
    print(f"{validation_accuracy=}")
    print()

    with open(f"{loss_folder}/validation-loss.jsonl", "a") as f:
        f.write(str(validation_loss) + "\n")
    with open(f"{loss_folder}/validation-accuracy.jsonl", "a") as f:
        f.write(str(validation_accuracy) + "\n")

    torch.save(
        model.state_dict(),
        f"{checkpoints_folder}/epoch_{epoch}-val_loss_{validation_loss}.ckp",
    )
