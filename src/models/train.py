import argparse
import json
import os
from glob import glob
import random
import torchinfo
import time
from math import floor
from pathlib import Path

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.general import get_deterministic_name
from src.utils.model import EarlyStop
from src.utils.data import NrrdDataset

parser = argparse.ArgumentParser()

parser.add_argument(
    "-nw",
    "--num_workers",
    default=24,
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

parser.add_argument(
    "-es",
    "--early_stop",
    default=7,
    type=int,
)

parser.add_argument("-pw", "--pos_weight", default=None, type=float)

parser.add_argument("-lr", "--learning_rate", default=1e-4, type=float)

parser.add_argument("-me", "--max_epochs", default=300, type=int)

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
    default="data/processed/datasets/crimson_duck-839.json",
    type=str,
)

#parser_dataset.add_argument(
#    "--only_birads_4",
#    default=True,
#    type=bool,
#)
#
#parser_dataset.add_argument(
#    "-s",
#    "--image_size",
#    default=512,
#    type=int,
#    help="side of the square images. if image_size==None, use original resolution",
#)
#
#parser_dataset.add_argument(
#    "-da",
#    "--data_augmentation",
#    default=0,
#    type=int,
#    help="use data augmentation",
#)

parser_dataset.add_argument(
    "-nq",
    "--normalization_quantile",
    type=float,
    default=.95,
)

parser_dataset.add_argument(
    "--breast_mask",
    type=int,
    default=0,
)

parser_system = parser.add_argument_group("system")
parser_system.add_argument(
    "--device",
    default="cuda",
    type=str,
)
parser_system.add_argument("-bs", "--batch_size", default=100000, type=int)
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
dataset_name = Path(hparams["dataset_path"]).stem

output_folder = f"{base_folder}/models/{hparams['model']}/{dataset_name}/{run_name}"

assert not os.path.exists(output_folder), f"'{output_folder}' exists"

print(f"{output_folder=}")
checkpoints_folder = f"{output_folder}/checkpoints"
loss_folder = f"{output_folder}/loss"

os.makedirs(checkpoints_folder, exist_ok=True)
os.makedirs(loss_folder, exist_ok=True)


#assert args.only_birads_4

n_channels = 5

hparams["computed_hparams"] = {}
hparams["computed_hparams"]["channels"] = n_channels


with open(output_folder + "/hparams.json", "w") as f:
    hparams["output_folder"] = output_folder
    json.dump(hparams, f, indent=4)

print(f"{hparams['normalization_quantile']=}")

train_dataset = NrrdDataset(
    hparams["dataset_path"],
    "train",
    normalization_quantile=hparams["normalization_quantile"],
    breast_mask=hparams["breast_mask"],
)

validation_dataset = NrrdDataset(
    hparams["dataset_path"],
    "val",
    normalization_quantile=hparams["normalization_quantile"],
    breast_mask=hparams["breast_mask"],
)

train_dataloader = DataLoader(train_dataset, 1, shuffle=True, num_workers=hparams["num_workers"])
validation_dataloader = DataLoader(
    validation_dataset, 1, shuffle=False, num_workers=hparams["num_workers"]
)


# load Model from the corresponding file
Model = getattr(__import__(f"src.models.{hparams['model']}.model", fromlist=["Model"]), "Model")
device = hparams["device"]
print(f"{device=}")
model = Model(
    hparams=hparams,
    in_channels=n_channels,
)
model.to(device)

torchinfo.summary(model)

if hparams["optimizer"] == "adam":
    optimizer = optim.Adam(model.parameters(), lr=hparams["learning_rate"])
elif hparams["optimizer"] == "manual":
    assert hparams["breast_mask"]
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

early_stop = EarlyStop(int(hparams["early_stop"]))

for epoch in range(hparams["max_epochs"]):
    print(f"{epoch=}")
    model.train()

    acc_loss = 0
    n = 0
    correct_preds = 0
    total_preds = 0
    positive_preds = 0
    positive_y = 0
    true_positive = 0
    len_batch = 0

    for X, y, metadata in tqdm(train_dataloader, ncols=60):
        X = X#.to(device)
        y = y#.to(device)


        exam = X["exam"]
        mask = X["mask"]


        exam = exam.permute(0,2,3,4,1)
        y = y.permute(0,2,3,4,1)
        input = exam[mask>.5,:]
        output = y[mask>.5,:]

        p = torch.randperm(input.shape[0])

        input = input[p, :]
        output = output[p, :]

        batch_size = hparams["batch_size"]
        in_batches = torch.split(input, batch_size)
        out_batches = torch.split(output, batch_size)
        len_batch += len(in_batches)

        for _X, _y in zip(in_batches[:-1], out_batches[:-1]):
            _X = _X.to(device)
            _y = _y.to(device)
            _pred = model(_X)

            optimizer.zero_grad()
            loss = loss_func(_pred, _y)
            loss.backward()
            optimizer.step()

            if hparams["loss"] == "bce":
                _pred = torch.sigmoid(_pred)

            positive_y += torch.sum(_y).item()
            true_positive += torch.sum(_y * (_pred > 0.5)).item()

            correct_preds += torch.sum((_pred > 0.5) == _y).item()
            positive_preds += torch.sum(_pred > 0.5).item()


            total_preds += torch.sum(_y > -1)

            acc_loss += loss.item()
            n += 1



    recall = true_positive / positive_y
    accuracy = float(correct_preds) / float(total_preds)
    precision = float(true_positive) / (float(positive_preds) +0.1)
    print(f"{positive_y=}")
    print(f"{positive_y/total_preds * hparams['pos_weight']=}")
    print(f"{positive_y/len_batch=}")
    print(f"{positive_preds=}")
    print(f"{recall=}")
    print(f"{precision=}")
    print(f"{accuracy=}")
    train_loss = acc_loss / n
    print(f"{train_loss=}")
    with open(f"{loss_folder}/train-loss.jsonl", "a") as f:
        f.write(str(train_loss) + "\n")
    with open(f"{loss_folder}/train-accuracy.jsonl", "a") as f:
        f.write(str(accuracy) + "\n")
    with open(f"{loss_folder}/train-recall.jsonl", "a") as f:
        f.write(str(recall) + "\n")
    with open(f"{loss_folder}/train-precision.jsonl", "a") as f:
        f.write(str(precision) + "\n")

    print()
    print("validation")
    model.eval()

    acc_loss = 0
    n = 0
    correct_preds = 0
    total_preds = 0
    positive_preds = 0
    positive_y = 0
    positive_y_mask = 0
    true_positive = 0

    with torch.no_grad():
        for X, y, metadata in tqdm(validation_dataloader, ncols=60):
            X = X#.to(device)
            y = y#.to(device)

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

                #print(f"{_y.shape=}")
                #print(f"{_pred_mask.shape=}")
                _pred = model(_X)

                if torch.sum(_pred_mask) > 0:
                    loss = loss_func(_pred[_pred_mask], _y[_pred_mask])
                    acc_loss += loss.item()
                    n += 1

                if hparams["loss"] == "bce":
                    _pred = torch.sigmoid(_pred)

                _pred = _pred * _pred_mask

                correct_preds += torch.sum((_pred > 0.5) == _y).item()
                total_preds += torch.sum(_y > -1)

                positive_y += torch.sum(_y).item()
                positive_y_mask += torch.sum(_y * _pred_mask).item()
                true_positive += torch.sum(_y * (_pred > 0.5)).item()
                positive_preds += torch.sum(_pred > 0.5).item()



    validation_loss = acc_loss / n
    validation_accuracy = float(correct_preds) / float(total_preds)
    validation_recall = true_positive / positive_y
    validation_precision = float(true_positive) / (float(positive_preds) +0.1)
    print(f"{positive_y=}")
    print(f"{positive_y_mask=}")
    print(f"{positive_preds=}")
    print(f"{validation_recall=}")
    print(f"{validation_precision=}")
    print(f"{validation_accuracy=}")
    print(f"{validation_loss=}")
    print()


    with open(f"{loss_folder}/validation-loss.jsonl", "a") as f:
        f.write(str(validation_loss) + "\n")
    with open(f"{loss_folder}/validation-recall.jsonl", "a") as f:
        f.write(str(validation_recall) + "\n")
    with open(f"{loss_folder}/validation-precision.jsonl", "a") as f:
        f.write(str(validation_precision) + "\n")
    with open(f"{loss_folder}/validation-accuracy.jsonl", "a") as f:
        f.write(str(validation_accuracy) + "\n")

    torch.save(
        model.state_dict(),
        f"{checkpoints_folder}/epoch_{epoch}-val_loss_{validation_loss}.ckp",
    )

    #early_stop(validation_loss)

