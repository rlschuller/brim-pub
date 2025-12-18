import argparse
import json
import os
import random
import subprocess
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.unet.model import Model
from src.utils.log import get_git_revision, get_modified_files
from src.utils.model import TiffDataset

parser = argparse.ArgumentParser()

parser.add_argument(
    "--dataset_path",
    "-d",
    default="data/processed/datasets/8089-harlequin_lamprey.json",
    help="path of the dataset to use",
)

parser.add_argument(
    "-j",
    "--json",
    type=str,
    help="bypass the argparser and load an hparams.json file",
)

parser.add_argument(
    "-bs",
    "--batch_size",
    type=int,
    default=1,
)

parser.add_argument(
    "-nw",
    "--num_workers",
    type=int,
    default=1,
)

parser.add_argument(
    "--pos_weight",
    default=1,
    type=float,
    help="pos_weight parameter for the bce loss",
)

parser.add_argument(
    "--learning_rate",
    default=1e-3,
    type=float,
)

parser.add_argument(
    "--image_size",
    default=None,
    type=int,
)


parser.add_argument(
    "--max_epochs",
    default=20,
    type=int,
)

parser.add_argument(
    "-n",
    "--name",
    default=None,
    type=str,
    help="name of the output folder",
)


parser.add_argument(
    "-dga",
    "--disable_git_asserts",
    action="store_true",
    help="allow mixing different git revisions in the same training session",
)

args, extra_args_list = parser.parse_known_args()

if args.json:
    with open(args.json, "r") as f:
        input_hparams = json.load(f)
        if args.disable_git_asserts:
            input_hparams["disable_git_asserts"] = True

        args = SimpleNamespace(**input_hparams)

dataset_name = os.path.basename(args.dataset_path).split(".")[0]

date = subprocess.check_output(["date", "-Is"]).decode("utf-8")[:-1]
hostname = os.uname()[1]
pid = os.getpid()

if args.name:
    name = args.name
else:
    name = f"{date}-{hostname}"

GIT_REVISION = get_git_revision()


if not args.disable_git_asserts:
    modified_files = get_modified_files()
    assert (
        modified_files == ""
    ), f"list of modified files (git) is nonempty:\n{modified_files}\nRun in a clean git tree or use --disable_git_asserts to disable this assert."


OUTPUT_DIR = f"models/unet/{dataset_name}/{name}"
LOSS_DIR = f"{OUTPUT_DIR}/loss"
CHECKPOINTS_DIR = f"{OUTPUT_DIR}/checkpoints"
LOG_PATH = f"{OUTPUT_DIR}/log.jsonl"

print(f"OUTPUT_DIR={OUTPUT_DIR}")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOSS_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

torch.use_deterministic_algorithms(True)

print(f"device={device}")


hparams = vars(args)

with open(f"{OUTPUT_DIR}/hparams.json", "w") as f:
    hparams["git"] = GIT_REVISION
    json.dump(hparams, f, indent=4)


sc_train_data = TiffDataset(
    args.dataset_path,
    "train",
    image_size=args.image_size,
)
sc_val_data = TiffDataset(
    args.dataset_path,
    "val",
    image_size=args.image_size,
)

first_image, first_label = sc_train_data[0]
model = Model(  # type: ignore
    bilinear=False,
    image_size=args.image_size,
    n_channels=first_image.shape[0],
).to(device=device)

pos_weight = torch.Tensor([args.pos_weight]).to(device=device)
# model.to(device)
loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
# optimizer = optim.Adam(model.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)


train_data = sc_train_data
val_data = sc_val_data


g = torch.Generator()
g.manual_seed(0)

train_dataloader = DataLoader(
    train_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, generator=g
)
val_dataloader = DataLoader(val_data, batch_size=args.batch_size, num_workers=args.num_workers)

completed_epochs = [-1]

if os.path.exists(LOG_PATH):
    with open(LOG_PATH, "r") as f:
        lines = f.readlines()
        log_data = [json.loads(x) for x in lines]
        completed_epochs = [-1] + [x["epoch"] for x in log_data if x["status"] == "done"]


# for epoch in range(args.max_epochs):  # loop over the dataset multiple times
while max(completed_epochs) < args.max_epochs - 1:
    train_losses = []

    running_loss = 0.0

    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, "r") as f:
            lines = f.readlines()
            log_data = [json.loads(x) for x in lines]
            completed_epochs = [-1] + [x["epoch"] for x in log_data if x["status"] == "done"]

    epoch = max(completed_epochs) + 1

    with open(LOG_PATH, "a") as f:
        f.write(
            json.dumps(
                {
                    "epoch": epoch,
                    "status": "started",
                    "pid": pid,
                    "host": hostname,
                    "git": GIT_REVISION,
                    "time": time.time(),
                }
            )
            + "\n"
        )
        f.flush()

    TRAIN_LOSS_PATH = "{}/{:08d}-train_loss.jsonl".format(LOSS_DIR, epoch)
    VAL_LOSS_PATH = "{}/{:08d}-val_loss.jsonl".format(LOSS_DIR, epoch)

    if os.path.exists(TRAIN_LOSS_PATH):
        os.remove(TRAIN_LOSS_PATH)

    if os.path.exists(VAL_LOSS_PATH):
        os.remove(VAL_LOSS_PATH)

    if epoch > 0:
        checkpoint_path = "{:s}/epoch_{:08d}.ckp".format(CHECKPOINTS_DIR, epoch - 1)

        checkpoint = torch.load(checkpoint_path)

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        g.set_state(checkpoint["generator"])

        if not args.disable_git_asserts:
            assert (
                GIT_REVISION == checkpoint["git"]
            ), f"current git revision {GIT_REVISION} differs from last saved checkpoint's {checkpoint['git']}. Run in a clean git tree or use --disable_git_asserts to disable this assert."

    for i, data in enumerate(tqdm(train_dataloader), 0):
        # get the inputs; data is a list of [inputs, labels, names]
        inputs, labels = data

        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        train_losses.append(loss.item())
        running_loss += loss.item()

        with open(TRAIN_LOSS_PATH, "a") as f:
            loss_info = {
                "epoch": epoch,
                "i": i,
                "loss": loss.item(),
            }
            f.write(json.dumps(loss_info) + "\n")

        if i % 200 == 199:  # print every 2000 mini-batches
            tqdm.write(f"[{epoch}, {i + 1:8d}] train loss: {running_loss / 200:.19f}")
            running_loss = 0.0

    with torch.no_grad():
        model.eval()

        avg_val_loss = 0
        for i, data in enumerate(tqdm(val_dataloader, desc="validation")):
            # get the inputs; data is a list of [inputs, labels, names]
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = loss_function(outputs, labels)

            with open(VAL_LOSS_PATH, "a") as f:
                loss_info = {
                    "epoch": epoch,
                    "i": i,
                    "loss": loss.item(),
                }
                f.write(json.dumps(loss_info) + "\n")
            avg_val_loss += loss.item()
        avg_val_loss /= len(val_dataloader)
        tqdm.write(f"validation loss: {avg_val_loss}")

    model.train()

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "generator": g.get_state(),
            "hparams": vars(args),
            "epoch": epoch,
            "git": GIT_REVISION,
            "val_loss": avg_val_loss,
            "time": time.time(),
        },
        "{:s}/epoch_{:08d}.ckp".format(CHECKPOINTS_DIR, epoch),
    )

    with open(LOG_PATH, "a") as f:
        f.write(
            json.dumps(
                {
                    "epoch": epoch,
                    "status": "done",
                    "pid": pid,
                    "host": hostname,
                    "git": GIT_REVISION,
                    "val_loss": avg_val_loss,
                    "time": time.time(),
                }
            )
            + "\n"
        )
        f.flush()

    completed_epochs += [epoch]
