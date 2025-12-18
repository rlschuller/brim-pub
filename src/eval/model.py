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
    "--predictions_folder",
    type=str,
    default="models/mlp/crimson_duck-844/fit_working_gerrilee/predictions"
)

parser.add_argument(
    "-d",
    "--dataset_path",
    type=str,
    default="data/processed/datasets/crimson_duck-839.json",
)

parser.add_argument(
    "--min_threshold",
    type=float,
    default=0,
)

parser.add_argument(
    "--max_threshold",
    type=float,
    default=1,
)

parser.add_argument(
    "-nt",
    "--n_thresholds",
    type=int,
    default=1001,
)

parser.add_argument(
    "--flip",
    action="store_true",
    help="flip exams listed in 'data/processed/flip_IDs.json' wrt axis=2 (4D)",
)

parser.add_argument(
    "--union",
    action="store_true",
    help="use union of birads 3 and 4 as labels",
)

parser.add_argument

EPS = 1e-5

args = parser.parse_args()

id_2_path = {}

for p in Path(args.predictions_folder).glob("**/*.npz"):
    id = p.stem
    id_2_path[id] = str(p)

if args.min_threshold:
    raise NotImplementedError("negative --min_threshold is not supported")

output_path = Path(args.predictions_folder).parent / f"{Path(args.predictions_folder).name}-eval.json"

out = {}

step = (args.max_threshold - args.min_threshold) / (args.n_thresholds-1)
thresholds = list(np.arange(args.min_threshold, args.max_threshold, step)) + [args.max_threshold]

flip_ids = set()

if args.flip:
    with Path("data/processed/flip_IDs.json").open("r") as f:
        flip_ids = set(json.load(f))

for subset in ["test", "val"]:
    out[subset] = {}

    for t in thresholds:
        out[subset][t] = {}
        out[subset][t]["id_2_metrics"] = {}
        out[subset][t]["tp"] = 0
        out[subset][t]["fn"] = 0
        out[subset][t]["tn"] = 0
        out[subset][t]["fp"] = 0
        out[subset][t]["mean"] = {}
        out[subset][t]["mean"]["precision"] = 0
        out[subset][t]["mean"]["recall"]    = 0
        out[subset][t]["mean"]["f1"]        = 0
        out[subset][t]["mean"]["accuracy"]  = 0

    #f = open(f"{predictions_folder}/{subset}.jsonl", "w")
    #dataset = TiffDataset(
    #    hparams["dataset_path"], subset=subset, image_size=hparams["image_size"], channels=channels
    #)
    dataset = NrrdDataset(
        args.dataset_path,
        subset,
        #normalization_quantile=hparams["normalization_quantile"],
    )

    acc_loss = 0
    n = 0


    total = len(dataset)


    for X, y, metadata in tqdm(dataset, desc=subset, ncols=60):
        pass
        X = np.array(X) #.to(device)#.unsqueeze(0)

        y = np.array(y > .5, dtype=np.float64)#.to(device)#.unsqueeze(0)

        id = metadata["id"].split("/")[-1]

        if id in flip_ids:
            X = np.flip(X, 2)
            y = np.flip(y, 2)


        if args.union:
            y = np.array((y[0,:,:,:] + y[1,:,:,:]) > .5, dtype=np.float64)

        npz_path = Path(id_2_path[id])
        pred = np.load(npz_path)
        pred = pred[list(pred.keys())[0]]

        if args.union and len(pred.shape) == 4:
            pred = np.maximum(pred[0, :, :, :], pred[1, :, :, :])

        if y.shape != pred.shape:
            if len(pred.shape) == 3:
                pred = torch.tensor(pred).unsqueeze(0).unsqueeze(0)
                pred = np.array(torch.nn.functional.interpolate(pred, size=y.shape[-3:]).squeeze(0).squeeze(0), dtype=np.float64)
            else:
                pred = torch.tensor(pred).unsqueeze(0)
                pred = np.array(torch.nn.functional.interpolate(pred, size=y.shape[-3:]).squeeze(0), dtype=np.float64)


        assert pred.shape == y.shape, f"{pred.shape=} is not equal to {y.shape=}"

        y_pred = np.sort((y * pred).flatten())
        ny_pred = np.sort(((1 - y) * pred).flatten())

        n_pos = np.sum(y)
        n_neg = len(y_pred) - n_pos

        i = 0
        j = 0

        for t in thresholds:

            while i < len(y_pred) and y_pred[i] <= t:
                i+=1

            while j < len(ny_pred) and ny_pred[j] <= t:
                j+=1


            tp = len(y_pred) - i
            fn = n_pos - tp

            fp = len(ny_pred) - j
            tn = n_neg - fp


            if (tp + fp) > 0:
                precision = tp / (tp + fp)
            else:
                precision = 1

            if (tp + fn) > 0:
                recall = tp / (tp + fn)
            else:
                recall = 1

            assert tp + fn == n_pos, "{tp + fn=} and {n_pos=} differ"
            if precision + recall <= EPS:
                f1 = 0
            else:
                f1 = 2 * precision * recall / (precision + recall)

            accuracy = (tp + tn) / len(y_pred)

            out[subset][t]["id_2_metrics"][id] = {}

            out[subset][t]["id_2_metrics"][id]["tp"] = tp
            out[subset][t]["id_2_metrics"][id]["fn"] = fn
            out[subset][t]["id_2_metrics"][id]["tn"] = tn
            out[subset][t]["id_2_metrics"][id]["fp"] = fp
            out[subset][t]["id_2_metrics"][id]["precision"] = precision
            out[subset][t]["id_2_metrics"][id]["recall"]    = recall
            out[subset][t]["id_2_metrics"][id]["f1"]        = f1
            out[subset][t]["id_2_metrics"][id]["accuracy"]  = accuracy

            out[subset][t]["tp"] += tp
            out[subset][t]["fn"] += fn
            out[subset][t]["tn"] += tn
            out[subset][t]["fp"] += fp
            out[subset][t]["mean"]["precision"] += precision / len(dataset)
            out[subset][t]["mean"]["recall"]    += recall / len(dataset)
            out[subset][t]["mean"]["f1"]        += f1 / len(dataset)
            out[subset][t]["mean"]["accuracy"]  += accuracy / len(dataset)

try:
    with output_path.open("w") as f:
        json.dump(out, f,indent=2)
except:
    with Path(str(output_path).replace("/","--")).open("w") as f:
        json.dump(out, f,indent=2)


