import argparse
import hashlib
import json
import os
import random
from math import floor

import numpy as np
import tifffile
from unique_names_generator import get_random_name

from src.utils.log import get_log_info

parser = argparse.ArgumentParser()

parser.add_argument(
    "-d",
    "--dataset",
    type=str,
)

parser.add_argument(
    "-m",
    "--metric",
    type=str,
    default="average-decrease-ndfi",
    help="a metric defined in the data/processed/metrics.jsonl file",
)

parser.add_argument(
    "-q", "--quantile", type=float, default=0.3, help="controls the number of selected entries, value in (0, 1)"
)

parser.add_argument(
    "-gt",
    "--greater",
    action="store_true",
    help="select entries with metric > threshold, instead of the default metric < threshold",
)

parser.add_argument(
    "-l",
    "--list_metrics",
    action="store_true",
    help="list all of available metrics and exit",
)

args = parser.parse_args()

METRICS_FILE = "data/processed/metrics.jsonl"

if args.list_metrics:
    with open(METRICS_FILE, "r") as f:
        first_line = f.readline()
    metrics = sorted([k for k in json.loads(first_line) if k != "id" and k != "label"])

    for m in metrics:
        print(m)
    exit(0)


hparams = vars(args)
hparams.pop("list_metrics")
random.seed(json.dumps(hparams))
name = get_random_name(separator="_", style="lowercase")
md5_hash = hashlib.md5(json.dumps(hparams).encode()).hexdigest()
name += f"-{md5_hash}"

OUTPUT_FOLDER = f"models/threshold/{os.path.basename(args.dataset).split('.')[0]}/{name}"
PREDICTION_FOLDER = f"{OUTPUT_FOLDER}/prediction"

with open(args.dataset, "r") as f:
    dataset = json.load(f)

os.makedirs(PREDICTION_FOLDER, exist_ok=True)

hparams["log"] = get_log_info()


with open(f"{OUTPUT_FOLDER}/hparams.json", "w") as f:
    f.write(json.dumps(hparams, indent=4))

# read the metrics
metric_data = []
with open(METRICS_FILE, "r") as f:
    for line in f.readlines():
        entry = json.loads(line)
        metric_data.append([entry[args.metric], entry["id"], entry["label"]])

if args.greater:
    metric_data = [[-x[0], x[1], x[2]] for x in metric_data]

metric_data.sort(key=lambda x: x[0])


# use the train dataset to compute the quantiles
n_select_train = max(floor(len(dataset["train"]) * args.quantile), 1)


train_ids = set([os.path.basename(x).split(".")[0] for x in dataset["train"]])

train_metric_data = [x for x in metric_data if x[1] in train_ids]
threshold = train_metric_data[n_select_train - 1][0]

for subset_name in ["train", "val", "test"]:
    subset = set(dataset[subset_name])
    subset_ids = set([os.path.basename(x).split(".")[0] for x in dataset[subset_name]])
    subset_data = [x for x in metric_data if x[1] in subset_ids]
    with open(f"{PREDICTION_FOLDER}/{subset_name}.jsonl", "w") as f:
        for x in subset_data:
            f.write(
                json.dumps({"id": x[1], "score": x[0], "prediction": int(x[0] < threshold), "target": int(x[2])}) + "\n"
            )
