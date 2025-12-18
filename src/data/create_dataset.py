import argparse
import json
import os
import random
import subprocess
from glob import glob

from unique_names_generator import get_random_name

OUTPUT_DIR = "data/processed/datasets"

parser = argparse.ArgumentParser()

parser.add_argument(
    "--name",
    type=str,
)

parser.add_argument(
    "--md5",
    type=str,
    default=None,
    help="path to the processed md5 file. Default behaviour is to pick the latest.",
)
parser.add_argument(
    "--test_proportion",
    type=float,
    default=0.15,
    help="proportion of the test subset.",
)
parser.add_argument(
    "--val_proportion",
    type=float,
    default=0.15,
    help="proportion of the validation subset.",
)
parser.add_argument(
    "-s",
    "--size",
    type=int,
    help="limit size of the dataset",
)
parser.add_argument(
    "--seed",
    type=int,
    default=0,
)
args = parser.parse_args()

if args.md5:
    md5_path = args.md5
else:
    md5_path = sorted(glob("data/processed/*.md5"))[-1]

with open(md5_path, "r") as f:
    md5_lines = f.readlines()

random.seed(args.seed)

paths = sorted(list(set([os.path.dirname(e[34:-1]) for e in md5_lines])))
random.shuffle(paths)

if args.size:
    paths = paths[: args.size]

test_size = int(len(paths) * args.test_proportion)
val_size = int(len(paths) * args.val_proportion)

test = sorted(paths[:test_size])
val = sorted(paths[test_size : test_size + val_size])
train = sorted(paths[test_size + val_size :])

if args.name:
    name = args.name
else:
    random.seed(json.dumps([test, val, train, md5_path]))
    name = get_random_name(separator="_", style="lowercase")
    name += f"-{len(paths)}"

print(f"name={name}")
dataset = {}
dataset["test"] = test
dataset["val"] = val
dataset["train"] = train
dataset["md5_path"] = md5_path
dataset["date"] = subprocess.check_output(["date", "-Is"]).decode("utf-8")[:-1]
dataset["name"] = name
dataset["args"] = vars(args)

os.makedirs(OUTPUT_DIR, exist_ok=True)

output_path = OUTPUT_DIR + "/" + name + ".json"

if os.path.exists(output_path):
    print(f"warning: {output_path} already exists, ignoring.")
else:
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=4)
