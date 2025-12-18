import argparse
from pathlib import Path

from torchvision.utils import save_image
from tqdm import trange

from src.utils.model import SingleClassHdfDataset2d

parser = argparse.ArgumentParser()

parser.add_argument(
    "-d",
    "--dataset_path",
    default="data/processed/datasets/1000-toy.json",
    type=str,
)

parser.add_argument(
    "--only_birads_4",
    default=True,
    type=bool,
)

parser.add_argument(
    "-s",
    "--image_size",
    default=512,
    type=int,
    help="side of the square images. if image_size==None, use original resolution",
)

parser.add_argument(
    "-da",
    "--data_augmentation",
    default=True,
    help="use data augmentation",
)

parser.add_argument(
    "-ni",
    "--normalize_inputs",
    default=True,
    type=bool,
)

args = parser.parse_args()
hparams = vars(args)

OUTPUT_FOLDER = Path("data") / "processed" / "pngs" / Path(hparams["dataset_path"]).stem

OUTPUT_FOLDER /= f"image_size-{hparams['image_size']}"

if hparams["normalize_inputs"]:
    OUTPUT_FOLDER /= "normalized"
else:
    OUTPUT_FOLDER /= "not_normalized"

OUTPUT_FOLDER.mkdir(exist_ok=True, parents=True)
print(f"{OUTPUT_FOLDER}")

for subset in ["test", "val"]:
    subset_folder = OUTPUT_FOLDER / subset
    subset_folder.mkdir(exist_ok=True, parents=True)

    dataset = SingleClassHdfDataset2d(
        hparams["dataset_path"],
        subset,
        image_size=hparams["image_size"],
        only_birads_4=hparams["only_birads_4"],
        normalize=hparams["normalize_inputs"],
        # balance=hparams["data_augmentation"],
        binary_classification=True,
    )

    for i in trange(len(dataset), desc=subset, ncols=60):
        X, y = dataset[i]
        info = dataset.get_info(i)

        output_path = (
            subset_folder / f"{info['exam']}-{info['slice']:03d}-{info['subslice']}.png"
        )

        save_image(X[0], output_path)
