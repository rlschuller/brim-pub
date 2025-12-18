from tqdm import tqdm

from src.utils.model import *

# flake8: noqa

dataset_2d = SingleClassHdfDataset2d(
    "data/processed/datasets/toy.json",
    "train",
    image_size=256,
    only_birads_4=True,
)

generator = torch.Generator()
generator.manual_seed(0)

shuffled_dataset = PartiallyShuffledSplitImages(
    dataset_2d, image_size=64, n_chunks_to_be_shuffled=32, generator=generator
)

print(shuffled_dataset.shuffled_indexes)
print(shuffled_dataset[0])


dataset_2d_bin = SingleClassHdfDataset2d(
    "data/processed/datasets/toy.json",
    "train",
    image_size=256,
    only_birads_4=True,
    binary_classification=True,
)
x, Y = dataset_2d_bin[0]

num_nonzero = 0
for i in tqdm(range(len(dataset_2d)), desc="test binary_classification"):
    x, Y = dataset_2d[i]
    x_bin, Y_bin = dataset_2d_bin[i]

    assert torch.equal(x, x)
    assert (Y.sum() != 0) == (Y_bin.sum() != 0)

    num_nonzero += Y_bin.sum() != 0


num_zero = len(dataset_2d) - num_nonzero
print(f"{num_nonzero=}")
print(f"{num_zero=}")

num_nonzero = 0
for i in tqdm(range(len(shuffled_dataset)), desc="test shuffled_dataset"):
    x, Y = shuffled_dataset[i]
    num_nonzero += Y.sum() != 0

num_zero = len(shuffled_dataset) - num_nonzero
print(f"{num_nonzero=}")
print(f"{num_zero=}")
