import itertools
import json
import os
import time

import h5py
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from tqdm import tqdm

from src.utils.general import get_git_revision


class SplitImages(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_2d,
        image_size=64,
        half_stride=False,
        binary_classification=False,
    ) -> None:
        if half_stride:
            raise NotImplementedError

        self.hparams = {k: dataset_2d.hparams[k] for k in dataset_2d.hparams.keys()}

        self.hparams["split_image_size"] = image_size
        self.hparams["half_stride"] = half_stride
        # print(json.dumps(self.hparams))

        self.dataset_2d = dataset_2d
        self.n = dataset_2d.image_size / image_size
        self.image_size = image_size
        # print(len(dataset_2d))
        assert int(self.n) == self.n, "Non integer stride"

        self.n = int(self.n)

        self.length = int(dataset_2d.length * self.n * self.n)
        self.binary_classification = binary_classification

    def __len__(self) -> int:
        return self.length

    def get_info(self, i: int):
        super_i = i // (self.n * self.n)
        sub_i = i - super_i * self.n * self.n
        info = self.dataset_2d.get_info(super_i)
        info["subslice"] = sub_i

        return info

    def __getitem__(self, i: int):
        super_i = i // (self.n * self.n)

        exam, target = self.dataset_2d[super_i]

        sub_i = i - super_i * self.n * self.n

        row = sub_i // self.n
        col = sub_i % self.n
        exam = exam[
            :,
            row * self.image_size : (row + 1) * self.image_size,
            col * self.image_size : (col + 1) * self.image_size,
        ]
        target = target[
            :,
            row * self.image_size : (row + 1) * self.image_size,
            col * self.image_size : (col + 1) * self.image_size,
        ]

        if self.binary_classification:
            target = torch.ones([1]) * (torch.sum(target) != 0)
        return exam, target


def shuffled_with_torch_generator(array, generator):
    perm = torch.randperm(len(array), generator=generator)
    return [array[perm[i]] for i in range(len(array))]


class PartiallyShuffledSplitImages(torch.utils.data.Dataset):
    """Perform a partial shuffle in SplitImages dataset (use it with shuffle=False)

    Avoid severe bottlenecks due to seek times, which are a consequence of the
    tiny images.
    """

    def __init__(
        self,
        dataset_2d,
        image_size=64,
        half_stride=False,
        n_chunks_to_be_shuffled=32,
        generator=None,
        binary_classification=False,
    ) -> None:
        assert generator

        if half_stride:
            raise NotImplementedError

        if image_size > dataset_2d.image_size:
            self.dataset = dataset_2d
        else:
            self.dataset = SplitImages(
                dataset_2d,
                image_size,
                half_stride,
                binary_classification=binary_classification,
            )

        self.dataset_2d = dataset_2d
        self.image_size = image_size
        self.n_chunks_to_be_shuffled = n_chunks_to_be_shuffled
        self.shuffle(generator)

    def shuffle(self, generator):
        list_of_indexes = list(range(len(self.dataset)))

        chunk_size = int((self.dataset_2d.image_size / self.image_size) ** 2)
        list_of_chunks = []

        for s in range(0, len(list_of_indexes), chunk_size):
            chunk = list_of_indexes[s : s + chunk_size]
            list_of_chunks.append(chunk)

        list_of_chunks = shuffled_with_torch_generator(list_of_chunks, generator)
        self.shuffled_indexes = []

        for k in range(0, len(list_of_chunks), self.n_chunks_to_be_shuffled):
            new_indexes = list(
                itertools.chain.from_iterable(
                    list_of_chunks[k : k + self.n_chunks_to_be_shuffled]
                )
            )
            # torch.randperm(new_indexes, generator=generator)
            new_indexes = shuffled_with_torch_generator(new_indexes, generator)
            self.shuffled_indexes.extend(new_indexes)

        assert sorted(self.shuffled_indexes) == sorted(list_of_indexes)

    def get_info(self, i: int):
        return self.dataset.get_info(self.shuffled_indexes[i])

    def __len__(self) -> int:
        return len(self.shuffled_indexes)

    def __getitem__(self, i: int):
        return self.dataset[self.shuffled_indexes[i]]


class SingleClassHdfDataset2d(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_path,
        subset,
        image_size=512,
        only_birads_4=False,
        binary_classification=False,
        balance=False,
        normalize=True,
    ) -> None:

        self.hparams = {
            "dataset_path": dataset_path,
            "subset": subset,
            "image_size": image_size,
            "only_birads_4": only_birads_4,
            "binary_classification": binary_classification,
            "balance": balance,
        }

        assert subset in ["train", "test", "val"]
        self.image_size = image_size
        self.only_birads_4 = only_birads_4
        self.binary_classification = binary_classification
        self.balance = balance
        self.normalize = normalize

        with open(dataset_path, "r") as f:
            self.dataset = json.load(f)

        hdf_path = self.dataset["md5_path"][:-3] + "hdf5"

        self.hdf_file = h5py.File(hdf_path, "r")

        hdf_groups = [x.replace("/", "_") for x in self.dataset[subset]]
        self.list_of_exam_and_target_key_pairs = []

        pos_count = 0
        neg_count = 0
        pos_multiplier = 1
        is_pos = {}
        if balance:
            for g in tqdm(hdf_groups, desc="computing balance"):
                slice_keys = list(self.hdf_file[g]["exam_slices"].keys())
                slice_keys.sort(key=int)

                for slice_key in slice_keys:
                    birads_slice = np.array(
                        self.hdf_file[g]["birads_slices"][slice_key]
                    )

                    if only_birads_4:
                        birads_slice = birads_slice == 4

                    birads_sum = np.sum(birads_slice != 0)
                    is_pos[str(g), slice_key] = birads_sum > 0
                    pos_count += is_pos[str(g), slice_key]
                    neg_count += not is_pos[str(g), slice_key]

            pos_multiplier = neg_count // pos_count

        for g in hdf_groups:
            assert len(self.hdf_file[g]["exam_slices"]) == len(
                self.hdf_file[g]["birads_slices"]
            )

            slice_keys = list(self.hdf_file[g]["exam_slices"].keys())
            slice_keys.sort(key=int)
            for slice_key in slice_keys:
                multiplier = 1

                if pos_multiplier > 1 and is_pos[str(g), slice_key]:
                    multiplier = pos_multiplier

                for i in range(multiplier):
                    self.list_of_exam_and_target_key_pairs.append(
                        [
                            f"{g}/exam_slices/{slice_key}",
                            f"{g}/birads_slices/{slice_key}",
                        ]
                    )

        self.length = len(self.list_of_exam_and_target_key_pairs)
        # print(json.dumps(list_of_slices))

    def __len__(self) -> int:
        return self.length

    def shuffle(self, generator):

        self.list_of_exam_and_target_key_pairs = shuffled_with_torch_generator(
            sorted(self.list_of_exam_and_target_key_pairs), generator
        )

    def get_info(self, i: int):
        info = {
            "subslice": 0,
            "slice": int(self.list_of_exam_and_target_key_pairs[i][0].split("/")[-1]),
            "exam": self.list_of_exam_and_target_key_pairs[i][0].split("/")[0],
        }
        return info

    def __getitem__(self, i: int):
        keys = self.list_of_exam_and_target_key_pairs[i]
        exam_key = keys[0]
        target_key = keys[1]

        transform = T.Resize(size=(self.image_size, self.image_size))

        exam = np.array(self.hdf_file[exam_key], dtype=np.float32)
        exam = exam / np.max(exam)

        exam = np.array(
            [
                np.array(transform(TF.to_pil_image(exam[i, :, :])))
                for i in range(exam.shape[0])
            ],
            dtype=np.float32,
        )

        if self.only_birads_4:
            target = np.array(
                np.array(self.hdf_file[target_key]) == 4, dtype=np.float32
            )
        else:
            target = np.array(
                np.array(self.hdf_file[target_key]), dtype=np.float32
            )

        target = np.array(transform(TF.to_pil_image(target)), dtype=np.float32)

        # make dimensions in exam and target compatible
        target = np.expand_dims(target, axis=2)

        exam = torch.Tensor(exam)
        target = torch.Tensor(target)
        target = target.permute(2, 0, 1)

        if self.balance:
            angle = (i % 4) * 90
            flip = (i // 8) % 2
            exam = T.functional.rotate(exam, angle=angle)
            target = T.functional.rotate(target, angle=angle)

            if flip:
                exam = T.functional.hflip(exam)
                target = T.functional.hflip(target)

        if self.binary_classification:
            target = torch.ones([1]) * (torch.sum(target) != 0)

        if self.normalize:
            exam = exam / torch.max(exam)

        return exam, target


def train_loop(
    model,
    hparams,
    max_epochs,
    optimizer,
    loss_function,
    generator,
    train_dataloader,
    train_dataset,
    val_dataloader,
    dataset_name,
    run_name,
    output_dir,
    device,
    proportion_train=None,
):

    LOSS_DIR = f"{output_dir}/loss"
    CHECKPOINTS_DIR = f"{output_dir}/checkpoints"
    LOG_PATH = f"{output_dir}/log.jsonl"

    GIT_REVISION = get_git_revision()
    hostname = os.uname()[1]
    pid = os.getpid()

    completed_epochs = [-1]

    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, "r") as f:
            lines = f.readlines()
            log_data = [json.loads(x) for x in lines]
            completed_epochs = [-1] + [
                x["epoch"] for x in log_data if x["status"] == "done"
            ]

    print(train_dataset)
    print(train_dataloader)
    # for epoch in range(max_epochs):  # loop over the dataset multiple times
    while max(completed_epochs) < max_epochs - 1:
        train_losses = []

        running_loss = 0.0

        if os.path.exists(LOG_PATH):
            with open(LOG_PATH, "r") as f:
                lines = f.readlines()
                log_data = [json.loads(x) for x in lines]
                completed_epochs = [-1] + [
                    x["epoch"] for x in log_data if x["status"] == "done"
                ]

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
            generator.set_state(checkpoint["generator"])

        train_dataset.shuffle(generator)
        train_length = len(train_dataloader)
        if proportion_train:
            train_length = len(train_dataloader)

        for i, data in enumerate(tqdm(train_dataloader, total=train_length), 0):
            if i < train_length:
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
                    tqdm.write(
                        f"[{epoch}, {i + 1:8d}] train loss: {running_loss / 200:.19f} {json.dumps(train_dataset.get_info(i))}"
                    )
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
                "generator": generator.get_state(),
                "hparams": hparams,
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

class EarlyStop:
    def __init__(self, patience):
        self.patience = patience
        self.history = []

    def __call__(self, loss):
        self.history.append(loss)

        min_x = 9999999
        min_i = 9999999
        for i, x in enumerate(self.history):
            if x < min_x:
                min_x = x
                min_i = i

        if len(self.history) - min_i > self.patience:
            print("stopping early...")
            exit(0)

