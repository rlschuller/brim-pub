import argparse
import csv
import gc
import hashlib
import json
import os
import shutil
import subprocess
from glob import glob
from typing import OrderedDict, Tuple

import h5py
import medpy.io
import numpy as np
from tqdm import tqdm

NRRD_OUTPUT_DIR = "data/processed/nrrd"


TABLE_PATH = "data/misc/table.csv"
OUTPUT_DIR = "data/processed"
REQUIRED_EXAM_BASENAMES = sorted([f"A{n}.nrrd" for n in range(1, 6)])
DEFAULT_BLACKLIST_FOLDER = "data/blacklist"
DEFAULT_GLOB_EXPRESSION = "data/raw/*/*"

parser = argparse.ArgumentParser()

parser.add_argument(
    "--dry", action="store_true", help="only print errors, without saving files"
)

parser.add_argument(
    "--glob",
    type=str,
    default=DEFAULT_GLOB_EXPRESSION,
    help="glob expression to capture all of the subject_dirs",
)

parser.add_argument(
    "--blacklist_folder",
    type=str,
    default=DEFAULT_BLACKLIST_FOLDER,
    help=(
        "folder that contains blacklisted subjects. a blacklist json file is a dictionary "
        "{subpath: list_of_strings}, in which list_of_strings is a list of errors."
    ),
)

parser.add_argument(
    "--ignore_blacklist",
    action="store_true",
)

parser.add_argument(
    "--ignore_table",
    action="store_true",
)

args = parser.parse_args()

assert args.dry or (args.glob == DEFAULT_GLOB_EXPRESSION)

BLACKLIST_FOLDER = args.blacklist_folder

if args.dry:
    tqdm = list

if os.path.exists(NRRD_OUTPUT_DIR):
    shutil.rmtree(NRRD_OUTPUT_DIR)

if os.path.exists(NRRD_OUTPUT_DIR + ".md5"):
    os.remove(NRRD_OUTPUT_DIR + ".md5")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def update_graph(graph_of_hashes, list_of_hashes):
    """Make the entries in list_of_hashes connected in graph_of_hashes

    Inputs:
    - a graph G, represented by a dictionary in which
    for all k in G.keys(), G[k] is the set of all hashes connected to k
    - a list of hashes to be connected among themselves
    """
    old = set()
    new = set(list_of_hashes)

    while len(old) != len(new):
        old = set(new)
        for x in old:
            if x not in graph_of_hashes:
                graph_of_hashes[x] = set()
            new.update(graph_of_hashes[x])

    for x in new:
        graph_of_hashes[x] = list(new)


def get_clusters(graph_of_hashes, hash_to_index):
    visited_hash = set()
    list_of_clusters = []
    for k in graph_of_hashes.keys():
        if k not in visited_hash:
            hash_cluster = graph_of_hashes[k]
            visited_hash.update(hash_cluster)
            cluster = set()
            for h in hash_cluster:
                cluster.update(hash_to_index[h])
            cluster = list(cluster)
            cluster.sort()
            list_of_clusters.append(cluster)
    return list_of_clusters


def main():
    if args.ignore_table:
        subject_id_and_segmentation_name_to_birads = dict()
    else:
        subject_id_and_segmentation_name_to_birads = load_diagnostics_table(TABLE_PATH)

    blacklist = dict()

    if not args.ignore_blacklist:
        for blacklist_path in sorted(glob(f"{BLACKLIST_FOLDER}/*.json")):
            print(f"{blacklist_path=}")
            with open(blacklist_path) as f:
                blacklist = blacklist | json.load(f)

    hashes_to_subject_dirs = {}
    graph_of_hashes = {}

    for subject_dir in tqdm(sorted(glob(args.glob))):
        try:
            if subject_dir in blacklist:
                str_error = f"{subject_dir}: " + "; ".join(blacklist[subject_dir])
                if args.dry:
                    print(str_error)
                else:
                    tqdm.write(str_error)
                continue

            check_paths(subject_dir, subject_id_and_segmentation_name_to_birads)
            exams = load_exams(subject_dir)

            exam_hashes = get_hashes_from_exams(exams)

            for h in exam_hashes:
                if h not in hashes_to_subject_dirs:
                    hashes_to_subject_dirs[h] = []
                hashes_to_subject_dirs[h].append(subject_dir)

            update_graph(graph_of_hashes, exam_hashes)

            birads = load_birads(
                subject_dir,
                exams[0][0].shape,
                subject_id_and_segmentation_name_to_birads,
            )

            if subject_dir not in blacklist:
                save_subject(exams, birads, subject_dir)

        except (
            FileNotFoundError,
            AssertionError,
            ValueError,
            RuntimeError,
            NotImplementedError,
            KeyError,
        ) as err:
            if args.dry:
                print(f"{subject_dir}: {err}")
            else:
                tqdm.write(f"{subject_dir}: {err}")

            if subject_dir not in blacklist:
                blacklist[subject_dir] = []
            blacklist[subject_dir].append(str(err))

        gc.collect()

    list_of_clusters = get_clusters(graph_of_hashes, hashes_to_subject_dirs)

    for cluster in list_of_clusters:
        if len(cluster) > 1:
            for subject_dir in cluster:
                if subject_dir not in blacklist:
                    blacklist[subject_dir] = []

                blacklist[subject_dir].append(
                    f"hashes coincide with subject(s): {', '.join(sorted(set(cluster)-set([subject_dir])))}"
                )

                print(f"{subject_dir}: " + "; ".join(blacklist[subject_dir]))

    # remove all subjects in blacklist
    for subject_dir in blacklist:
        remove_subject(subject_dir)

    output_dir_in_versioning_system = add_output_to_versioning_system()

    with open(output_dir_in_versioning_system + ".blacklist", "w") as f:
        f.write(json.dumps(blacklist, indent=4))


    #if not args.dry:
    #    save_2d_hdf_dataset(output_dir_in_versioning_system)


def get_hashes_from_exams(exams):
    all_hashes = []
    for data, _ in exams:
        single_exam_hashes = []
        images = [
            np.array(data[int(data.shape[0] * 0.5), :, :], dtype=float),
            np.array(data[:, int(data.shape[1] * 0.5), :], dtype=float),
            np.array(data[:, :, int(data.shape[2] * 0.5)], dtype=float),
        ]

        for image in images:
            image_flip = np.fliplr(image)

            for k in range(4):
                single_exam_hashes.append(
                    hashlib.md5(
                        json.dumps(np.rot90(image, k=k).tolist()).encode()
                    ).hexdigest()
                )
                single_exam_hashes.append(
                    hashlib.md5(
                        json.dumps(np.rot90(image_flip, k=k).tolist()).encode()
                    ).hexdigest()
                )

        all_hashes.append(single_exam_hashes)

    for i in range(5):
        assert len(set(all_hashes[i])) == len(
            all_hashes[i]
        ), f"unexpected symmetry in 'A{i+1}.nrrd'"
        for j in range(i + 1, 5):
            assert (
                len(set(all_hashes[i]).intersection(set(all_hashes[j]))) == 0
            ), f"'A{i+1}.nrrd' and 'A{j+1}.nrrd' are repeated"

    return [h for hashes in all_hashes for h in hashes]


def save_2d_hdf_dataset(output_dir_in_versioning_system, axis=2):
    hdf_path = output_dir_in_versioning_system + ".hdf5"
    f = h5py.File(hdf_path, "w")

    for subject_dir in tqdm(
        sorted(glob(f"{output_dir_in_versioning_system}/*/*")),
        desc="saving hdf dataset",
    ):
        subject_subpath = "_".join(subject_dir.split("/")[-2:])

        subject_group = f.create_group(subject_subpath)
        exam_slices_group = subject_group.create_group("exam_slices")
        birads_slices_group = subject_group.create_group("birads_slices")

        exam_paths = [f"{subject_dir}/A{k}.nrrd" for k in range(1, 6)]
        birads_path = f"{subject_dir}/birads.nrrd"

        exams = [medpy.io.load(p) for p in exam_paths]
        birads = medpy.io.load(birads_path)

        for i in range(exams[0][0].shape[axis]):
            if axis == 0:
                exam_slice = np.array(
                    [exams[a][0][i, :, :] for a in range(0, 5)], dtype=np.float32
                )
                birads_slice = np.array(birads[0][i, :, :], dtype=np.uint8)
            if axis == 1:
                exam_slice = np.array(
                    [exams[a][0][:, i, :] for a in range(0, 5)], dtype=np.float32
                )
                birads_slice = np.array(birads[0][:, i, :], dtype=np.uint8)

            if axis == 2:
                exam_slice = np.array(
                    [exams[a][0][:, :, i] for a in range(0, 5)], dtype=np.float32
                )
                birads_slice = np.array(birads[0][:, :, i], dtype=np.uint8)

            exam_slices_group.create_dataset(f"{i}", data=exam_slice)
            birads_slices_group.create_dataset(f"{i}", data=birads_slice)


def add_output_to_versioning_system():
    subprocess.run(["src/utils/md5dir", NRRD_OUTPUT_DIR])

    with open(f"{NRRD_OUTPUT_DIR}.md5", "r") as f:
        md5lines = sorted(f.readlines())

    dirs = set()
    for line in md5lines:
        dirname = os.path.dirname(line[34:-1])
        dirs.add(dirname)

    print(f"\nnumber of subjects:\n{len(dirs)}")

    os.makedirs("data/processed/pool", exist_ok=True)
    for line in md5lines:
        md5 = line[:32]
        path = line[34:-1]

        if not os.path.exists(f"data/processed/pool/{md5}.nrrd"):
            shutil.move(f"{NRRD_OUTPUT_DIR}/{path}", f"data/processed/pool/{md5}.nrrd")

    if len(glob(f"{NRRD_OUTPUT_DIR}-*.md5")) > 0:
        last_md5_path = sorted(glob(f"{NRRD_OUTPUT_DIR}-*.md5"))[-1]
        with open(last_md5_path, "r") as f:
            last_md5lines = sorted(f.readlines())
    else:
        last_md5lines = None

    date = subprocess.check_output(["date", "-Is"]).decode("utf-8")[:-1]

    if last_md5lines is None or last_md5lines != md5lines:
        shutil.move(f"{NRRD_OUTPUT_DIR}.md5", f"{NRRD_OUTPUT_DIR}-{date}.md5")

        output_dir_in_versioning_system = f"{NRRD_OUTPUT_DIR}-{date}"
        for line in md5lines:
            md5 = line[:32]
            path = line[34:-1]
            dirname = os.path.dirname(f"{NRRD_OUTPUT_DIR}-{date}/{path}")
            os.makedirs(dirname, exist_ok=True)
            os.symlink(
                f"../../../pool/{md5}.nrrd", f"{output_dir_in_versioning_system}/{path}"
            )

        out = subprocess.check_output(f"python src/data/compute_metrics.py -p {output_dir_in_versioning_system}", shell=True)

    else:
        output_dir_in_versioning_system = last_md5_path.split(".")[0]
        os.remove(f"{NRRD_OUTPUT_DIR}.md5")

    if os.path.exists(NRRD_OUTPUT_DIR):
        shutil.rmtree(NRRD_OUTPUT_DIR)
        shutil.copytree(output_dir_in_versioning_system, NRRD_OUTPUT_DIR, symlinks=True)

    return output_dir_in_versioning_system


def load_diagnostics_table(table_path):
    # Example: {("01", "E1") : 3}
    subject_id_and_segmentation_name_to_birads = {}

    with open(table_path) as f:
        csv_reader = csv.reader(f)
        header = csv_reader.__next__()
        key_to_col = {header[i]: i for i in range(len(header))}

        table = []
        for row in csv_reader:
            table.append(row)

            subject_id_and_segmentation_name_to_birads[
                (int(row[key_to_col["Código"]]), row[key_to_col["anotação"]])
            ] = row[key_to_col["BI-RADS"]]

    return subject_id_and_segmentation_name_to_birads


def get_exam_basenames(subject_dir):
    return sorted([os.path.basename(x) for x in glob(f"{subject_dir}/A*.nrrd")])


def get_segmentation_basenames(subject_dir):
    exam_basenames = get_exam_basenames(subject_dir)

    segmentation_basenames = set(
        [os.path.basename(x) for x in glob(f"{subject_dir}/*.nrrd")]
    ) - set(exam_basenames)
    return segmentation_basenames


def get_raw_format_version(subject_dir):
    """Return format version of the given subject directory.

    Output: a string with one of the formats described bellow

    - "letters":
        A1,A2,A3,A4 e A5 are volumes generated, corresponding to the 5 times in which the machine registered an image;
        A6 is the image used for segmentation (usually a convex combination of A1-15);
        (B) is the segmentation that contains only 1 Bi-rads 3 tumor;
        (C) is the segmentation that contains only 1 Bi-rads 4 tumor;
        (D) is the segmentation that contains more than 1 Bi-rads 3 tumor;
        (E1,E2...) used when there are multiple tumors, with at least one of them Bi-rads 4 (use table)

    - "numbers":
        A1,A2,A3,A4 e A5 are volumes generated, corresponding to the 5 times in which the machine registered an image;
        A6 is the image used for segmentation (usually a convex combination of A1-15);
        (3) is the segmentation of Bi-rads 3 tumors;
        (4) is the segmentation of Bi-rads 4 tumors;
    """

    segmentation_basenames = get_segmentation_basenames(subject_dir)

    if segmentation_basenames.issubset(set(["3.nrrd", "4.nrrd"])):
        return "numbers"
    else:
        assert segmentation_basenames.issubset(
            set(["B.nrrd", "C.nrrd", "D.nrrd"] + [f"E{k}.nrrd" for k in range(1, 20)])
        ), f"invalid segmentation files: {', '.join(sorted(list(segmentation_basenames)))}"
        return "letters"


def check_paths(subject_dir, subject_id_and_segmentation_name_to_birads):
    subject_id = int(os.path.basename(subject_dir))
    exam_basenames = get_exam_basenames(subject_dir)
    segmentation_basenames = get_segmentation_basenames(subject_dir)

    raw_format = get_raw_format_version(subject_dir)

    assert (
        len(set(REQUIRED_EXAM_BASENAMES) - set(exam_basenames)) == 0
    ), f"missing the following files: {', '.join(sorted(list(set(REQUIRED_EXAM_BASENAMES)-set(exam_basenames))))}"

    if raw_format == "letters":
        # note that issubset allows empty segmentation_basenames
        assert (
            segmentation_basenames == set(["B.nrrd"])
            or segmentation_basenames == set(["C.nrrd"])
            or segmentation_basenames == set(["D.nrrd"])
            or segmentation_basenames.issubset(
                set([f"E{k}.nrrd" for k in range(1, 100)])
            )
        ), f"invalid segmentation files: {', '.join(sorted(list(segmentation_basenames)))}"

        if not args.ignore_table:
            for segmentation_basename in segmentation_basenames:
                segmentation_name = segmentation_basename.split(".")[0]
                try:
                    table_birads = int(
                        subject_id_and_segmentation_name_to_birads[
                            (subject_id, segmentation_name)
                        ]
                    )
                except KeyError:
                    raise ValueError(
                        f"folder '{subject_dir}' contains the file '{segmentation_basename}', but the table does not"
                    )

                if segmentation_name in ["B", "D"]:
                    assert (
                        table_birads == 3
                    ), f"segmentation name {segmentation_name} does not match the table information"
                if segmentation_name == "C":
                    assert (
                        table_birads == 4
                    ), f"segmentation name {segmentation_name} does not match the table information (BI-RADS  {table_birads})"
    elif raw_format == "numbers":
        pass
    else:
        raise NotImplementedError("support for the new format is not implemented yet")


def fix_flips(data, header):

    abs_closeness_to_identity = np.amax(
        abs(header.direction) - np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    )
    assert (
        abs_closeness_to_identity < 0.2
    ), f"unexpected orientation, with direction={json.dumps(header.direction.tolist())}"

    # check if orientation is consistent with a centralized image
    signs = [
        np.sign(header.direction[0][0]),
        np.sign(header.direction[1][1]),
        np.sign(header.direction[2][2]),
    ]
    origin_signs = np.sign(header.offset)

    assert (
        signs == -origin_signs
    ).all(), (
        f"image is not properly centralized, signs={signs}, origin_signs={origin_signs}"
    )

    # flip image if necessary
    for k in range(3):
        header.direction[:, k] *= signs[k]
        data = np.flip(data, k)

    header.offset = (
        header.offset[0] * signs[0],
        header.offset[1] * signs[1],
        header.offset[2] * signs[2],
    )

    return data, header


def load_exams(subject_dir):
    # load exams
    exams = []

    for basename in REQUIRED_EXAM_BASENAMES:
        data, header = medpy.io.load(os.path.join(subject_dir, basename))
        exams.append([data, header])

    assert all(
        [len(e[0].shape) == 3 for e in exams]
    ), "invalid number of dimensions found in 'AX.nrrd' for X in [1 : 5]"

    assert all(
        [e[0].shape == exams[0][0].shape for e in exams]
    ), "dimensions differ among the 'AX.nrrd', for X in [1 : 5]"

    matrix_exam = np.array(exams[0][1].direction)
    exam_origin = np.array(exams[0][1].offset)

    assert all(
        e[1].direction.shape == (3, 3) for e in exams
    ), f"invalid shape matrix_exam.shape={matrix_exam.shape}"

    assert all(
        len(e[1].offset) == 3 for e in exams
    ), f"invalid shape exam_origin.shape={exam_origin.shape}"

    assert all([np.allclose(matrix_exam, e[1].direction) for e in exams]), (
        "matrix_exam differ among A1-A5: \n"
        + "\n\n".join([str(e[1].direction) for e in exams])
        + "\n\n\n"
    )

    assert all([np.allclose(exam_origin, e[1].offset) for e in exams]), (
        "exam_origin differ among A1-A5: \n"
        + "\n\n".join([str(e[1].offset) for e in exams])
        + "\n\n\n"
    )

    # flip images if necessary
    for i in range(len(exams)):
        exams[i][0], exams[i][1] = fix_flips(exams[i][0], exams[i][1])

    return exams


def check_segmentation_compatibility(
    exam_basename,
    exam_data,
    exam_header,
    segmentation_basename,
    segmentation_data,
    segmentation_header,
):

    exam_matrix = np.array(exam_header.direction)
    exam_origin = np.array(exam_header.offset)

    segmentation_matrix = np.array(segmentation_header.direction)
    segmentation_origin = np.array(segmentation_header.offset)

    assert (
        exam_data.shape == segmentation_data.shape
    ), f"incompatible resolutions: {exam_basename}={exam_data.shape}, {segmentation_basename}={segmentation_data.shape}"

    assert np.allclose(
        exam_matrix, segmentation_matrix
    ), "incompatible exam and segmentation directions"
    assert np.allclose(
        exam_origin, segmentation_origin
    ), "incompatible exam and segmentation origins"


def load_birads(subject_dir, shape, subject_id_and_segmentation_name_to_birads):

    exam_basename = "A1.nrrd"
    exam_data, exam_header = medpy.io.load(os.path.join(subject_dir, "A1.nrrd"))

    segmentation_basenames = get_segmentation_basenames(subject_dir)

    raw_format = get_raw_format_version(subject_dir)

    birads = np.zeros(shape, dtype=int)
    subject_id = int(os.path.basename(subject_dir))

    if raw_format == "letters":
        for basename in segmentation_basenames:
            ek = basename.split(".")[0]
            segmentation_data, segmentation_header = medpy.io.load(
                os.path.join(subject_dir, basename)
            )
            check_segmentation_compatibility(
                exam_basename,
                exam_data,
                exam_header,
                basename,
                segmentation_data,
                segmentation_header,
            )
            segmentation_data, segmentation_header = fix_flips(
                segmentation_data, segmentation_header
            )
            segmentation_data = np.array(segmentation_data != 0, dtype=int)
            if basename == "B.nrrd" or basename == "D.nrrd":
                segmentation_data *= 3
            elif basename == "C.nrrd":
                segmentation_data *= 4
            elif (subject_id, ek) in subject_id_and_segmentation_name_to_birads:
                if subject_id_and_segmentation_name_to_birads[(subject_id, ek)] in [
                    3,
                    4,
                ]:
                    segmentation_data *= subject_id_and_segmentation_name_to_birads[
                        (subject_id, ek)
                    ]
                else:
                    segmentation_data *= 0
            else:
                raise ValueError(
                    f"segmentation basename not found on table.csv: {basename}"
                )

            birads = np.maximum(birads, segmentation_data)
    elif raw_format == "numbers":
        for basename in segmentation_basenames:
            segmentation_data, segmentation_header = medpy.io.load(
                os.path.join(subject_dir, basename)
            )
            check_segmentation_compatibility(
                exam_basename,
                exam_data,
                exam_header,
                basename,
                segmentation_data,
                segmentation_header,
            )
            segmentation_data, segmentation_header = fix_flips(
                segmentation_data, segmentation_header
            )
            segmentation_data = np.array(segmentation_data != 0, dtype=int)
            if basename == "3.nrrd":
                segmentation_data *= 3
            elif basename == "4.nrrd":
                segmentation_data *= 4
            else:
                raise ValueError(f"invalid segmentation filename: {basename}")
            assert (
                np.sum(segmentation_data * birads) == 0
            ), "3.nrrd and 4.nrrd are overlapping"
            birads = np.maximum(birads, segmentation_data)
        pass
    else:
        raise NotImplementedError("support for the new format is not implemented yet")

    return birads


def save_subject(exams, birads, subject_dir):
    subject_nrrd_output_dir = os.path.join(
        NRRD_OUTPUT_DIR,
        os.path.basename(os.path.dirname(subject_dir)),
        os.path.basename(subject_dir),
    )
    os.makedirs(subject_nrrd_output_dir, exist_ok=True)
    for i, e in enumerate(exams):
        medpy.io.save(e[0], os.path.join(subject_nrrd_output_dir, f"A{i+1}.nrrd"), e[1], use_compression=True)
    medpy.io.save(
        birads, os.path.join(subject_nrrd_output_dir, "birads.nrrd"), exams[0][1], use_compression=True
    )


def remove_subject(subject_dir):
    subject_nrrd_output_dir = os.path.join(
        NRRD_OUTPUT_DIR,
        os.path.basename(os.path.dirname(subject_dir)),
        os.path.basename(subject_dir),
    )

    if os.path.isdir(subject_nrrd_output_dir):
        shutil.rmtree(subject_nrrd_output_dir)


def fix_segmentation_translation(
    exam_data: np.ndarray,
    exam_header: OrderedDict,
    seg_data: np.ndarray,
    seg_header: OrderedDict,
) -> Tuple[np.ndarray, OrderedDict]:
    """Given that the headers are correct, return a new segment with correct
    dimensions and translation. This fix does not check for out-of-bounds
    errors."""

    matrix_exam = exam_header.direction.T
    exam_origin = exam_header.offset
    inv_matrix_exam = np.linalg.inv(matrix_exam)
    seg_origin = seg_header.offset

    # computes origin and endpoint in voxel exam space
    origin = np.rint((inv_matrix_exam @ (seg_origin - exam_origin))).astype(int)
    endpoint = origin + seg_data.shape

    new_seg_data = np.zeros(exam_data.shape)
    new_seg_data[
        origin[0] : endpoint[0], origin[1] : endpoint[1], origin[2] : endpoint[2]
    ] = seg_data

    return new_seg_data, exam_header


if __name__ == "__main__":
    main()
