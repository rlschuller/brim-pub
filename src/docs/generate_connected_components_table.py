import argparse
import re

import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument("-p", "--path_to_nrrd_folder", type=str)

args = parser.parse_args()

OUTPUT_FOLDER = re.sub(r"/*$", "", args.path_to_nrrd_folder) + "-metrics"
JSONL_PATH = f"{OUTPUT_FOLDER}/all.jsonl"

df = pd.read_json(JSONL_PATH, lines=True)

df = df.drop(
    [
        "shape",
        "voxel_spacing",
        "voxel_volume",
        "voxels",
        "voxels_with_birads_3",
        "voxels_with_birads_4",
    ],
    axis=1,
)


df = df.rename(
    columns={
        "volume_with_birads_3": "BIRADS 3 (mL)",
        "volume_with_birads_4": "BIRADS 4 (mL)",
        "number_of_connected_components": "Components",
        "number_of_connected_components-after_dilation_of_1_voxel": "Components (dilation = 1)",
        "number_of_connected_components-after_dilation_of_2_voxels": "Components (dilation = 2)",
    }
)
df = df.set_index("id")
print(df)
table_html = df.to_html()

with open(OUTPUT_FOLDER + "/components_table.html", "w") as f:
    f.write(table_html)

df.to_excel(OUTPUT_FOLDER + "/components_table.xlsx")
