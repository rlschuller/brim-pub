#!/bin/bash

set -x
set -e

export PYTHONPATH=.

`poetry env activate`

python src/data/process.py

mkdir -p data/processed/hdf5/

# Convert NRRD files to HDF format
echo "Converting NRRD files to HDF format..."
python src/data/save_HDF.py nrrd-to-hdf \
    --filename raw \
    --folderpath 'data/processed/nrrd/*/*'


# Process the raw data to create a processed dataset
echo "Processing raw data to create a processed dataset..."
python src/data/save_HDF.py save-exams \
    --filename processed \
    --resize \
    --fix_flip \
    --normalize \
    --dataset raw

mkdir -p data/processed/pre_segmentation_data/

# Generate Segmentation masks from the processed dataset
python src/models/segmentation/generate_dataset.py \
    --data-dir data/processed/pre_segmentation_data/ \
    --hdf-file data/processed/hdf5/processed.hdf5


# Exit the virtual environment if it was activated
echo "Deactivating any existing virtual environment..."
deactivate || true


# Clone pre-trained segmentation model's repository if not already cloned
echo "Cloning the pre-trained segmentation model repository..."
git clone https://github.com/mazurowski-lab/3D-Breast-FGT-and-Blood-Vessel-Segmentation.git lib/3D-Breast-FGT-and-Blood-Vessel-Segmentation || true

# Create python virtual environment if it doesn't exist
echo "Creating a Python virtual environment..."
python3 -m venv lib/3D-Breast-FGT-and-Blood-Vessel-Segmentation/venv || true

# Activate the virtual environment
echo "Activating the virtual environment..."
source lib/3D-Breast-FGT-and-Blood-Vessel-Segmentation/venv/bin/activate

# Install required dependencies for the segmentation model
echo "Installing required dependencies..."
pip install -r lib/3D-Breast-FGT-and-Blood-Vessel-Segmentation/requirements.txt

# Ensure the directory for pre-segmentation masks exists
echo "Creating directory for pre-segmentation masks..."
mkdir -p data/processed/pre_segmentation_masks

# Fix bug in the segmentation model
sed -i "/mask = batch\['mask'\]/d" lib/3D-Breast-FGT-and-Blood-Vessel-Segmentation/model_utils.py
sed -i "/mask = mask\.to(device, dtype=torch\.float32)/d" lib/3D-Breast-FGT-and-Blood-Vessel-Segmentation/model_utils.py

# Run the pre-trained segmentation model
echo "Running the pre-trained segmentation model..."
python3 lib/3D-Breast-FGT-and-Blood-Vessel-Segmentation/predict.py \
    --target-tissue breast \
    --image data/processed/pre_segmentation_data/ \
    --model-save-path lib/3D-Breast-FGT-and-Blood-Vessel-Segmentation/trained_models/breast_model.pth \
    --save-masks-dir data/processed/pre_segmentation_masks \

# Exit the virtual environment
echo "Deactivating the virtual environment..."
deactivate

# Activate the virtual environment for post-processing
`poetry env activate`

# Post processing of segmentation masks
echo "Post processing of segmentation masks..."
python src/models/segmentation/save_seg.py \
    --output-path data/processed/hdf5/breast_segmentation_new.hdf5 \
    --masks-dir data/processed/pre_segmentation_masks \
    --exam-dir data/processed/pre_segmentation_data

# Split exam into cubes
echo "Splitting exam into cubes..."
python src/data/save_HDF.py save-cubes \
    --size 32 \
    --stride 16 \
    --filename data/processed/hdf5/cubes32stride16.hdf5 \
    --dataset data/processed/hdf5/processed.hdf5 \
    --segmentation-path data/processed/hdf5/breast_segmentation_new.hdf5

echo "Training model..."
python src/models/DLPT/train.py
