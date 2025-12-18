#!/usr/bin/env bash

datasets=(
    data/processed/datasets/8089-harlequin_lamprey.json
)

# use all of the available metrics
metrics=(
    $( python src/models/threshold/predict.py -l )
)

quantiles=( $( seq 0.01 0.01 0.99 ) )

greater=(
    '-gt'
    ''
)


for dataset in "${datasets[@]}" ; do
    dataset_name="$( basename "$dataset" | sed 's/\..*//g' )"
    echo "removing $dataset_name..."
    rm -r "models/threshold/$dataset_name" || echo "$dataset_name not found, ignoring..."
    echo
done

echo "total number of jobs: $(( ${#datasets[@]} * ${#metrics[@]} * ${#quantiles[@]} * ${#greater[@]} ))"
parallel --progress --halt-on-error 2 'eval python src/models/threshold/predict.py -d {1} -m {2} -q {3} {4} &> /dev/null' ::: \
    "${datasets[@]}" ::: \
    "${metrics[@]}" ::: \
    "${quantiles[@]}" ::: \
    "${greater[@]}"
