#!/bin/bash


cd "$(dirname "$0")/.."
export PYTHONPATH=$(pwd)

echo "Running scripts/03_generate_ood.py ..."
python scripts/03_generate_ood.py --dataset "ImageNet100"  --fake_num_per_class 1000  --n_class 100 \
--k 400  --noise_scale 3 --temperature 0.5 --mean_group_size 200  --seed 42


echo "Press any key to continue..."


read -n 1 -s -r -p "" key

echo