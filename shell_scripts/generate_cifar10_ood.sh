#!/bin/bash


cd "$(dirname "$0")/.."
export PYTHONPATH=$(pwd)

echo "Running scripts/03_generate_ood.py ..."
python scripts/03_generate_ood.py --dataset "cifar10"  --fake_num_per_class 100  --n_class 10  \
--k 50  --feature_path "./output/01_extract_features/cifar10_features.pt" --noise_scale 1 \
--temperature 0.4 --mean_group_size 20 --size 32 --seed 42 --ip_adapter_scale 0.05 --num_inference_steps 4


echo "Press any key to continue..."


read -n 1 -s -r -p "" key


echo