#!/bin/bash

# 进入项目根目录
cd "$(dirname "$0")/.."
export PYTHONPATH=$(pwd)

echo "Running scripts/01_extract_features.py..."
python scripts/01_extract_features --dataset CIFAR10 --save_file cifar10_features


echo "Press any key to continue..."


read -n 1 -s -r -p "" key

echo