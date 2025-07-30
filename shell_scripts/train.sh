#!/bin/bash

# 进入项目根目录
cd "$(dirname "$0")/.."
export PYTHONPATH=$(pwd)


python scripts/04_train.py --batch_size 256 --epochs 200  --dataset ImageNet100 --experiment_name ImageNet100