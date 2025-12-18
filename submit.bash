#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gres=gpu:2g.20gb:1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL

conda init
conda activate torch_env
python train.py