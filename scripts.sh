#!/bin/sh

#SBATCH --job-name=3dgs
#SBATCH --time=30:00
#SBATCH --gpus=titanrtx:1
#SBATCH --mail-user=cyang_09@u.nus.edu
#SBATCH --partition=standard

# # split data to different sizes
# python data_split.py

# # script to colmap just with images
# python convert.py -s /home/c/chenggan/datasets/tandt/train/100_percent

# python train.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00020 --densify_grad_threshold 0.00020 --eval

python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00020 --model_paths results/db/playroom/dgt_00050 results/db/playroom/dgt_00100 --combinedDebug --strategy fov
python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00020 --model_paths results/db/playroom/dgt_00050 results/db/playroom/dgt_00100 --strategy fov
