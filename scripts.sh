#!/bin/sh

#SBATCH --job-name=3dgs
#SBATCH --time=48:00:00
#SBATCH --gpus=titanrtx:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=cyang_09@u.nus.edu
#SBATCH --partition=long

# # split data to different sizes
# python data_split.py

# # script to colmap just with images
# python convert.py -s /home/c/chenggan/datasets/tandt/train/100_percent

# # script to train gaussian splatting for different compression with the same dataset(same colmap with different size of training dataset)
# python train.py -s /home/c/chenggan/datasets/apartment_25_non_gt -m results/apartment_25_non_gt/dgt_0002_random_03 --densify_grad_threshold 0.0002 --eval -p random

# # script to train gaussian splatting for different compression with the different dataset(not the same colmap with different size of training and testing dataset)
# python train.py -s /home/c/chenggan/datasets/tandt/train/ -m results/train/diff_colmap/dgt_0002 --densify_grad_threshold 0.0002 --eval --compression --diff_colmap
# python train.py -s /home/c/chenggan/datasets/tandt/train/ -m results/train/diff_colmap/dgt_0005 --densify_grad_threshold 0.0005 --eval --compression --diff_colmap
# python train.py -s /home/c/chenggan/datasets/tandt/train/ -m results/train/diff_colmap/dgt_0010 --densify_grad_threshold 0.0010 --eval --compression --diff_colmap
# python train.py -s /home/c/chenggan/datasets/tandt/train/ -m results/train/diff_colmap/dgt_0020 --densify_grad_threshold 0.0020 --eval --compression --diff_colmap
# python train.py -s /home/c/chenggan/datasets/tandt/train/ -m results/train/diff_colmap/dgt_0050 --densify_grad_threshold 0.0050 --eval --compression --diff_colmap
# python train.py -s /home/c/chenggan/datasets/tandt/train/ -m results/train/diff_colmap/dgt_0080 --densify_grad_threshold 0.0080 --eval --compression --diff_colmap
# python train.py -s /home/c/chenggan/datasets/tandt/train/ -m results/train/diff_colmap/dgt_0100 --densify_grad_threshold 0.0100 --eval --compression --diff_colmap

# # script to train gaussian splatting for different density threshold
# python train.py -s /home/c/chenggan/datasets/db/playroom -m results/playroom/dgt_00015 --densify_grad_threshold 0.00015 --eval
# python render.py -m results/playroom/dgt_00015 --skip_train
# python train.py -s /home/c/chenggan/datasets/db/playroom -m results/playroom/dgt_00020 --densify_grad_threshold 0.00020 --eval
# python render.py -m results/playroom/dgt_00020 --skip_train
# python train.py -s /home/c/chenggan/datasets/db/playroom -m results/playroom/dgt_00025 --densify_grad_threshold 0.00025 --eval
# python render.py -m results/playroom/dgt_00025 --skip_train
# python train.py -s /home/c/chenggan/datasets/db/playroom -m results/playroom/dgt_00030 --densify_grad_threshold 0.00030 --eval
# python render.py -m results/playroom/dgt_00030 --skip_train
# python train.py -s /home/c/chenggan/datasets/db/playroom -m results/playroom/dgt_00035 --densify_grad_threshold 0.00035 --eval
# python render.py -m results/playroom/dgt_00035 --skip_train
# python train.py -s /home/c/chenggan/datasets/db/playroom -m results/playroom/dgt_00040 --densify_grad_threshold 0.00040 --eval
# python render.py -m results/playroom/dgt_00040 --skip_train
# python train.py -s /home/c/chenggan/datasets/db/playroom -m results/playroom/dgt_00045 --densify_grad_threshold 0.00045 --eval
# python render.py -m results/playroom/dgt_00045 --skip_train
# python train.py -s /home/c/chenggan/datasets/db/playroom -m results/playroom/dgt_00050 --densify_grad_threshold 0.00050 --eval
# python render.py -m results/playroom/dgt_00050 --skip_train

# python convert.py -s /home/c/chenggan/datasets/apartment_25_non_gt --skip_matching
# python train.py -s /home/c/chenggan/datasets/apartment_25_non_gt -m results/apartment_25_non_gt/dgt_00020 --densify_grad_threshold 0.00020 --eval
# python train.py -s /home/c/chenggan/datasets/apartment_25_non_gt -m results/apartment_25_non_gt/dgt_00021 --densify_grad_threshold 0.00021 --eval
# python train.py -s /home/c/chenggan/datasets/apartment_25_non_gt -m results/apartment_25_non_gt/dgt_00022 --densify_grad_threshold 0.00022 --eval
# python train.py -s /home/c/chenggan/datasets/apartment_25_non_gt -m results/apartment_25_non_gt/dgt_00023 --densify_grad_threshold 0.00023 --eval
# python train.py -s /home/c/chenggan/datasets/apartment_25_non_gt -m results/apartment_25_non_gt/dgt_00024 --densify_grad_threshold 0.00024 --eval
# python train.py -s /home/c/chenggan/datasets/apartment_25_non_gt -m results/apartment_25_non_gt/dgt_00025 --densify_grad_threshold 0.00025 --eval
# python train.py -s /home/c/chenggan/datasets/apartment_25_non_gt -m results/apartment_25_non_gt/dgt_00026 --densify_grad_threshold 0.00026 --eval
# python train.py -s /home/c/chenggan/datasets/apartment_25_non_gt -m results/apartment_25_non_gt/dgt_00027 --densify_grad_threshold 0.00027 --eval
# python train.py -s /home/c/chenggan/datasets/apartment_25_non_gt -m results/apartment_25_non_gt/dgt_00028 --densify_grad_threshold 0.00028 --eval
# python train.py -s /home/c/chenggan/datasets/apartment_25_non_gt -m results/apartment_25_non_gt/dgt_00029 --densify_grad_threshold 0.00029 --eval
# python train.py -s /home/c/chenggan/datasets/apartment_25_non_gt -m results/apartment_25_non_gt/dgt_00030 --densify_grad_threshold 0.00030 --eval
# python train.py -s /home/c/chenggan/datasets/apartment_25_non_gt -m results/apartment_25_non_gt/dgt_00035 --densify_grad_threshold 0.00035 --eval
# python train.py -s /home/c/chenggan/datasets/apartment_25_non_gt -m results/apartment_25_non_gt/dgt_00040 --densify_grad_threshold 0.00040 --eval
# python train.py -s /home/c/chenggan/datasets/apartment_25_non_gt -m results/apartment_25_non_gt/dgt_00045 --densify_grad_threshold 0.00045 --eval
# python train.py -s /home/c/chenggan/datasets/apartment_25_non_gt -m results/apartment_25_non_gt/dgt_00050 --densify_grad_threshold 0.00050 --eval
# python train.py -s /home/c/chenggan/datasets/apartment_25_non_gt -m results/apartment_25_non_gt/dgt_00060 --densify_grad_threshold 0.00060 --eval
# python train.py -s /home/c/chenggan/datasets/apartment_25_non_gt -m results/apartment_25_non_gt/dgt_00070 --densify_grad_threshold 0.00070 --eval
# python train.py -s /home/c/chenggan/datasets/apartment_25_non_gt -m results/apartment_25_non_gt/dgt_00080 --densify_grad_threshold 0.00080 --eval
# python train.py -s /home/c/chenggan/datasets/apartment_25_non_gt -m results/apartment_25_non_gt/dgt_00090 --densify_grad_threshold 0.00090 --eval
# python train.py -s /home/c/chenggan/datasets/apartment_25_non_gt -m results/apartment_25_non_gt/dgt_00100 --densify_grad_threshold 0.00100 --eval
# python train.py -s /home/c/chenggan/datasets/apartment_25_non_gt -m results/apartment_25_non_gt/dgt_00110 --densify_grad_threshold 0.00110 --eval
# python train.py -s /home/c/chenggan/datasets/apartment_25_non_gt -m results/apartment_25_non_gt/dgt_00120 --densify_grad_threshold 0.00120 --eval
# python train.py -s /home/c/chenggan/datasets/apartment_25_non_gt -m results/apartment_25_non_gt/dgt_00130 --densify_grad_threshold 0.00130 --eval
# python train.py -s /home/c/chenggan/datasets/apartment_25_non_gt -m results/apartment_25_non_gt/dgt_00140 --densify_grad_threshold 0.00140 --eval
# python train.py -s /home/c/chenggan/datasets/apartment_25_non_gt -m results/apartment_25_non_gt/dgt_00150 --densify_grad_threshold 0.00150 --eval
# python train.py -s /home/c/chenggan/datasets/apartment_25_non_gt -m results/apartment_25_non_gt/dgt_00160 --densify_grad_threshold 0.00160 --eval
# python train.py -s /home/c/chenggan/datasets/apartment_25_non_gt -m results/apartment_25_non_gt/dgt_00170 --densify_grad_threshold 0.00170 --eval
# python train.py -s /home/c/chenggan/datasets/apartment_25_non_gt -m results/apartment_25_non_gt/dgt_00180 --densify_grad_threshold 0.00180 --eval
# python train.py -s /home/c/chenggan/datasets/apartment_25_non_gt -m results/apartment_25_non_gt/dgt_00190 --densify_grad_threshold 0.00190 --eval
# python train.py -s /home/c/chenggan/datasets/apartment_25_non_gt -m results/apartment_25_non_gt/dgt_00200 --densify_grad_threshold 0.00200 --eval
# python train.py -s /home/c/chenggan/datasets/apartment_25_non_gt -m results/apartment_25_non_gt/dgt_00250 --densify_grad_threshold 0.00250 --eval
# python train.py -s /home/c/chenggan/datasets/apartment_25_non_gt -m results/apartment_25_non_gt/dgt_00300 --densify_grad_threshold 0.00300 --eval
# python train.py -s /home/c/chenggan/datasets/apartment_25_non_gt -m results/apartment_25_non_gt/dgt_00350 --densify_grad_threshold 0.00350 --eval
# python train.py -s /home/c/chenggan/datasets/apartment_25_non_gt -m results/apartment_25_non_gt/dgt_00400 --densify_grad_threshold 0.00400 --eval
# python train.py -s /home/c/chenggan/datasets/apartment_25_non_gt -m results/apartment_25_non_gt/dgt_00500 --densify_grad_threshold 0.00500 --eval
# python train.py -s /home/c/chenggan/datasets/apartment_25_non_gt -m results/apartment_25_non_gt/dgt_00600 --densify_grad_threshold 0.00600 --eval
# python train.py -s /home/c/chenggan/datasets/apartment_25_non_gt -m results/apartment_25_non_gt/dgt_00700 --densify_grad_threshold 0.00700 --eval
# python train.py -s /home/c/chenggan/datasets/apartment_25_non_gt -m results/apartment_25_non_gt/dgt_00800 --densify_grad_threshold 0.00800 --eval
# python train.py -s /home/c/chenggan/datasets/apartment_25_non_gt -m results/apartment_25_non_gt/dgt_00900 --densify_grad_threshold 0.00900 --eval
# python train.py -s /home/c/chenggan/datasets/apartment_25_non_gt -m results/apartment_25_non_gt/dgt_01000 --densify_grad_threshold 0.01000 --eval

python train.py -s /home/c/chenggan/datasets/apartment_25_gt -m results/apartment_25_gt/gt_Pose_random_Ply --eval