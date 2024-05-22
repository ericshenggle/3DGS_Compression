#!/bin/sh

#SBATCH --job-name=3dgs
#SBATCH --time=24:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=cyang_09@u.nus.edu
#SBATCH --partition=long

# # split data to different sizes
# python data_split.py

# # script to colmap just with images
# python convert.py -s /home/c/chenggan/datasets/tandt/train/100_percent

# python train.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00020 --densify_grad_threshold 0.00020 --eval

# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00020 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00021 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00022 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00023 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00024 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00025 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00026 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00027 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00028 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00029 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00030 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00035 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00040 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00045 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00050 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00060 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00070 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00080 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00090 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00100 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00110 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00120 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00130 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00140 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00150 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00160 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00170 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00180 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00190 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00200 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00250 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00300 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00350 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00400 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00500 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00600 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00700 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00800 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00900 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_01000 --strategy fov --render_image


# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00025 --model_paths results/db/playroom/dgt_00100 results/db/playroom/dgt_00200 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00025 --model_paths results/db/playroom/dgt_00090 results/db/playroom/dgt_00190 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00025 --model_paths results/db/playroom/dgt_00080 results/db/playroom/dgt_00180 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00025 --model_paths results/db/playroom/dgt_00070 results/db/playroom/dgt_00170 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00025 --model_paths results/db/playroom/dgt_00060 results/db/playroom/dgt_00160 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00025 --model_paths results/db/playroom/dgt_00100 results/db/playroom/dgt_00150 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00025 --model_paths results/db/playroom/dgt_00090 results/db/playroom/dgt_00140 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00025 --model_paths results/db/playroom/dgt_00080 results/db/playroom/dgt_00130 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00025 --model_paths results/db/playroom/dgt_00070 results/db/playroom/dgt_00120 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00025 --model_paths results/db/playroom/dgt_00060 results/db/playroom/dgt_00110 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00030 --model_paths results/db/playroom/dgt_00100 results/db/playroom/dgt_00200 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00030 --model_paths results/db/playroom/dgt_00090 results/db/playroom/dgt_00190 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00030 --model_paths results/db/playroom/dgt_00080 results/db/playroom/dgt_00180 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00030 --model_paths results/db/playroom/dgt_00070 results/db/playroom/dgt_00170 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00030 --model_paths results/db/playroom/dgt_00060 results/db/playroom/dgt_00160 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00030 --model_paths results/db/playroom/dgt_00100 results/db/playroom/dgt_00150 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00030 --model_paths results/db/playroom/dgt_00090 results/db/playroom/dgt_00140 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00030 --model_paths results/db/playroom/dgt_00080 results/db/playroom/dgt_00130 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00030 --model_paths results/db/playroom/dgt_00070 results/db/playroom/dgt_00120 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00030 --model_paths results/db/playroom/dgt_00060 results/db/playroom/dgt_00110 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00035 --model_paths results/db/playroom/dgt_00100 results/db/playroom/dgt_00200 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00035 --model_paths results/db/playroom/dgt_00090 results/db/playroom/dgt_00190 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00035 --model_paths results/db/playroom/dgt_00080 results/db/playroom/dgt_00180 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00035 --model_paths results/db/playroom/dgt_00070 results/db/playroom/dgt_00170 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00035 --model_paths results/db/playroom/dgt_00060 results/db/playroom/dgt_00160 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00035 --model_paths results/db/playroom/dgt_00100 results/db/playroom/dgt_00150 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00035 --model_paths results/db/playroom/dgt_00090 results/db/playroom/dgt_00140 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00035 --model_paths results/db/playroom/dgt_00080 results/db/playroom/dgt_00130 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00035 --model_paths results/db/playroom/dgt_00070 results/db/playroom/dgt_00120 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00035 --model_paths results/db/playroom/dgt_00060 results/db/playroom/dgt_00110 --strategy fov --render_image
