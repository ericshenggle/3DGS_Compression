#!/bin/sh

#SBATCH --job-name=3dgs
#SBATCH --time=3:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=cyang_09@u.nus.edu
#SBATCH --partition=normal

# sbatch --gres=gpu:nv:1 -C cuda75 scripts.sh


# python train.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00020 --densify_grad_threshold 0.00020 --eval

# python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00020 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00021 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00022 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00023 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00024 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00025 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00026 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00027 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00028 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00029 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00030 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00035 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00040 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00045 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00050 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00060 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00070 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00080 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00090 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00100 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00110 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00120 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00130 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00140 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00150 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00160 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00170 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00180 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00190 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00200 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00250 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00300 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00350 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00400 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00500 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00600 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00700 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00800 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00900 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_01000 --strategy fov --render_image


# python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00025 --model_paths results/db/drjohnson/dgt_00100 results/db/drjohnson/dgt_00200 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00025 --model_paths results/db/drjohnson/dgt_00090 results/db/drjohnson/dgt_00190 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00025 --model_paths results/db/drjohnson/dgt_00080 results/db/drjohnson/dgt_00180 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00025 --model_paths results/db/drjohnson/dgt_00070 results/db/drjohnson/dgt_00170 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00025 --model_paths results/db/drjohnson/dgt_00060 results/db/drjohnson/dgt_00160 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00025 --model_paths results/db/drjohnson/dgt_00100 results/db/drjohnson/dgt_00150 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00025 --model_paths results/db/drjohnson/dgt_00090 results/db/drjohnson/dgt_00140 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00025 --model_paths results/db/drjohnson/dgt_00080 results/db/drjohnson/dgt_00130 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00025 --model_paths results/db/drjohnson/dgt_00070 results/db/drjohnson/dgt_00120 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00025 --model_paths results/db/drjohnson/dgt_00060 results/db/drjohnson/dgt_00110 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00030 --model_paths results/db/drjohnson/dgt_00100 results/db/drjohnson/dgt_00200 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00030 --model_paths results/db/drjohnson/dgt_00090 results/db/drjohnson/dgt_00190 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00030 --model_paths results/db/drjohnson/dgt_00080 results/db/drjohnson/dgt_00180 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00030 --model_paths results/db/drjohnson/dgt_00070 results/db/drjohnson/dgt_00170 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00030 --model_paths results/db/drjohnson/dgt_00060 results/db/drjohnson/dgt_00160 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00030 --model_paths results/db/drjohnson/dgt_00100 results/db/drjohnson/dgt_00150 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00030 --model_paths results/db/drjohnson/dgt_00090 results/db/drjohnson/dgt_00140 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00030 --model_paths results/db/drjohnson/dgt_00080 results/db/drjohnson/dgt_00130 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00030 --model_paths results/db/drjohnson/dgt_00070 results/db/drjohnson/dgt_00120 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00030 --model_paths results/db/drjohnson/dgt_00060 results/db/drjohnson/dgt_00110 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00035 --model_paths results/db/drjohnson/dgt_00100 results/db/drjohnson/dgt_00200 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00035 --model_paths results/db/drjohnson/dgt_00090 results/db/drjohnson/dgt_00190 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00035 --model_paths results/db/drjohnson/dgt_00080 results/db/drjohnson/dgt_00180 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00035 --model_paths results/db/drjohnson/dgt_00070 results/db/drjohnson/dgt_00170 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00035 --model_paths results/db/drjohnson/dgt_00060 results/db/drjohnson/dgt_00160 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00035 --model_paths results/db/drjohnson/dgt_00100 results/db/drjohnson/dgt_00150 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00035 --model_paths results/db/drjohnson/dgt_00090 results/db/drjohnson/dgt_00140 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00035 --model_paths results/db/drjohnson/dgt_00080 results/db/drjohnson/dgt_00130 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00035 --model_paths results/db/drjohnson/dgt_00070 results/db/drjohnson/dgt_00120 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00035 --model_paths results/db/drjohnson/dgt_00060 results/db/drjohnson/dgt_00110 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00060 --model_paths results/db/drjohnson/dgt_00150 results/db/drjohnson/dgt_00200 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00060 --model_paths results/db/drjohnson/dgt_00140 results/db/drjohnson/dgt_00190 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00060 --model_paths results/db/drjohnson/dgt_00130 results/db/drjohnson/dgt_00180 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00060 --model_paths results/db/drjohnson/dgt_00120 results/db/drjohnson/dgt_00170 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00060 --model_paths results/db/drjohnson/dgt_00110 results/db/drjohnson/dgt_00160 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00060 --model_paths results/db/drjohnson/dgt_00100 results/db/drjohnson/dgt_00150 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00060 --model_paths results/db/drjohnson/dgt_00090 results/db/drjohnson/dgt_00140 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00060 --model_paths results/db/drjohnson/dgt_00080 results/db/drjohnson/dgt_00130 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00060 --model_paths results/db/drjohnson/dgt_00070 results/db/drjohnson/dgt_00120 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00060 --model_paths results/db/drjohnson/dgt_00070 results/db/drjohnson/dgt_00110 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00070 --model_paths results/db/drjohnson/dgt_00160 results/db/drjohnson/dgt_00200 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00070 --model_paths results/db/drjohnson/dgt_00150 results/db/drjohnson/dgt_00190 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00070 --model_paths results/db/drjohnson/dgt_00140 results/db/drjohnson/dgt_00180 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00070 --model_paths results/db/drjohnson/dgt_00130 results/db/drjohnson/dgt_00170 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00070 --model_paths results/db/drjohnson/dgt_00120 results/db/drjohnson/dgt_00160 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00070 --model_paths results/db/drjohnson/dgt_00110 results/db/drjohnson/dgt_00150 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00070 --model_paths results/db/drjohnson/dgt_00100 results/db/drjohnson/dgt_00140 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00070 --model_paths results/db/drjohnson/dgt_00090 results/db/drjohnson/dgt_00130 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00070 --model_paths results/db/drjohnson/dgt_00080 results/db/drjohnson/dgt_00120 --strategy fov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00070 --model_paths results/db/drjohnson/dgt_00080 results/db/drjohnson/dgt_00110 --strategy fov --render_image

python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00025 --model_paths results/db/drjohnson/dgt_00100 results/db/drjohnson/dgt_00200 --strategy distFov --render_image