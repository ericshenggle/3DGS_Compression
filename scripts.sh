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

# python train.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00020 --densify_grad_threshold 0.00020 --eval

python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00020 --strategy dist --render_image
python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00021 --strategy dist --render_image
python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00022 --strategy dist --render_image
python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00023 --strategy dist --render_image
python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00024 --strategy dist --render_image
python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00025 --strategy dist --render_image
python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00026 --strategy dist --render_image
python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00027 --strategy dist --render_image
python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00028 --strategy dist --render_image
python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00029 --strategy dist --render_image
python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00030 --strategy dist --render_image
python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00035 --strategy dist --render_image
python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00040 --strategy dist --render_image
python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00045 --strategy dist --render_image
python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00050 --strategy dist --render_image
python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00060 --strategy dist --render_image
python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00070 --strategy dist --render_image
python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00080 --strategy dist --render_image
python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00090 --strategy dist --render_image
python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00100 --strategy dist --render_image
python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00110 --strategy dist --render_image
python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00120 --strategy dist --render_image
python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00130 --strategy dist --render_image
python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00140 --strategy dist --render_image
python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00150 --strategy dist --render_image
python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00160 --strategy dist --render_image
python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00170 --strategy dist --render_image
python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00180 --strategy dist --render_image
python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00190 --strategy dist --render_image
python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00200 --strategy dist --render_image
python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00250 --strategy dist --render_image
python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00300 --strategy dist --render_image
python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00350 --strategy dist --render_image
python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00400 --strategy dist --render_image
python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00500 --strategy dist --render_image
python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00600 --strategy dist --render_image
python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00700 --strategy dist --render_image
python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00800 --strategy dist --render_image
python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_00900 --strategy dist --render_image
python render_multiModel.py -s /home/c/chenggan/datasets/db/drjohnson -m results/db/drjohnson/dgt_01000 --strategy dist --render_image

