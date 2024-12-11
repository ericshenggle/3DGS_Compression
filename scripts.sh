#!/bin/sh

#SBATCH --job-name=3dgs
#SBATCH --time=120:00:00
#SBATCH --partition=gpu-long
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cyang_09@u.nus.edu

# sbatch --gres=gpu:nv:1 -C cuda75 scripts.sh
# sbatch -w xcnf27 scripts.sh

python line3d.py -s /home/c/chenggan/datasets/room_res1 -m /home/c/chenggan/gaussian-splatting/results/room_res1 --baseline 2

#python train.py -s /home/c/chenggan/datasets/tandt/truck -m /home/c/chenggan/gaussian-splatting/results/tandt/truck_combined/dgt_00020 --densify_grad_threshold 0.00020 --eval
##python train.py -s /home/c/chenggan/datasets/tandt/truck -m /home/c/chenggan/gaussian-splatting/results/tandt/truck_combined/dgt_00021 --densify_grad_threshold 0.00021 --eval
##python train.py -s /home/c/chenggan/datasets/tandt/truck -m /home/c/chenggan/gaussian-splatting/results/tandt/truck_combined/dgt_00022 --densify_grad_threshold 0.00022 --eval
#python train.py -s /home/c/chenggan/datasets/tandt/truck -m /home/c/chenggan/gaussian-splatting/results/tandt/truck_combined/dgt_00023 --densify_grad_threshold 0.00023 --eval
##python train.py -s /home/c/chenggan/datasets/tandt/truck -m /home/c/chenggan/gaussian-splatting/results/tandt/truck_combined/dgt_00024 --densify_grad_threshold 0.00024 --eval
##python train.py -s /home/c/chenggan/datasets/tandt/truck -m /home/c/chenggan/gaussian-splatting/results/tandt/truck_combined/dgt_00025 --densify_grad_threshold 0.00025 --eval
#python train.py -s /home/c/chenggan/datasets/tandt/truck -m /home/c/chenggan/gaussian-splatting/results/tandt/truck_combined/dgt_00026 --densify_grad_threshold 0.00026 --eval
##python train.py -s /home/c/chenggan/datasets/tandt/truck -m /home/c/chenggan/gaussian-splatting/results/tandt/truck_combined/dgt_00027 --densify_grad_threshold 0.00027 --eval
##python train.py -s /home/c/chenggan/datasets/tandt/truck -m /home/c/chenggan/gaussian-splatting/results/tandt/truck_combined/dgt_00028 --densify_grad_threshold 0.00028 --eval
#python train.py -s /home/c/chenggan/datasets/tandt/truck -m /home/c/chenggan/gaussian-splatting/results/tandt/truck_combined/dgt_00029 --densify_grad_threshold 0.00029 --eval
##python train.py -s /home/c/chenggan/datasets/tandt/truck -m /home/c/chenggan/gaussian-splatting/results/tandt/truck_combined/dgt_00030 --densify_grad_threshold 0.00030 --eval
##python train.py -s /home/c/chenggan/datasets/tandt/truck -m /home/c/chenggan/gaussian-splatting/results/tandt/truck_combined/dgt_00035 --densify_grad_threshold 0.00035 --eval
#python train.py -s /home/c/chenggan/datasets/tandt/truck -m /home/c/chenggan/gaussian-splatting/results/tandt/truck_combined/dgt_00040 --densify_grad_threshold 0.00040 --eval
##python train.py -s /home/c/chenggan/datasets/tandt/truck -m /home/c/chenggan/gaussian-splatting/results/tandt/truck_combined/dgt_00045 --densify_grad_threshold 0.00045 --eval
##python train.py -s /home/c/chenggan/datasets/tandt/truck -m /home/c/chenggan/gaussian-splatting/results/tandt/truck_combined/dgt_00050 --densify_grad_threshold 0.00050 --eval
#python train.py -s /home/c/chenggan/datasets/tandt/truck -m /home/c/chenggan/gaussian-splatting/results/tandt/truck_combined/dgt_00060 --densify_grad_threshold 0.00060 --eval
##python train.py -s /home/c/chenggan/datasets/tandt/truck -m /home/c/chenggan/gaussian-splatting/results/tandt/truck_combined/dgt_00070 --densify_grad_threshold 0.00070 --eval
##python train.py -s /home/c/chenggan/datasets/tandt/truck -m /home/c/chenggan/gaussian-splatting/results/tandt/truck_combined/dgt_00080 --densify_grad_threshold 0.00080 --eval
#python train.py -s /home/c/chenggan/datasets/tandt/truck -m /home/c/chenggan/gaussian-splatting/results/tandt/truck_combined/dgt_00090 --densify_grad_threshold 0.00090 --eval
##python train.py -s /home/c/chenggan/datasets/tandt/truck -m /home/c/chenggan/gaussian-splatting/results/tandt/truck_combined/dgt_00100 --densify_grad_threshold 0.00100 --eval
##python train.py -s /home/c/chenggan/datasets/tandt/truck -m /home/c/chenggan/gaussian-splatting/results/tandt/truck_combined/dgt_00110 --densify_grad_threshold 0.00110 --eval
#python train.py -s /home/c/chenggan/datasets/tandt/truck -m /home/c/chenggan/gaussian-splatting/results/tandt/truck_combined/dgt_00120 --densify_grad_threshold 0.00120 --eval
##python train.py -s /home/c/chenggan/datasets/tandt/truck -m /home/c/chenggan/gaussian-splatting/results/tandt/truck_combined/dgt_00130 --densify_grad_threshold 0.00130 --eval
##python train.py -s /home/c/chenggan/datasets/tandt/truck -m /home/c/chenggan/gaussian-splatting/results/tandt/truck_combined/dgt_00140 --densify_grad_threshold 0.00140 --eval
#python train.py -s /home/c/chenggan/datasets/tandt/truck -m /home/c/chenggan/gaussian-splatting/results/tandt/truck_combined/dgt_00150 --densify_grad_threshold 0.00150 --eval
##python train.py -s /home/c/chenggan/datasets/tandt/truck -m /home/c/chenggan/gaussian-splatting/results/tandt/truck_combined/dgt_00160 --densify_grad_threshold 0.00160 --eval
##python train.py -s /home/c/chenggan/datasets/tandt/truck -m /home/c/chenggan/gaussian-splatting/results/tandt/truck_combined/dgt_00170 --densify_grad_threshold 0.00170 --eval
#python train.py -s /home/c/chenggan/datasets/tandt/truck -m /home/c/chenggan/gaussian-splatting/results/tandt/truck_combined/dgt_00180 --densify_grad_threshold 0.00180 --eval
##python train.py -s /home/c/chenggan/datasets/tandt/truck -m /home/c/chenggan/gaussian-splatting/results/tandt/truck_combined/dgt_00190 --densify_grad_threshold 0.00190 --eval
##python train.py -s /home/c/chenggan/datasets/tandt/truck -m /home/c/chenggan/gaussian-splatting/results/tandt/truck_combined/dgt_00200 --densify_grad_threshold 0.00200 --eval
#python train.py -s /home/c/chenggan/datasets/tandt/truck -m /home/c/chenggan/gaussian-splatting/results/tandt/truck_combined/dgt_00250 --densify_grad_threshold 0.00250 --eval
##python train.py -s /home/c/chenggan/datasets/tandt/truck -m /home/c/chenggan/gaussian-splatting/results/tandt/truck_combined/dgt_00300 --densify_grad_threshold 0.00300 --eval
##python train.py -s /home/c/chenggan/datasets/tandt/truck -m /home/c/chenggan/gaussian-splatting/results/tandt/truck_combined/dgt_00350 --densify_grad_threshold 0.00350 --eval
#python train.py -s /home/c/chenggan/datasets/tandt/truck -m /home/c/chenggan/gaussian-splatting/results/tandt/truck_combined/dgt_00400 --densify_grad_threshold 0.00400 --eval
##python train.py -s /home/c/chenggan/datasets/tandt/truck -m /home/c/chenggan/gaussian-splatting/results/tandt/truck_combined/dgt_00500 --densify_grad_threshold 0.00500 --eval
##python train.py -s /home/c/chenggan/datasets/tandt/truck -m /home/c/chenggan/gaussian-splatting/results/tandt/truck_combined/dgt_00600 --densify_grad_threshold 0.00600 --eval
#python train.py -s /home/c/chenggan/datasets/tandt/truck -m /home/c/chenggan/gaussian-splatting/results/tandt/truck_combined/dgt_00700 --densify_grad_threshold 0.00700 --eval
##python train.py -s /home/c/chenggan/datasets/tandt/truck -m /home/c/chenggan/gaussian-splatting/results/tandt/truck_combined/dgt_00800 --densify_grad_threshold 0.00800 --eval
##python train.py -s /home/c/chenggan/datasets/tandt/truck -m /home/c/chenggan/gaussian-splatting/results/tandt/truck_combined/dgt_00900 --densify_grad_threshold 0.00900 --eval
#python train.py -s /home/c/chenggan/datasets/tandt/truck -m /home/c/chenggan/gaussian-splatting/results/tandt/truck_combined/dgt_01000 --densify_grad_threshold 0.01000 --eval
#
#python render_multiModel.py -s /home/c/chenggan/datasets/tandt/truck -m results/tandt/truck_combined/dgt_00020 --strategy dist --render_image
##python render_multiModel.py -s /home/c/chenggan/datasets/tandt/truck -m results/tandt/truck_combined/dgt_00021 --strategy dist --render_image
##python render_multiModel.py -s /home/c/chenggan/datasets/tandt/truck -m results/tandt/truck_combined/dgt_00022 --strategy dist --render_image
#python render_multiModel.py -s /home/c/chenggan/datasets/tandt/truck -m results/tandt/truck_combined/dgt_00023 --strategy dist --render_image
##python render_multiModel.py -s /home/c/chenggan/datasets/tandt/truck -m results/tandt/truck_combined/dgt_00024 --strategy dist --render_image
##python render_multiModel.py -s /home/c/chenggan/datasets/tandt/truck -m results/tandt/truck_combined/dgt_00025 --strategy dist --render_image
#python render_multiModel.py -s /home/c/chenggan/datasets/tandt/truck -m results/tandt/truck_combined/dgt_00026 --strategy dist --render_image
##python render_multiModel.py -s /home/c/chenggan/datasets/tandt/truck -m results/tandt/truck_combined/dgt_00027 --strategy dist --render_image
##python render_multiModel.py -s /home/c/chenggan/datasets/tandt/truck -m results/tandt/truck_combined/dgt_00028 --strategy dist --render_image
#python render_multiModel.py -s /home/c/chenggan/datasets/tandt/truck -m results/tandt/truck_combined/dgt_00029 --strategy dist --render_image
##python render_multiModel.py -s /home/c/chenggan/datasets/tandt/truck -m results/tandt/truck_combined/dgt_00030 --strategy dist --render_image
##python render_multiModel.py -s /home/c/chenggan/datasets/tandt/truck -m results/tandt/truck_combined/dgt_00035 --strategy dist --render_image
#python render_multiModel.py -s /home/c/chenggan/datasets/tandt/truck -m results/tandt/truck_combined/dgt_00040 --strategy dist --render_image
##python render_multiModel.py -s /home/c/chenggan/datasets/tandt/truck -m results/tandt/truck_combined/dgt_00045 --strategy dist --render_image
##python render_multiModel.py -s /home/c/chenggan/datasets/tandt/truck -m results/tandt/truck_combined/dgt_00050 --strategy dist --render_image
#python render_multiModel.py -s /home/c/chenggan/datasets/tandt/truck -m results/tandt/truck_combined/dgt_00060 --strategy dist --render_image
##python render_multiModel.py -s /home/c/chenggan/datasets/tandt/truck -m results/tandt/truck_combined/dgt_00070 --strategy dist --render_image
##python render_multiModel.py -s /home/c/chenggan/datasets/tandt/truck -m results/tandt/truck_combined/dgt_00080 --strategy dist --render_image
#python render_multiModel.py -s /home/c/chenggan/datasets/tandt/truck -m results/tandt/truck_combined/dgt_00090 --strategy dist --render_image
##python render_multiModel.py -s /home/c/chenggan/datasets/tandt/truck -m results/tandt/truck_combined/dgt_00100 --strategy dist --render_image
##python render_multiModel.py -s /home/c/chenggan/datasets/tandt/truck -m results/tandt/truck_combined/dgt_00110 --strategy dist --render_image
#python render_multiModel.py -s /home/c/chenggan/datasets/tandt/truck -m results/tandt/truck_combined/dgt_00120 --strategy dist --render_image
##python render_multiModel.py -s /home/c/chenggan/datasets/tandt/truck -m results/tandt/truck_combined/dgt_00130 --strategy dist --render_image
##python render_multiModel.py -s /home/c/chenggan/datasets/tandt/truck -m results/tandt/truck_combined/dgt_00140 --strategy dist --render_image
#python render_multiModel.py -s /home/c/chenggan/datasets/tandt/truck -m results/tandt/truck_combined/dgt_00150 --strategy dist --render_image
##python render_multiModel.py -s /home/c/chenggan/datasets/tandt/truck -m results/tandt/truck_combined/dgt_00160 --strategy dist --render_image
##python render_multiModel.py -s /home/c/chenggan/datasets/tandt/truck -m results/tandt/truck_combined/dgt_00170 --strategy dist --render_image
#python render_multiModel.py -s /home/c/chenggan/datasets/tandt/truck -m results/tandt/truck_combined/dgt_00180 --strategy dist --render_image
##python render_multiModel.py -s /home/c/chenggan/datasets/tandt/truck -m results/tandt/truck_combined/dgt_00190 --strategy dist --render_image
##python render_multiModel.py -s /home/c/chenggan/datasets/tandt/truck -m results/tandt/truck_combined/dgt_00200 --strategy dist --render_image
#python render_multiModel.py -s /home/c/chenggan/datasets/tandt/truck -m results/tandt/truck_combined/dgt_00250 --strategy dist --render_image
##python render_multiModel.py -s /home/c/chenggan/datasets/tandt/truck -m results/tandt/truck_combined/dgt_00300 --strategy dist --render_image
##python render_multiModel.py -s /home/c/chenggan/datasets/tandt/truck -m results/tandt/truck_combined/dgt_00350 --strategy dist --render_image
#python render_multiModel.py -s /home/c/chenggan/datasets/tandt/truck -m results/tandt/truck_combined/dgt_00400 --strategy dist --render_image
##python render_multiModel.py -s /home/c/chenggan/datasets/tandt/truck -m results/tandt/truck_combined/dgt_00500 --strategy dist --render_image
##python render_multiModel.py -s /home/c/chenggan/datasets/tandt/truck -m results/tandt/truck_combined/dgt_00600 --strategy dist --render_image
#python render_multiModel.py -s /home/c/chenggan/datasets/tandt/truck -m results/tandt/truck_combined/dgt_00700 --strategy dist --render_image
##python render_multiModel.py -s /home/c/chenggan/datasets/tandt/truck -m results/tandt/truck_combined/dgt_00800 --strategy dist --render_image
##python render_multiModel.py -s /home/c/chenggan/datasets/tandt/truck -m results/tandt/truck_combined/dgt_00900 --strategy dist --render_image
#python render_multiModel.py -s /home/c/chenggan/datasets/tandt/truck -m results/tandt/truck_combined/dgt_01000 --strategy dist --render_image

#python line3d.py -s /home/c/chenggan/datasets/ABC-NEF/ -m /home/c/chenggan/gaussian-splatting/results/ABC-NEF/ --baseline 3
#python line3d.py -s /home/c/chenggan/datasets/ABC-NEF/00000146 -m /home/c/chenggan/gaussian-splatting/results/ABC-NEF/00000146 --baseline 2
#python line3d.py -s /home/c/chenggan/datasets/ABC-NEF/00000699 -m /home/c/chenggan/gaussian-splatting/results/ABC-NEF/00000699 --baseline 2
#python line3d.py -s /home/c/chenggan/datasets/ABC-NEF/00000006 -m /home/c/chenggan/gaussian-splatting/results/ABC-NEF/00000006 --baseline 2
#python line3d.py -s /home/c/chenggan/datasets/ABC-NEF/00003823 -m /home/c/chenggan/gaussian-splatting/results/ABC-NEF/00003823 --baseline 2
#python line3d.py -s /home/c/chenggan/datasets/ABC-NEF/00004383 -m /home/c/chenggan/gaussian-splatting/results/ABC-NEF/00004383 --baseline 2

#python line3d.py -s /home/c/chenggan/datasets/myBlender/simple1 -m /home/c/chenggan/gaussian-splatting/results/myBlender/simple1 --baseline 2
#python line3d.py -s /home/c/chenggan/datasets/myBlender/simple2 -m /home/c/chenggan/gaussian-splatting/results/myBlender/simple2 --baseline 2
#python line3d.py -s /home/c/chenggan/datasets/db/playroom -m /home/c/chenggan/gaussian-splatting/results/db/playroom/ --baseline 2
#python line3d.py -s /home/c/chenggan/datasets/db/drjohnson -m /home/c/chenggan/gaussian-splatting/results/db/drjohnson/ --baseline 2
#python line3d.py -s /home/c/chenggan/datasets/tandt/train -m /home/c/chenggan/gaussian-splatting/results/tandt/train/ --baseline 2
#python line3d.py -s /home/c/chenggan/datasets/tandt/truck -m /home/c/chenggan/gaussian-splatting/results/tandt/truck/ --baseline 2

# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00020 --model_paths results/db/playroom/dgt_00100 results/db/playroom/dgt_00200 --strategy dist --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00020 --model_paths results/db/playroom/dgt_00090 results/db/playroom/dgt_00190 --strategy dist --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00020 --model_paths results/db/playroom/dgt_00080 results/db/playroom/dgt_00180 --strategy dist --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00020 --model_paths results/db/playroom/dgt_00070 results/db/playroom/dgt_00170 --strategy dist --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00020 --model_paths results/db/playroom/dgt_00060 results/db/playroom/dgt_00160 --strategy dist --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00020 --model_paths results/db/playroom/dgt_00050 results/db/playroom/dgt_00150 --strategy dist --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00020 --model_paths results/db/playroom/dgt_00050 results/db/playroom/dgt_00100 --strategy dist --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00020 --model_paths results/db/playroom/dgt_00045 results/db/playroom/dgt_00090 --strategy dist --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00020 --model_paths results/db/playroom/dgt_00040 results/db/playroom/dgt_00080 --strategy dist --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00020 --model_paths results/db/playroom/dgt_00035 results/db/playroom/dgt_00070 --strategy dist --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00020 --model_paths results/db/playroom/dgt_00030 results/db/playroom/dgt_00060 --strategy dist --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00020 --model_paths results/db/playroom/dgt_00030 results/db/playroom/dgt_00050 --strategy dist --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00025 --model_paths results/db/playroom/dgt_00100 results/db/playroom/dgt_00200 --strategy dist --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00025 --model_paths results/db/playroom/dgt_00090 results/db/playroom/dgt_00190 --strategy dist --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00025 --model_paths results/db/playroom/dgt_00080 results/db/playroom/dgt_00180 --strategy dist --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00025 --model_paths results/db/playroom/dgt_00070 results/db/playroom/dgt_00170 --strategy dist --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00025 --model_paths results/db/playroom/dgt_00060 results/db/playroom/dgt_00160 --strategy dist --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00025 --model_paths results/db/playroom/dgt_00100 results/db/playroom/dgt_00150 --strategy dist --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00025 --model_paths results/db/playroom/dgt_00090 results/db/playroom/dgt_00140 --strategy dist --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00025 --model_paths results/db/playroom/dgt_00080 results/db/playroom/dgt_00130 --strategy dist --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00025 --model_paths results/db/playroom/dgt_00070 results/db/playroom/dgt_00120 --strategy dist --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00025 --model_paths results/db/playroom/dgt_00060 results/db/playroom/dgt_00110 --strategy dist --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00030 --model_paths results/db/playroom/dgt_00100 results/db/playroom/dgt_00200 --strategy dist --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00030 --model_paths results/db/playroom/dgt_00090 results/db/playroom/dgt_00190 --strategy dist --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00030 --model_paths results/db/playroom/dgt_00080 results/db/playroom/dgt_00180 --strategy dist --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00030 --model_paths results/db/playroom/dgt_00070 results/db/playroom/dgt_00170 --strategy dist --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00030 --model_paths results/db/playroom/dgt_00060 results/db/playroom/dgt_00160 --strategy dist --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00030 --model_paths results/db/playroom/dgt_00100 results/db/playroom/dgt_00150 --strategy dist --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00030 --model_paths results/db/playroom/dgt_00090 results/db/playroom/dgt_00140 --strategy dist --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00030 --model_paths results/db/playroom/dgt_00080 results/db/playroom/dgt_00130 --strategy dist --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00030 --model_paths results/db/playroom/dgt_00070 results/db/playroom/dgt_00120 --strategy dist --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00030 --model_paths results/db/playroom/dgt_00060 results/db/playroom/dgt_00110 --strategy dist --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00035 --model_paths results/db/playroom/dgt_00100 results/db/playroom/dgt_00200 --strategy dist --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00035 --model_paths results/db/playroom/dgt_00090 results/db/playroom/dgt_00190 --strategy dist --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00035 --model_paths results/db/playroom/dgt_00080 results/db/playroom/dgt_00180 --strategy dist --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00035 --model_paths results/db/playroom/dgt_00070 results/db/playroom/dgt_00170 --strategy dist --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00035 --model_paths results/db/playroom/dgt_00060 results/db/playroom/dgt_00160 --strategy dist --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00035 --model_paths results/db/playroom/dgt_00100 results/db/playroom/dgt_00150 --strategy dist --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00035 --model_paths results/db/playroom/dgt_00090 results/db/playroom/dgt_00140 --strategy dist --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00035 --model_paths results/db/playroom/dgt_00080 results/db/playroom/dgt_00130 --strategy dist --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00035 --model_paths results/db/playroom/dgt_00070 results/db/playroom/dgt_00120 --strategy dist --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00035 --model_paths results/db/playroom/dgt_00060 results/db/playroom/dgt_00110 --strategy dist --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00040 --model_paths results/db/playroom/dgt_00100 results/db/playroom/dgt_00200 --strategy dist --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00040 --model_paths results/db/playroom/dgt_00090 results/db/playroom/dgt_00190 --strategy dist --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00040 --model_paths results/db/playroom/dgt_00080 results/db/playroom/dgt_00180 --strategy dist --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00040 --model_paths results/db/playroom/dgt_00070 results/db/playroom/dgt_00170 --strategy dist --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00040 --model_paths results/db/playroom/dgt_00060 results/db/playroom/dgt_00160 --strategy dist --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00040 --model_paths results/db/playroom/dgt_00100 results/db/playroom/dgt_00150 --strategy dist --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00040 --model_paths results/db/playroom/dgt_00090 results/db/playroom/dgt_00140 --strategy dist --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00040 --model_paths results/db/playroom/dgt_00080 results/db/playroom/dgt_00130 --strategy dist --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00040 --model_paths results/db/playroom/dgt_00070 results/db/playroom/dgt_00120 --strategy dist --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00040 --model_paths results/db/playroom/dgt_00060 results/db/playroom/dgt_00110 --strategy dist --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00045 --model_paths results/db/playroom/dgt_00100 results/db/playroom/dgt_00200 --strategy dist --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00045 --model_paths results/db/playroom/dgt_00090 results/db/playroom/dgt_00190 --strategy dist --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00045 --model_paths results/db/playroom/dgt_00080 results/db/playroom/dgt_00180 --strategy dist --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00045 --model_paths results/db/playroom/dgt_00070 results/db/playroom/dgt_00170 --strategy dist --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00045 --model_paths results/db/playroom/dgt_00060 results/db/playroom/dgt_00160 --strategy dist --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00045 --model_paths results/db/playroom/dgt_00100 results/db/playroom/dgt_00150 --strategy dist --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00045 --model_paths results/db/playroom/dgt_00090 results/db/playroom/dgt_00140 --strategy dist --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00045 --model_paths results/db/playroom/dgt_00080 results/db/playroom/dgt_00130 --strategy dist --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00045 --model_paths results/db/playroom/dgt_00070 results/db/playroom/dgt_00120 --strategy dist --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00045 --model_paths results/db/playroom/dgt_00060 results/db/playroom/dgt_00110 --strategy dist --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00050 --model_paths results/db/playroom/dgt_00100 results/db/playroom/dgt_00200 --strategy dist --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00050 --model_paths results/db/playroom/dgt_00090 results/db/playroom/dgt_00190 --strategy dist --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00050 --model_paths results/db/playroom/dgt_00080 results/db/playroom/dgt_00180 --strategy dist --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00050 --model_paths results/db/playroom/dgt_00070 results/db/playroom/dgt_00170 --strategy dist --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00050 --model_paths results/db/playroom/dgt_00060 results/db/playroom/dgt_00160 --strategy dist --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00050 --model_paths results/db/playroom/dgt_00100 results/db/playroom/dgt_00150 --strategy dist --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00050 --model_paths results/db/playroom/dgt_00090 results/db/playroom/dgt_00140 --strategy dist --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00050 --model_paths results/db/playroom/dgt_00080 results/db/playroom/dgt_00130 --strategy dist --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00050 --model_paths results/db/playroom/dgt_00070 results/db/playroom/dgt_00120 --strategy dist --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00050 --model_paths results/db/playroom/dgt_00060 results/db/playroom/dgt_00110 --strategy dist --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00060 --model_paths results/db/playroom/dgt_00150 results/db/playroom/dgt_00200 --strategy dist --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00060 --model_paths results/db/playroom/dgt_00140 results/db/playroom/dgt_00190 --strategy dist --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00060 --model_paths results/db/playroom/dgt_00130 results/db/playroom/dgt_00180 --strategy dist --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00060 --model_paths results/db/playroom/dgt_00120 results/db/playroom/dgt_00170 --strategy dist --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00060 --model_paths results/db/playroom/dgt_00110 results/db/playroom/dgt_00160 --strategy dist --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00060 --model_paths results/db/playroom/dgt_00100 results/db/playroom/dgt_00150 --strategy dist --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00060 --model_paths results/db/playroom/dgt_00090 results/db/playroom/dgt_00140 --strategy dist --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00060 --model_paths results/db/playroom/dgt_00080 results/db/playroom/dgt_00130 --strategy dist --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00060 --model_paths results/db/playroom/dgt_00070 results/db/playroom/dgt_00120 --strategy dist --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00060 --model_paths results/db/playroom/dgt_00070 results/db/playroom/dgt_00110 --strategy dist --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00070 --model_paths results/db/playroom/dgt_00160 results/db/playroom/dgt_00200 --strategy dist --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00070 --model_paths results/db/playroom/dgt_00150 results/db/playroom/dgt_00190 --strategy dist --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00070 --model_paths results/db/playroom/dgt_00140 results/db/playroom/dgt_00180 --strategy dist --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00070 --model_paths results/db/playroom/dgt_00130 results/db/playroom/dgt_00170 --strategy dist --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00070 --model_paths results/db/playroom/dgt_00120 results/db/playroom/dgt_00160 --strategy dist --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00070 --model_paths results/db/playroom/dgt_00110 results/db/playroom/dgt_00150 --strategy dist --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00070 --model_paths results/db/playroom/dgt_00100 results/db/playroom/dgt_00140 --strategy dist --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00070 --model_paths results/db/playroom/dgt_00090 results/db/playroom/dgt_00130 --strategy dist --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00070 --model_paths results/db/playroom/dgt_00080 results/db/playroom/dgt_00120 --strategy dist --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00070 --model_paths results/db/playroom/dgt_00080 results/db/playroom/dgt_00110 --strategy dist --render_image
