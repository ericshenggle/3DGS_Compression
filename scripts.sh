#!/bin/sh

#SBATCH --job-name=3dgs
#SBATCH --time=120:00:00
#SBATCH --partition=gpu-long
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cyang_09@u.nus.edu

# sbatch --gres=gpu:nv:1 -C cuda75 scripts.sh
# sbatch -w xcnf27 scripts.sh

#python cameraPose_to_nerf.py
#python train.py -s /home/c/chenggan/datasets/P25 -m /home/c/chenggan/gaussian-splatting/results/P25
#python line3d.py -s /home/c/chenggan/datasets/P25 -m /home/c/chenggan/gaussian-splatting/results/P25 --baseline 2

python line3d.py -s /home/c/chenggan/datasets/ABC-NEF/ -m /home/c/chenggan/gaussian-splatting/results/ABC-NEF/ --baseline 3
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

# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00020 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00021 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00022 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00023 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00024 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00025 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00026 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00027 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00028 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00029 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00030 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00035 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00040 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00045 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00050 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00060 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00070 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00080 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00090 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00100 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00110 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00120 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00130 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00140 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00150 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00160 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00170 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00180 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00190 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00200 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00250 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00300 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00350 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00400 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00500 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00600 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00700 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00800 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00900 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_01000 --strategy distFov --render_image


# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00020 --model_paths results/db/playroom/dgt_00100 results/db/playroom/dgt_00200 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00020 --model_paths results/db/playroom/dgt_00090 results/db/playroom/dgt_00190 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00020 --model_paths results/db/playroom/dgt_00080 results/db/playroom/dgt_00180 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00020 --model_paths results/db/playroom/dgt_00070 results/db/playroom/dgt_00170 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00020 --model_paths results/db/playroom/dgt_00060 results/db/playroom/dgt_00160 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00020 --model_paths results/db/playroom/dgt_00050 results/db/playroom/dgt_00150 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00020 --model_paths results/db/playroom/dgt_00050 results/db/playroom/dgt_00100 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00020 --model_paths results/db/playroom/dgt_00045 results/db/playroom/dgt_00090 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00020 --model_paths results/db/playroom/dgt_00040 results/db/playroom/dgt_00080 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00020 --model_paths results/db/playroom/dgt_00035 results/db/playroom/dgt_00070 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00020 --model_paths results/db/playroom/dgt_00030 results/db/playroom/dgt_00060 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00020 --model_paths results/db/playroom/dgt_00030 results/db/playroom/dgt_00050 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00025 --model_paths results/db/playroom/dgt_00100 results/db/playroom/dgt_00200 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00025 --model_paths results/db/playroom/dgt_00090 results/db/playroom/dgt_00190 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00025 --model_paths results/db/playroom/dgt_00080 results/db/playroom/dgt_00180 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00025 --model_paths results/db/playroom/dgt_00070 results/db/playroom/dgt_00170 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00025 --model_paths results/db/playroom/dgt_00060 results/db/playroom/dgt_00160 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00025 --model_paths results/db/playroom/dgt_00100 results/db/playroom/dgt_00150 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00025 --model_paths results/db/playroom/dgt_00090 results/db/playroom/dgt_00140 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00025 --model_paths results/db/playroom/dgt_00080 results/db/playroom/dgt_00130 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00025 --model_paths results/db/playroom/dgt_00070 results/db/playroom/dgt_00120 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00025 --model_paths results/db/playroom/dgt_00060 results/db/playroom/dgt_00110 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00030 --model_paths results/db/playroom/dgt_00100 results/db/playroom/dgt_00200 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00030 --model_paths results/db/playroom/dgt_00090 results/db/playroom/dgt_00190 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00030 --model_paths results/db/playroom/dgt_00080 results/db/playroom/dgt_00180 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00030 --model_paths results/db/playroom/dgt_00070 results/db/playroom/dgt_00170 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00030 --model_paths results/db/playroom/dgt_00060 results/db/playroom/dgt_00160 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00030 --model_paths results/db/playroom/dgt_00100 results/db/playroom/dgt_00150 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00030 --model_paths results/db/playroom/dgt_00090 results/db/playroom/dgt_00140 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00030 --model_paths results/db/playroom/dgt_00080 results/db/playroom/dgt_00130 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00030 --model_paths results/db/playroom/dgt_00070 results/db/playroom/dgt_00120 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00030 --model_paths results/db/playroom/dgt_00060 results/db/playroom/dgt_00110 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00035 --model_paths results/db/playroom/dgt_00100 results/db/playroom/dgt_00200 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00035 --model_paths results/db/playroom/dgt_00090 results/db/playroom/dgt_00190 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00035 --model_paths results/db/playroom/dgt_00080 results/db/playroom/dgt_00180 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00035 --model_paths results/db/playroom/dgt_00070 results/db/playroom/dgt_00170 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00035 --model_paths results/db/playroom/dgt_00060 results/db/playroom/dgt_00160 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00035 --model_paths results/db/playroom/dgt_00100 results/db/playroom/dgt_00150 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00035 --model_paths results/db/playroom/dgt_00090 results/db/playroom/dgt_00140 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00035 --model_paths results/db/playroom/dgt_00080 results/db/playroom/dgt_00130 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00035 --model_paths results/db/playroom/dgt_00070 results/db/playroom/dgt_00120 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00035 --model_paths results/db/playroom/dgt_00060 results/db/playroom/dgt_00110 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00040 --model_paths results/db/playroom/dgt_00100 results/db/playroom/dgt_00200 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00040 --model_paths results/db/playroom/dgt_00090 results/db/playroom/dgt_00190 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00040 --model_paths results/db/playroom/dgt_00080 results/db/playroom/dgt_00180 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00040 --model_paths results/db/playroom/dgt_00070 results/db/playroom/dgt_00170 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00040 --model_paths results/db/playroom/dgt_00060 results/db/playroom/dgt_00160 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00040 --model_paths results/db/playroom/dgt_00100 results/db/playroom/dgt_00150 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00040 --model_paths results/db/playroom/dgt_00090 results/db/playroom/dgt_00140 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00040 --model_paths results/db/playroom/dgt_00080 results/db/playroom/dgt_00130 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00040 --model_paths results/db/playroom/dgt_00070 results/db/playroom/dgt_00120 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00040 --model_paths results/db/playroom/dgt_00060 results/db/playroom/dgt_00110 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00045 --model_paths results/db/playroom/dgt_00100 results/db/playroom/dgt_00200 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00045 --model_paths results/db/playroom/dgt_00090 results/db/playroom/dgt_00190 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00045 --model_paths results/db/playroom/dgt_00080 results/db/playroom/dgt_00180 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00045 --model_paths results/db/playroom/dgt_00070 results/db/playroom/dgt_00170 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00045 --model_paths results/db/playroom/dgt_00060 results/db/playroom/dgt_00160 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00045 --model_paths results/db/playroom/dgt_00100 results/db/playroom/dgt_00150 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00045 --model_paths results/db/playroom/dgt_00090 results/db/playroom/dgt_00140 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00045 --model_paths results/db/playroom/dgt_00080 results/db/playroom/dgt_00130 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00045 --model_paths results/db/playroom/dgt_00070 results/db/playroom/dgt_00120 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00045 --model_paths results/db/playroom/dgt_00060 results/db/playroom/dgt_00110 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00050 --model_paths results/db/playroom/dgt_00100 results/db/playroom/dgt_00200 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00050 --model_paths results/db/playroom/dgt_00090 results/db/playroom/dgt_00190 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00050 --model_paths results/db/playroom/dgt_00080 results/db/playroom/dgt_00180 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00050 --model_paths results/db/playroom/dgt_00070 results/db/playroom/dgt_00170 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00050 --model_paths results/db/playroom/dgt_00060 results/db/playroom/dgt_00160 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00050 --model_paths results/db/playroom/dgt_00100 results/db/playroom/dgt_00150 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00050 --model_paths results/db/playroom/dgt_00090 results/db/playroom/dgt_00140 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00050 --model_paths results/db/playroom/dgt_00080 results/db/playroom/dgt_00130 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00050 --model_paths results/db/playroom/dgt_00070 results/db/playroom/dgt_00120 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00050 --model_paths results/db/playroom/dgt_00060 results/db/playroom/dgt_00110 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00060 --model_paths results/db/playroom/dgt_00150 results/db/playroom/dgt_00200 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00060 --model_paths results/db/playroom/dgt_00140 results/db/playroom/dgt_00190 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00060 --model_paths results/db/playroom/dgt_00130 results/db/playroom/dgt_00180 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00060 --model_paths results/db/playroom/dgt_00120 results/db/playroom/dgt_00170 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00060 --model_paths results/db/playroom/dgt_00110 results/db/playroom/dgt_00160 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00060 --model_paths results/db/playroom/dgt_00100 results/db/playroom/dgt_00150 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00060 --model_paths results/db/playroom/dgt_00090 results/db/playroom/dgt_00140 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00060 --model_paths results/db/playroom/dgt_00080 results/db/playroom/dgt_00130 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00060 --model_paths results/db/playroom/dgt_00070 results/db/playroom/dgt_00120 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00060 --model_paths results/db/playroom/dgt_00070 results/db/playroom/dgt_00110 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00070 --model_paths results/db/playroom/dgt_00160 results/db/playroom/dgt_00200 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00070 --model_paths results/db/playroom/dgt_00150 results/db/playroom/dgt_00190 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00070 --model_paths results/db/playroom/dgt_00140 results/db/playroom/dgt_00180 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00070 --model_paths results/db/playroom/dgt_00130 results/db/playroom/dgt_00170 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00070 --model_paths results/db/playroom/dgt_00120 results/db/playroom/dgt_00160 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00070 --model_paths results/db/playroom/dgt_00110 results/db/playroom/dgt_00150 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00070 --model_paths results/db/playroom/dgt_00100 results/db/playroom/dgt_00140 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00070 --model_paths results/db/playroom/dgt_00090 results/db/playroom/dgt_00130 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00070 --model_paths results/db/playroom/dgt_00080 results/db/playroom/dgt_00120 --strategy distFov --render_image
# python render_multiModel.py -s /home/c/chenggan/datasets/db/playroom -m results/db/playroom/dgt_00070 --model_paths results/db/playroom/dgt_00080 results/db/playroom/dgt_00110 --strategy distFov --render_image
