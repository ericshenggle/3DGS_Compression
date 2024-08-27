#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
import csv
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render_multiModel, render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.image_utils import psnr
from utils.loss_utils import ssim
from lpipsPyTorch import lpips

import collections
import numpy as np
from gaussian_renderer.strategy import transform_points_to_screen_space
from utils.graphics_utils import fov2focal
from scene.colmap_loader import rotmat2qvec


Image = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])

def write_extrinsics_text(images, path, gt=False):
    """
    Write the images dictionary back to a text file.
    """
    with open(path, "w") as fid:
        for image_id, image in enumerate(images):
            # Write the first line
            qvec_str = " ".join(map(str, image.qvec))
            tvec_str = " ".join(map(str, image.tvec))
            image_name = image.name + ".png"
            fid.write(f"{image.id} {qvec_str} {tvec_str} {image.camera_id} {image_name}\n")

            # Write the second line, unless gt is True
            if not gt:
                xys_str = " ".join(f"{xy[0]} {xy[1]} {int(pt_id)}" for xy, pt_id in zip(image.xys, image.point3D_ids))
                fid.write(f"{xys_str}\n")

def write_intrinsics_text(cameras, path):
    """
    Write the cameras dictionary back to a text file.
    """
    with open(path, "w") as fid:
        for camera_id, camera in enumerate(cameras):
            # Ensure that the model is PINHOLE as expected
            assert camera.model == "PINHOLE", "This function assumes the camera model is PINHOLE"
            
            # Convert parameters to string
            params_str = " ".join(map(str, camera.params))
            
            # Write the camera information to the file
            fid.write(f"{camera.id} {camera.model} {camera.width} {camera.height} {params_str}\n")

def write_points3D_text(means3D, path):
    with open(path, "w") as f:
        for point in means3D:
            x, y, z, point3d_id = point  # 忽略Z坐标
            f.write(f"{int(point3d_id)} {x} {y} {z}\n")


def line3d_baseline2D(dataset : ModelParams, iteration : int):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration)

        means3D = gaussians.get_xyz
        pIDs = torch.arange(1, means3D.shape[0] + 1).unsqueeze(1).cuda()
        means3D_with_ids = torch.cat((means3D, pIDs), dim=1)
        indices = torch.randperm(means3D_with_ids.shape[0])[:10000]
        means3D_with_ids = means3D_with_ids[indices]

        views = scene.getTrainCameras()

        cam_extrinsics = []
        cam_intrinsics = []

        for idx, view in enumerate(tqdm(views, desc="Getting progress")):

            # masking to remove points outside the visible screen area
            screen_positions = transform_points_to_screen_space(means3D_with_ids[:, :3], view)
            within_bounds_x = (screen_positions[:, 0] >= 0) & (screen_positions[:, 0] <= view.image_width - 1)
            within_bounds_y = (screen_positions[:, 1] >= 0) & (screen_positions[:, 1] <= view.image_height - 1)
            mask = (within_bounds_x & within_bounds_y)
            original_count = means3D_with_ids.shape[0]

            filtered_count = mask.sum().item()
            print(f"before project: {original_count}")
            print(f"after project: {filtered_count}")

            view_means3D = means3D_with_ids[mask].detach().cpu().numpy()

            R = view.R
            T = view.T
            height = view.image_height
            width = view.image_width
            FoVx = view.FoVx
            FoVy = view.FoVy

            focal_length_x = fov2focal(FoVx, width)
            focal_length_y = fov2focal(FoVy, height)

            qvec = rotmat2qvec(np.transpose(R))
            tvec = T.tolist()
            qvec_str = " ".join(map(str, qvec))
            tvec_str = " ".join(map(str, tvec))
            cam_extrinsics.append(Image(
                id=idx + 1, qvec=qvec, tvec=tvec,
                camera_id=view.colmap_id, name=view.image_name,
                xys=view_means3D[:, :2], point3D_ids=view_means3D[:, 3]))
            cam_intrinsics.append(Camera(id=view.colmap_id,
                                        model="PINHOLE",
                                        width=width,
                                        height=height,
                                        params=np.array([focal_length_x, focal_length_y, width / 2, height/ 2])))

        dir_path = os.path.join(args.source_path, "colmap")
        makedirs(dir_path, exist_ok=True)
        write_extrinsics_text(cam_extrinsics, os.path.join(dir_path, "images.txt"))
        write_intrinsics_text(cam_intrinsics, os.path.join(dir_path, "cameras.txt"))
        write_points3D_text(means3D_with_ids, os.path.join(dir_path, "points3D.txt"))
   
def line3d_baseline3D(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, paths : list):
    
    pass

if __name__ == "__main__":
    # Start Time
    import time 
    start_time = time.time()

    # Set up command line argument parser
    parser = ArgumentParser(description="Getting 3D Lines for 3DGS")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--model_paths", nargs="*", type=str, default=[])
    parser.add_argument("--combinedDebug", action="store_true")
    parser.add_argument("--strategy", type=str, default="dist")
    parser.add_argument("--render_image", action="store_true")
    parser.add_argument("--baseline", default=1, type=int)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    print("Combined with " + str(args.model_paths))

    # Initialize system state (RNG)
    safe_state(args.quiet)

    if args.baseline == 1:
        # store the 3DGS coord information as colmap text format in source_path, in order to use line3D++
        line3d_baseline2D(model.extract(args), args.iteration)

    if args.baseline == 2:
        # apply the line3D++ cluster algorithm directly on 3DGS
        line3d_baseline3D(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.model_paths)

    # End time
    end_time = time.time()

    # Save time
    from utils.system_utils import save_timeline
    save_timeline('render', start_time, end_time, args.model_path)