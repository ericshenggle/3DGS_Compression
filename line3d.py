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
import sys

import numpy as np
import torch

try:
    import wandb
    WANDB_FOUND = True
except ImportError:
    WANDB_FOUND = False

from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

from gaussian_renderer.strategy import transform_points_to_screen_space
from utils.graphics_utils import fov2focal
from scene.colmap_loader import rotmat2qvec
from lines import Line3D
from lines.utils import *
from lines.octree import *


def line3d_baseline2D(dataset: ModelParams, iteration: int, write_colmap=True, write_visualSFM=False):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians)

        means3D = gaussians.get_xyz
        pIDs = torch.arange(1, means3D.shape[0] + 1).unsqueeze(1).cuda()
        means3D_with_ids = torch.cat((means3D, pIDs), dim=1)
        # indices = torch.randperm(means3D_with_ids.shape[0])[:10000]
        # means3D_with_ids = means3D_with_ids[indices]

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
            screen_positions = screen_positions[mask].detach().cpu().numpy()
            # adjust screen positions to the center of the pixel
            screen_positions[:, 0] -= view.image_width / 2
            screen_positions[:, 1] -= view.image_height / 2

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
            cam_center = -np.dot(R, T)
            cam_extrinsics.append(Image(
                id=idx + 1, qvec=qvec, tvec=tvec,
                camera_id=view.colmap_id, name=view.image_name,
                xys=screen_positions, point3D_ids=view_means3D[:, -1]))
            cam_intrinsics.append(Camera(id=view.colmap_id,
                                         model="PINHOLE",
                                         width=width,
                                         height=height,
                                         center=cam_center,
                                         params=np.array([focal_length_x, focal_length_y, width / 2, height / 2])))

        # write the colmap text files
        if write_colmap:
            dir_path = os.path.join(dataset.source_path, "colmap")
            makedirs(dir_path, exist_ok=True)
            write_extrinsics_text(cam_extrinsics, os.path.join(dir_path, "images.txt"))
            write_intrinsics_text(cam_intrinsics, os.path.join(dir_path, "cameras.txt"))
            write_points3D_text(means3D_with_ids, os.path.join(dir_path, "points3D.txt"))

        # # write visualSFM format
        if write_visualSFM:
            dir_path = os.path.join(dataset.source_path, "visualSFM")
            makedirs(dir_path, exist_ok=True)
            write_nvm_file(cam_extrinsics, cam_intrinsics, means3D_with_ids, os.path.join(dir_path, "result.nvm"))


def get_octree(means3D, max_depth, max_points):
    bounds = np.array([[means3D[:, 0].min(), means3D[:, 1].min(), means3D[:, 2].min()],
                       [means3D[:, 0].max(), means3D[:, 1].max(), means3D[:, 2].max()]])
    octree = Octree(bounds, max_depth=max_depth, max_points=max_points)
    for i in range(means3D.shape[0]):
        octree.insert(means3D[i], i)
    return octree


def line3d_baseline3D(dataset: ModelParams, iteration: int, pipeline: PipelineParams, use_cuda: bool):
    dir_path = os.path.join(dataset.source_path, "colmap")
    line3d = Line3D()
    line3d.load3DLinesFromTXT(os.path.join(dir_path, "Line3D++"))
    # lines
    lines = line3d.lines3D()
    means3D = load_ply(os.path.join(dataset.model_path,
                                    "point_cloud",
                                    "iteration_" + str(iteration),
                                    "point_cloud.ply"))

    # downsample the 3D points
    margin = get_margin(means3D, fixed=False, fixed_margin=0.1, dist_ratio=0.02)
    print(f"Margin: {margin}")
    sys.stdout.flush()

    # construct the octree
    octree = get_octree(means3D, max_depth=8, max_points=10)
    # octree.print_tree()
    # save the points that each line3D contains
    makedirs(os.path.join(dir_path, "octree"), exist_ok=True)
    for i, line in enumerate(lines):
        coll = line.collinear3Dsegments()
        for j, s in enumerate(coll):
            filter_points, _ = s.filter_points_within_segment_or_gap(octree, margin=margin)
            save_ply(os.path.join(dir_path, "octree", f"Line{i}_segment{j}.ply"), filter_points)

    line3d.evaluate3Dlines(dir_path, "before", means3D, margin=margin)

    # calculate the density of the all segment3D
    density_list = []
    for i, line in enumerate(lines):
        coll = line.collinear3Dsegments()
        for j, s in enumerate(coll):
            rmse = s.optimize_line(octree, margin=margin, linearRegression=True)
            density, _ = s.calculate_density(octree, margin=margin, recalculate=True)
            print(f"After optimizing line {i}, segment {j}, density: {density}, rmse: {rmse}")
            density_list.append(density)
        line.set_segments(coll)
    density_threshold = calculate_density_threshold(density_list, threshold_ratio=0.05)
    sys.stdout.flush()

    for i, line in enumerate(lines):
        coll = line.collinear3Dsegments()
        # drop the segment3D if can
        print(f"Start dropping Line {i}")
        tmp_coll = []
        for j, s in enumerate(coll):
            density, _ = s.calculate_density(octree, margin=margin)
            if density < density_threshold:
                print(f"Drop segment {j}, density: {density}")
                continue
            tmp_coll.append(s)
        # merge the collinear segment3D if can
        if len(tmp_coll) > 1:
            print(f"Start merging Line {i}, segment3D count: {len(tmp_coll)}")
            coll = merge_all_segments(coll, octree, margin=margin)
            if len(coll) < len(tmp_coll):
                print(f"New number segment3D of Line{i} is {len(coll)}")
        line.set_segments(tmp_coll)
    sys.stdout.flush()

    for i, line in enumerate(lines):
        coll = line.collinear3Dsegments()
        # cropping the segment3D if can
        print(f"Start cropping Line {i}")
        for j, s in enumerate(coll):
            print(f"Start cropping segment {j}, segment length: {s.length()}")
            s.try_cropping(octree, margin=margin)
            s.calculate_density(octree, margin=margin, recalculate=True)
            s.calculate_rmse(octree, margin=margin, recalculate=True)
    sys.stdout.flush()

    # represent the segment3D in the same 3D cluster
    line3d.cluster_3d_segments(octree, margin=margin, dist_threshold=margin)

    line3d.evaluate3Dlines(dir_path, "after", means3D, margin=margin)
    line3d.Write3DlinesToSTL(os.path.join(dir_path, "Line3D++_test"))
    pass


if __name__ == "__main__":
    # Start Time
    import time, datetime

    start_time = time.time()
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    # Set up command line argument parser
    parser = ArgumentParser(description="Getting 3D Lines for 3DGS")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--baseline", default=1, type=int)
    parser.add_argument("--use_cuda", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--write_visualSFM", action="store_true")
    args = get_combined_args(parser)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    if args.baseline == 1:
        # store the 3DGS coord information as colmap text format in source_path, in order to use line3D++
        line3d_baseline2D(model.extract(args), args.iteration, write_visualSFM=args.write_visualSFM)

    if args.baseline == 2:
        # apply the line3D++ cluster algorithm directly on 3DGS
        if WANDB_FOUND and args.wandb:
            wandb.init(project="3Dlines", dir=args.source_path, config=vars(args), name="line3D_" + args.model_path.split("/")[-1], group="baseline3D")
        line3d_baseline3D(model.extract(args), args.iteration, pipeline.extract(args), args.use_cuda)

    # End time
    end_time = time.time()

    # Save time
    from utils.system_utils import save_timeline

    save_timeline(f"line3D: {now}", start_time, end_time, args.model_path)
