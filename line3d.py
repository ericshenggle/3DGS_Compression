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
import matplotlib.pyplot as plt
from sympy.matrices.expressions.blockmatrix import bounds

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
from arguments import ModelParams, PipelineParams, get_combined_args, SegmentParams
from gaussian_renderer import GaussianModel

from gaussian_renderer.strategy import transform_points_to_screen_space
from utils.graphics_utils import fov2focal
from scene.colmap_loader import rotmat2qvec
from lines import Line3D, Segment3D
from lines.utils import *
from lines.octree import *


def line3d_baseline2D(dataset: ModelParams, iteration: int, write_colmap=True, write_visualSFM=False):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians)

        means3D = gaussians.get_xyz
        pIDs = torch.arange(1, means3D.shape[0] + 1).unsqueeze(1).cuda()
        means3D_with_ids = torch.cat((means3D, pIDs), dim=1)
        if means3D_with_ids.shape[0] > 20000:
            indices = torch.randperm(means3D_with_ids.shape[0])[:20000]
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
            # original_count = means3D_with_ids.shape[0]
            #
            # filtered_count = mask.sum().item()
            # print(f"before project: {original_count}")
            # print(f"after project: {filtered_count}")

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
        # if write_visualSFM:
        #     dir_path = os.path.join(dataset.source_path, "visualSFM")
        #     makedirs(dir_path, exist_ok=True)
        #     write_nvm_file(cam_extrinsics, cam_intrinsics, means3D_with_ids, os.path.join(dir_path, "result.nvm"))


def get_octree(means3D, line3d, args : SegmentParams):
    bounds = line3d.get_bounds()
    # filter the points that are outside the bounds
    mask = (means3D[:, 0] >= bounds[0][0]) & (means3D[:, 0] <= bounds[1][0]) & \
              (means3D[:, 1] >= bounds[0][1]) & (means3D[:, 1] <= bounds[1][1]) & \
                (means3D[:, 2] >= bounds[0][2]) & (means3D[:, 2] <= bounds[1][2])
    means3D = means3D[mask]
    octree = Octree(bounds, max_depth=args.max_depth, max_points=args.max_points)
    for i in range(means3D.shape[0]):
        octree.insert(means3D[i], i)
    return octree, means3D


def line3d_baseline3D(source_path, model_path, args : SegmentParams):
    dir_path = os.path.join(source_path, "colmap")
    line3d = Line3D()
    line3d.load3DLinesFromOBJ(os.path.join(dir_path, "Line3D++"))
    # lines
    lines = line3d.lines3D()

    means3D = load_ply(os.path.join(model_path,
                                    "point_cloud",
                                    "iteration_30000",
                                    "point_cloud.ply"))
    # construct the octree
    octree, means3D = get_octree(means3D, line3d, args)
    # octree.save_ply(os.path.join(dir_path, "octree.ply"), line3d.lines3D(), margin=args.margin)
    # return

    # downsample the 3D points
    preprocess_margin(means3D, args)
    print(f"Margin: {args.margin}")
    sys.stdout.flush()

    before_val = line3d.evaluate3Dlines(dir_path, "before", octree, margin=args.eval_margin)

    # calculate the density of the all segment3D
    density_list = []
    for i, line in enumerate(lines):
        coll = line.collinear3Dsegments()
        for j, s in enumerate(coll):
            rmse = s.optimize_line(octree, args)
            density, _ = s.calculate_density(octree, margin=args.margin, recalculate=True)
            # print(f"After optimizing line {i}, segment {j}, density: {density}, rmse: {rmse}")
            density_list.append(density)
        line.set_segments(coll)
    density_threshold = calculate_density_threshold(density_list, args)
    sys.stdout.flush()

    for i, line in enumerate(lines):
        coll = line.collinear3Dsegments()
        # drop the segment3D if can
        # print(f"Start dropping Line {i}")
        tmp_coll = []
        for j, s in enumerate(coll):
            density, _ = s.calculate_density(octree, margin=args.margin)
            if density <= density_threshold:
                # print(f"Drop segment {j}, density: {density}")
                continue
            tmp_coll.append(s)
        # merge the collinear segment3D if can
        if len(tmp_coll) > 1:
            # print(f"Start merging Line {i}, segment3D count: {len(tmp_coll)}")
            tmp_coll = merge_all_segments(tmp_coll, octree, args, margin=args.margin)
        line.set_segments(tmp_coll)
    sys.stdout.flush()

    # represent the segment3D in the same 3D cluster
    line3d.cluster_3d_segments(octree, args)

    for i, line in enumerate(lines):
        coll = line.collinear3Dsegments()
        # cropping the segment3D if can
        # print(f"Start cropping Line {i}")
        for j, s in enumerate(coll):
            # print(f"Start cropping segment {j}, segment length: {s.length()}")
            s.try_cropping(octree, args)
            s.calculate_density(octree, margin=args.margin, recalculate=True)
            s.calculate_rmse(octree, margin=args.margin, recalculate=True)
    sys.stdout.flush()

    # save the 3D lines
    line3d.Write3DlinesToSTL(os.path.join(dir_path, "Line3D++_test"))

    after_val = line3d.evaluate3Dlines(dir_path, "after", octree, margin=args.eval_margin)
    print(f"Before: {before_val}")
    print(f"After: {after_val}")

    return before_val, after_val

eval_param_choices = {
    # "den_threshold_ratio": [0.05, 0.1, 0.15, 0.2, 0.25, -1],
    # "margin": [0.01, 0.03, 0.05, 0.07, 0.09, -1],
    # "join_length_threshold": [0.5, 0.6, 0.7, 0.8, 0.9],
    # "join_den_threshold": [0.3, 0.4, 0.5, 0.6, 0.7],
    # "join_rmse_threshold": [0.9, 1.0, 1.1, 1.2, 1.3],
    # "merge_dist_threshold": [0.1, 0.15, 0.2, 0.25, 0.3],
    # "merge_den_threshold": [0.3, 0.4, 0.5, 0.6, 0.7],
    "eval_margin": [0.1]
}

def eval_segmentParams(dataset : ModelParams, args : SegmentParams):
    source_path = dataset.source_path
    model_path = dataset.model_path

    # Find the path that contains the colmap directory
    dir_paths = []
    model_paths = []
    for root, dirs, files in os.walk(source_path):
        if "colmap" in dirs:
            dir_paths.append(root)
            if os.path.basename(root) == os.path.basename(model_path):
                model_paths.append(model_path)
            else:
                model_paths.append(os.path.join(model_path, os.path.basename(root)))

    # random shuffle the paths
    np.random.seed(42)
    indices = np.random.permutation(len(dir_paths))
    dir_paths = [dir_paths[i] for i in indices]
    test_set = len(dir_paths)
    dir_paths = dir_paths[:test_set] if len(dir_paths) > test_set else dir_paths
    model_paths = [model_paths[i] for i in indices]
    model_paths = model_paths[:test_set] if len(model_paths) > test_set else model_paths

    for eval_param, param_values in eval_param_choices.items():
        param_before = []
        param_after = []
        param_improvement = []
        for param in param_values:
            print(f"Start evaluating {eval_param} with value {param}")
            before = []
            after = []
            improvement = []
            for dir_path, model_path in zip(dir_paths, model_paths):
                setattr(args, eval_param, param)
                print(f"Start evaluating {os.path.basename(dir_path)}")
                before_val, after_val = line3d_baseline3D(dir_path, model_path, args)
                # calculate the improvement
                rmse_improvement = (before_val[0] - after_val[0]) / before_val[0] * 100
                points_improvement = (after_val[1] - before_val[1]) / before_val[1] * 100
                length_improvement = (before_val[2] - after_val[2]) / before_val[2] * 100
                score_improvement = (after_val[3] - before_val[3]) / before_val[3] * 100
                print(f"rmse improvement: {rmse_improvement:.3f}%, points improvement: {points_improvement:.3f}%, " +
                      f"length improvement: {length_improvement:.3f}%, score improvement: {score_improvement:.3f}%")
                before.append(before_val)
                after.append(after_val)
                improvement.append([rmse_improvement, points_improvement, length_improvement, score_improvement])
            before = np.array(before)
            after = np.array(after)
            improvement = np.array(improvement)
            # print the dir_path that has the best improvement
            best_idx = np.argmax(improvement[:, 3])
            print(f"Best improvement in {dir_paths[best_idx]}")
            before = np.mean(before, axis=0)
            after = np.mean(after, axis=0)
            improvement = np.mean(improvement, axis=0)
            param_before.append(before)
            param_after.append(after)
            param_improvement.append(improvement)

        # plot all kinds of improvement in one figure
        # plot the bar chart
        # x-axis is the parameter value
        # y-axis is the value
        param_dict = {
            "before": param_before,
            "after": param_after,
            "improvement": param_improvement
        }
        for key, value in param_dict.items():
            value = np.array(value)
            ind = np.arange(len(param_values))
            width = 0.2
            plt.figure(figsize=(10, 8))

            b1 = plt.bar(ind - 1.5 * width, value[:, 0], width, label="RMSE")
            b2 = plt.bar(ind - 0.5 * width, value[:, 1], width, label="Coverage")
            b3 = plt.bar(ind + 0.5 * width, value[:, 2], width, label="Length")
            b4 = plt.bar(ind + 1.5 * width, value[:, 3], width, label="Score")
            plt.bar_label(b1, labels=np.round(value[:, 0], 2), label_type="edge", fontsize=6, padding=3)
            plt.bar_label(b2, labels=np.round(value[:, 1], 2), label_type="edge", fontsize=6, padding=3)
            plt.bar_label(b3, labels=np.round(value[:, 2], 2), label_type="edge", fontsize=6, padding=3)
            plt.bar_label(b4, labels=np.round(value[:, 3], 2), label_type="edge", fontsize=6, padding=3)

            plt.xlabel(eval_param, fontsize=12)
            plt.ylabel("Improvement (%)", fontsize=12)
            plt.title(f"Improvement of Line3D++ with different {eval_param}")
            plt.xticks(ind, param_values)
            plt.legend(loc="best")
            # save the figure
            plt.savefig(os.path.join(source_path, f"{eval_param}_{key}.png"))


if __name__ == "__main__":
    # Start Time
    import time, datetime

    start_time = time.time()
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    # Set up command line argument parser
    parser = ArgumentParser(description="Getting 3D Lines for 3DGS")
    model = ModelParams(parser, sentinel=True)
    # SegmentParams contains the parameters for the lineGS algorithm
    seg = SegmentParams(parser)
    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--baseline", default=2, type=int)
    parser.add_argument("--use_cuda", action="store_true")
    parser.add_argument("--write_visualSFM", action="store_true")
    args = parser.parse_args(sys.argv[1:])

    # Initialize system state (RNG)
    safe_state(args.quiet)

    if args.baseline == 1:
        # store the 3DGS coord information as colmap text format in source_path, in order to use line3D++
        args = get_combined_args(parser)
        line3d_baseline2D(model.extract(args), args.iteration, write_visualSFM=args.write_visualSFM)

    if args.baseline == 2:
        # apply the line3D++ cluster algorithm directly on 3DGS
        dataset = model.extract(args)
        line3d_baseline3D(dataset.source_path, dataset.model_path, seg.extract(args))

    if args.baseline == 3:
        eval_segmentParams(model.extract(args), seg.extract(args))

    # End time
    end_time = time.time()

    # Save time
    from utils.system_utils import save_timeline

    save_timeline(f"line3D: {now}", start_time, end_time, args.model_path)
