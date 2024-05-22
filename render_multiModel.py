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


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, combinedDebug, strategy, render_image=False):
    render_dir_path = ''
    if render_image:
        render_dir_path = os.path.join(model_path, name)
        makedirs(render_dir_path, exist_ok=True)
    # for key, value in gaussians.items():
    #     makedirs(os.path.join(model_path, key), exist_ok=True)

    avg_points = 0

    psnr_metric = []
    ssim_metric = []
    lpips_metric = []

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        _result = render_multiModel(view, list(gaussians.values()), pipeline, background, combinedDebug=combinedDebug, strategy=strategy)
        rendering = _result["render"]
        avg_points = avg_points * idx / (idx + 1) + _result["num_points"] / (idx + 1)
        gt = view.original_image[0:3, :, :]

        psnr_metric.append(psnr(rendering, gt))
        ssim_metric.append(ssim(rendering, gt))
        lpips_metric.append(lpips(rendering, gt, net_type="vgg"))

        if render_image:
            torchvision.utils.save_image(rendering, os.path.join(render_dir_path, '{0:05d}'.format(idx) + ".png"))

    metrics = {
        f"{name}_PSNR": torch.stack(psnr_metric).mean().item(),
        f"{name}_SSIM": torch.stack(ssim_metric).mean().item(),
        f"{name}_LPIPS": torch.stack(lpips_metric).mean().item()
    }

    # if render_image:
    #     for key, value in gaussians.items():
    #         for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
    #             rendering = render(view, value, pipeline, background)["render"]
    #             gt = view.original_image[0:3, :, :]
    #             torchvision.utils.save_image(rendering, os.path.join(model_path, key, '{0:05d}'.format(idx) + ".png"))

    return int(avg_points), metrics

def render_gt(model_path, name, views):
    render_dir_path = os.path.join(model_path, name)
    makedirs(render_dir_path, exist_ok=True)

    if len(os.listdir(render_dir_path)) > 0:
        return

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(gt, os.path.join(render_dir_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, paths : list, combinedDebug : bool, strategy : str, render_image : bool):
    with torch.no_grad():
        scene = Scene(dataset, None, load_iteration=iteration, shuffle=False, unloadGaussians=True)
        gaussians = {}
        gaussians_temp = GaussianModel(dataset.sh_degree)
        gaussians_temp.load_ply(os.path.join(dataset.model_path, "point_cloud", "iteration_" + str(iteration), "point_cloud.ply"))
        gaussians[os.path.basename(dataset.model_path)] = gaussians_temp
        for path in paths:
            gaussians_temp = GaussianModel(dataset.sh_degree)
            gaussians_temp.load_ply(os.path.join(path, "point_cloud", "iteration_" + str(iteration), "point_cloud.ply"))
            gaussians[os.path.basename(path)] = gaussians_temp

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")


        p_model_path = os.path.dirname(dataset.model_path) + "_combined_" + strategy
        new_model_path = os.path.basename(dataset.model_path)
        for path in paths:
            new_model_path += "_" + os.path.basename(path)
        model_path = os.path.join(p_model_path, new_model_path)
        if combinedDebug:
            model_path += "_debug"

        render_gt(os.path.join(p_model_path, "ground_truth"), "train", scene.getTrainCameras())
        render_gt(os.path.join(p_model_path, "ground_truth"), "test", scene.getTestCameras())

        # sort the gaussians by the number of points
        # if strategy == "fov":
        #     gaussians = {k: v for k, v in sorted(gaussians.items(), key=lambda item: item[1].get_xyz.shape[0])}
        # else:
        #     gaussians = {k: v for k, v in sorted(gaussians.items(), key=lambda item: item[1].get_xyz.shape[0], reverse=True)}
        gaussians = {k: v for k, v in sorted(gaussians.items(), key=lambda item: item[1].get_xyz.shape[0], reverse=True)}

        train_points, test_points = 0, 0
        train_metrics, test_metrics = {}, {}

        if not skip_train:
            train_points, train_metrics = render_set(model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, combinedDebug, strategy, False)

        if not skip_test:
            test_points, test_metrics = render_set(model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, combinedDebug, strategy, render_image)

        file_path = os.path.join(p_model_path, "metrics.csv")
        file_exists = os.path.isfile(file_path)

        with open(file_path, "a", newline='') as csvfile:
            fieldnames = ['model', 'train_points', 'train_PSNR', 'train_SSIM', 'train_LPIPS', 'test_points', 'test_PSNR', 'test_SSIM', 'test_LPIPS']
    
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()

            row_dict = {
                'model': '_'.join(gaussians.keys()),
            }
            if not skip_train:
                row_dict['train_points'] = train_points
                for metric_key in train_metrics:
                    row_dict[metric_key] = train_metrics[metric_key]
            if not skip_test:
                row_dict['test_points'] = test_points
                for metric_key in test_metrics:
                    row_dict[metric_key] = test_metrics[metric_key]
            writer.writerow(row_dict)
   

if __name__ == "__main__":
    # Start Time
    import time 
    start_time = time.time()

    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
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
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    print("Combined with " + str(args.model_paths))

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.model_paths, args.combinedDebug, args.strategy, args.render_image)

    # End time
    end_time = time.time()

    # Save time
    from utils.system_utils import save_timeline
    save_timeline('render', start_time, end_time, args.model_path)