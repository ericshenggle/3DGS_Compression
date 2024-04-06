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
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render_multiModel
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, combinedDebug=False):
    render_path = os.path.join(model_path, name, "renders")
    gts_path = os.path.join(model_path, name, "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render_multiModel(view, gaussians, pipeline, background, combinedDebug=combinedDebug)["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, paths : list, combinedDebug : bool):
    with torch.no_grad():
        scene = Scene(dataset, None, load_iteration=iteration, shuffle=False, unloadGaussians=True)
        gaussians = []
        gaussians_temp = GaussianModel(dataset.sh_degree)
        gaussians_temp.load_ply(os.path.join(dataset.model_path, "point_cloud", "iteration_" + str(iteration), "point_cloud.ply"))
        gaussians.append(gaussians_temp)
        for path in paths:
            gaussians_temp = GaussianModel(dataset.sh_degree)
            gaussians_temp.load_ply(os.path.join(path, "point_cloud", "iteration_" + str(iteration), "point_cloud.ply"))
            gaussians.append(gaussians_temp)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")


        p_model_path = os.path.dirname(dataset.model_path) + "_combined"
        new_model_path = os.path.basename(dataset.model_path)
        for path in paths:
            new_model_path += "_" + os.path.basename(path)
        model_path = os.path.join(p_model_path, new_model_path)

        if not skip_train:
            render_set(model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, combinedDebug=combinedDebug)

        if not skip_test:
            render_set(model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, combinedDebug=combinedDebug)
   

if __name__ == "__main__":
    # Start Time
    import time 
    start_time = time.time()

    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--skip_train", action="store_true", default=True)
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--model_paths", nargs="*", type=str, required=True, default=[])
    parser.add_argument("--combinedDebug", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    print("Combined with " + str(args.model_paths))

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.model_paths, args.combinedDebug)

    # End time
    end_time = time.time()

    # Save time
    from utils.system_utils import save_timeline
    save_timeline('render', start_time, end_time, args.model_path)