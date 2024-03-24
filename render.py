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
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, compression : bool):
    if not compression:
        with torch.no_grad():
            gaussians = GaussianModel(dataset.sh_degree)
            scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

            bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
            background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

            if not skip_train:
                render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

            if not skip_test:
                render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)
    else:
        with torch.no_grad():
            origin_model_path = dataset.model_path
            gaussians_dict = {}
            scene_dict = {}
            for i in range(5, 11, 1):
                new_model_path = os.path.join(origin_model_path, str(int(i * 10)) + "_percent")
                dataset.model_path = new_model_path
                gaussians = GaussianModel(dataset.sh_degree)
                scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
                gaussians_dict[i] = gaussians
                scene_dict[i] = scene
            
            for i in range(5, 11, 1):
                bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
                background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

                if not skip_train:
                    render_set(scene_dict[i].model_path, "train", scene_dict[i].loaded_iter, scene_dict[i].getTrainCameras(), gaussians_dict[i], pipeline, background)

                if not skip_test:
                    render_set(scene_dict[i].model_path, "test", scene_dict[i].loaded_iter, scene_dict[10].getTestCameras(), gaussians_dict[i], pipeline, background)
   

if __name__ == "__main__":
    # Start Time
    import time 
    start_time = time.time()

    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--compression", action='store_true', default=False)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.compression)

    # End time
    end_time = time.time()

    # Save time
    from utils.system_utils import save_timeline
    save_timeline('render', start_time, end_time, args.model_path)