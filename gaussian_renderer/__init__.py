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
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from gaussian_renderer.strategy import *

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}

def render_multiModel(viewpoint_camera, pc : list, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, combinedDebug = False, strategy=""):
    """
    Render the scene using multiple Gaussian models.    
    
    Background tensor (bg_color) must be on GPU!
    """
    assert len(pc) >= 1, "At least one Gaussian model must be provided for multi-model rendering."
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means\

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=math.tan(viewpoint_camera.FoVx * 0.5),
        tanfovy=math.tan(viewpoint_camera.FoVy * 0.5),
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc[0].active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    if strategy == "fov":
        masks = foveated_selection(pc, viewpoint_camera)
    else:
        masks = distance_based_selection(pc, viewpoint_camera)

    mean3D, mean2D, opacity, scales, rotations, cov3D_precomp, shs, colors_precomp = [], [], [], [], [], [], [], []

    for i, mask in enumerate(masks):
        if combinedDebug and i % 2 != 0:
            continue
        mean3D_tmp = pc[i].get_xyz

        print("Model ", i, " has original points: ", len(pc[i].get_xyz), " and after filter: ", len(pc[i].get_xyz[mask]))
        mean3D.append(mean3D_tmp[mask])
        mean2D.append(torch.zeros_like(mean3D[-1], dtype=mean3D[-1].dtype, requires_grad=True, device="cuda") + 0)
        opacity.append(pc[i].get_opacity[mask])

        scales_tmp = None
        rotations_tmp = None
        cov3D_precomp_tmp = None
        if pipe.compute_cov3D_python:
            cov3D_precomp_tmp = pc[i].get_covariance(scaling_modifier)[mask]
        else:
            scales_tmp = pc[i].get_scaling[mask]
            rotations_tmp = pc[i].get_rotation[mask]

        shs_tmp = None
        colors_precomp_tmp = None
        if override_color is None:
            if pipe.convert_SHs_python:
                shs_view = pc[i].get_features.transpose(1, 2).view(-1, 3, (pc[i].max_sh_degree+1)**2)
                dir_pp = (pc[i].get_xyz - viewpoint_camera.camera_center.repeat(pc[i].get_features.shape[0], 1))
                dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(pc[i].active_sh_degree, shs_view, dir_pp_normalized)
                colors_precomp_tmp = torch.clamp_min(sh2rgb + 0.5, 0.0)[mask]
            else:
                shs_tmp = pc[i].get_features[mask]
        else:
            colors_precomp_tmp = override_color
        
        scales.append(scales_tmp)
        rotations.append(rotations_tmp)
        cov3D_precomp.append(cov3D_precomp_tmp)
        shs.append(shs_tmp)
        colors_precomp.append(colors_precomp_tmp)

    # Concatenate the list of tensors to single tensor
    mean3D = torch.cat(mean3D, dim=0)
    num_points = mean3D.shape[0]
    screenspace_points = torch.cat(mean2D, dim=0)
    try:
        screenspace_points.retain_grad()
    except:
        pass
    opacity = torch.cat(opacity, dim=0)
    scales = torch.cat(scales, dim=0) if scales[0] is not None else None
    rotations = torch.cat(rotations, dim=0) if rotations[0] is not None else None
    cov3D_precomp = torch.cat(cov3D_precomp, dim=0) if cov3D_precomp[0] is not None else None
    shs = torch.cat(shs, dim=0) if shs[0] is not None else None
    colors_precomp = torch.cat(colors_precomp, dim=0) if colors_precomp[0] is not None else None

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii = rasterizer(
        means3D = mean3D,
        means2D = screenspace_points,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "num_points": num_points}
