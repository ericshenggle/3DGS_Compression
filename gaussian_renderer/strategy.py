import torch

def distance_based_selection(pc, viewpoint_camera, overlap_ratio=0.1):
    """
    Select points based on distance thresholds, with thresholds calculated internally.
    
    :param pc: A list of point clouds.
    :param viewpoint_camera: The camera object containing camera center and other properties.
    :return: A list of masks, each corresponding to a point cloud in `pc`.
    """
    all_distances = torch.cat([torch.sqrt(((pc[0].get_xyz - viewpoint_camera.camera_center) ** 2).sum(dim=1))])
    percentiles = [torch.quantile(all_distances, i / len(pc)) for i in range(len(pc) + 1)]
    
    masks = []
    for i, pc_i in enumerate(pc):
        distances = torch.sqrt(((pc_i.get_xyz - viewpoint_camera.camera_center) ** 2).sum(dim=1))

        lower_bound = percentiles[max(i, 0)] - (percentiles[1] - percentiles[0]) * overlap_ratio if i > 0 else 0
        upper_bound = percentiles[min(i + 1, len(pc) - 1)] + (percentiles[1] - percentiles[0]) * overlap_ratio
        if i == 0:
            mask = (distances <= upper_bound)
        elif i == len(pc) - 1:
            mask = (distances >= lower_bound)
        else:
            mask = (distances >= lower_bound) & (distances < upper_bound)
        masks.append(mask)
    return masks


def foveated_selection(pc, viewpoint_camera, overlap_ratio=0.1):
    """
    Select points based on their distance from the image center, simulating a foveated rendering effect.
    
    :param pc: A list of point clouds.
    :param viewpoint_camera: The camera object containing camera center and other properties.
    :return: A list of masks, each corresponding to a point cloud in `pc`.
    """
    image_width, image_height = viewpoint_camera.image_width, viewpoint_camera.image_height
    min_dimension = min(image_width, image_height)
    max_radius = min_dimension / 1.5
    step_size = max_radius / len(pc) 
    fov_steps = [step_size * i for i in range(len(pc) + 1)]

    image_center = torch.tensor([image_width / 2, image_height / 2], device=viewpoint_camera.data_device)
    masks = []
    for i, pc_i in enumerate(pc):
        screen_positions = transform_points_to_screen_space(pc_i.get_xyz, viewpoint_camera)
        distances_from_center = torch.sqrt(((screen_positions[:, :2] - image_center) ** 2).sum(dim=1))

        lower_bound = fov_steps[max(i, 0)] - step_size * overlap_ratio if i > 0 else 0
        upper_bound = fov_steps[min(i + 1, len(pc) - 1)] + step_size * overlap_ratio

        if i == 0:
            mask = (distances_from_center <= upper_bound)
        elif i == len(pc) - 1:
            mask = (distances_from_center >= lower_bound)
        else:
            mask = (distances_from_center >= lower_bound) & (distances_from_center <= upper_bound)
        
        masks.append(mask)
    return masks

def transform_points_to_screen_space(points, viewpoint_camera):
    """
    Transform points from world space to screen space using the full projection transform.

    :param points: A tensor of shape (N, 3) representing N points in world space.
    :param viewpoint_camera: The camera object containing camera center and other properties.
    :return: A tensor of shape (N, 2) representing the screen space coordinates of the points.
    """
    ones = torch.ones((points.shape[0], 1), device=viewpoint_camera.data_device)
    points_homogeneous = torch.cat([points, ones], dim=1)
    
    points_clip = torch.mm(points_homogeneous, viewpoint_camera.full_proj_transform)
    
    points_ndc = points_clip[:, :3] / points_clip[:, 3].unsqueeze(1)
    
    screen_x = ((points_ndc[:, 0] + 1) * viewpoint_camera.image_width - 1) * 0.5
    screen_y = ((points_ndc[:, 1] + 1) * viewpoint_camera.image_height - 1) * 0.5
    
    return torch.stack([screen_x, screen_y], dim=1)
