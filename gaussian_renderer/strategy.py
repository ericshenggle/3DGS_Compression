import torch
import math

# define the Sector of 3D
class Sector3D:
    def __init__(self, center, radius, height, angle_range, direction):
        self.center = center
        self.radius = radius
        self.height_range = -height, height
        self.angle_range = angle_range
        self.direction = direction

# Judge whether the points are in the sector
def is_points_in_sector_3d(points, sector):
    # calculate the vector from the center to the points  
    vectors = points - sector.center
    distances = torch.norm(vectors, dim=1)
    
    in_radius = distances <= sector.radius
    
    directions = vectors / distances.unsqueeze(1)
    cos_angles = torch.sum(directions * sector.direction, dim=1)
    angles = torch.acos(cos_angles).mul(180 / torch.pi)  # calculate the angle between the direction and the sector direction
    
    start_angle, end_angle = sector.angle_range
    if start_angle <= end_angle:
        in_angle = (start_angle <= angles) & (angles <= end_angle)
    else:
        in_angle = (start_angle <= angles) | (angles <= end_angle)

    return in_radius & in_angle

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
    
    screen_x = ((points_ndc[:, 0] + 1) * 0.5) * (viewpoint_camera.image_width - 1)
    screen_y = ((points_ndc[:, 1] + 1) * 0.5) * (viewpoint_camera.image_height - 1)
    
    return torch.stack([screen_x, screen_y], dim=1)

def distance_based_selection(pc, viewpoint_camera, overlap_ratio=0.1):
    """
    Select points based on distance thresholds, with thresholds calculated internally.
    
    :param pc: A list of point clouds.
    :param viewpoint_camera: The camera object containing camera center and other properties.
    :return: A list of masks, each corresponding to a point cloud in `pc`.
    """
    image_width, image_height = viewpoint_camera.image_width, viewpoint_camera.image_height
    all_distances = [torch.sqrt(((pc[0].get_xyz - viewpoint_camera.camera_center) ** 2).sum(dim=1))]
    percentiles = [torch.quantile(all_distances, i / len(pc)) for i in range(len(pc) + 1)]
    
    masks = []
    for i, pc_i in enumerate(pc):
        distances = torch.sqrt(((pc_i.get_xyz - viewpoint_camera.camera_center) ** 2).sum(dim=1))
        lower_bound = percentiles[i] - (percentiles[1] - percentiles[0]) * overlap_ratio if i > 0 else 0
        upper_bound = percentiles[i + 1] + (percentiles[1] - percentiles[0]) * overlap_ratio
        if i == 0:
            mask = (distances <= upper_bound)
        elif i == len(pc) - 1:
            mask = (distances >= lower_bound)
        else:
            mask = (distances >= lower_bound) & (distances < upper_bound)

        # Additional masking to remove points outside the visible screen area
        screen_positions = transform_points_to_screen_space(pc_i.get_xyz, viewpoint_camera)
        within_bounds_x = (screen_positions[:, 0] >= 0) & (screen_positions[:, 0] <= image_width - 1)
        within_bounds_y = (screen_positions[:, 1] >= 0) & (screen_positions[:, 1] <= image_height - 1)
        mask &= (within_bounds_x & within_bounds_y)

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
    min_dimension = math.sqrt(math.pow(image_width, 2) + math.pow(image_height, 2))
    max_radius = min_dimension / 2
    step_size = max_radius / len(pc) 
    fov_steps = [step_size * i for i in range(len(pc) + 1)]

    image_center = torch.tensor([image_width / 2, image_height / 2], device=viewpoint_camera.data_device)
    masks = []
    for i, pc_i in enumerate(pc):
        screen_positions = transform_points_to_screen_space(pc_i.get_xyz, viewpoint_camera)
        distances_from_center = torch.sqrt(((screen_positions[:, :2] - image_center) ** 2).sum(dim=1))
        lower_bound = fov_steps[i] - step_size * overlap_ratio if i > 0 else 0
        upper_bound = fov_steps[i + 1] + step_size * overlap_ratio

        if i == 0:
            mask = (distances_from_center <= upper_bound)
        elif i == len(pc) - 1:
            mask = (distances_from_center >= lower_bound)
        else:
            mask = (distances_from_center >= lower_bound) & (distances_from_center <= upper_bound)

        # Additional masking to remove points outside the visible screen area
        within_bounds_x = (screen_positions[:, 0] >= 0) & (screen_positions[:, 0] <= image_width - 1)
        within_bounds_y = (screen_positions[:, 1] >= 0) & (screen_positions[:, 1] <= image_height - 1)
        mask &= (within_bounds_x & within_bounds_y)

        masks.append(mask)
    return masks

def distFoveated_selection(pc, viewpoint_camera, overlap_ratio=0.1):
    """
    Select points based on their distance from the image center, simulating a foveated rendering effect.
    
    :param pc: A list of point clouds.
    :param viewpoint_camera: The camera object containing camera center and other properties.
    :return: A list of masks, each corresponding to a point cloud in `pc`.
    """
    image_width, image_height = viewpoint_camera.image_width, viewpoint_camera.image_height
    all_distances = torch.sqrt(((pc[0].get_xyz - viewpoint_camera.camera_center) ** 2).sum(dim=1))
    sector_num = len(pc) - 1
    percentiles = [torch.quantile(all_distances, i / sector_num) for i in range(sector_num + 1)]

    # Get the direction vector of the camera
    direction_vector = viewpoint_camera.world_view_transform.inverse()[:3, 2]
    direction_vector = direction_vector / torch.norm(direction_vector)

    sector = []
    for i in range(sector_num):
        center = viewpoint_camera.camera_center
        # Add a small forward offset to center
        center += direction_vector * percentiles[i] * 0.05
        radius = percentiles[i + 1]
        height = percentiles[i + 1]
        angle_range = (30, 150)
        sector.append(Sector3D(center, radius, height, angle_range, direction_vector))

    masks = []
    for i, pc_i in enumerate(pc):
        # Count the number of sectors that each point belongs to
        counts = torch.zeros(pc_i.get_xyz.shape[0], dtype=torch.int32, device=viewpoint_camera.data_device)
        for j in range(len(sector)):
            counts += is_points_in_sector_3d(pc_i.get_xyz, sector[j]).to(torch.int32)
        
        # Select points that based on the number of sectors they belong to, i.e., the better the model, the more sectors it belongs to
        mask = (counts == len(pc) - 1 - i)

        # Additional masking to remove points outside the visible screen area
        screen_positions = transform_points_to_screen_space(pc_i.get_xyz, viewpoint_camera)
        within_bounds_x = (screen_positions[:, 0] >= 0) & (screen_positions[:, 0] <= image_width - 1)
        within_bounds_y = (screen_positions[:, 1] >= 0) & (screen_positions[:, 1] <= image_height - 1)
        mask &= (within_bounds_x & within_bounds_y)

        masks.append(mask)
    return masks

