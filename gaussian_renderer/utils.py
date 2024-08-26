import torch

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