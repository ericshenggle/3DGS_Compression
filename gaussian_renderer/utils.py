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
