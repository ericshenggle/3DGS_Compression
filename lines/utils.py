import numpy as np
import collections
from plyfile import PlyData, PlyElement
import itertools

from arguments import SegmentParams
from lines import Octree

Image = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "center", "params"])


def write_extrinsics_text(images, path, gt=False):
    """
    Write the images dictionary back to a text file.
    """
    with open(path, "w") as fid:
        for image_id, image in enumerate(images):
            # Write the first line
            qvec_str = " ".join(map(str, image.qvec))
            tvec_str = " ".join(map(str, image.tvec))
            image_name = image.name + ".png"
            fid.write(f"{image.id} {qvec_str} {tvec_str} {image.camera_id} {image_name}\n")

            # Write the second line, unless gt is True
            if not gt:
                xys_str = " ".join(f"{xy[0]} {xy[1]} {int(pt_id)}" for xy, pt_id in zip(image.xys, image.point3D_ids))
                fid.write(f"{xys_str}\n")


def write_intrinsics_text(cameras, path):
    """
    Write the cameras dictionary back to a text file.
    """
    with open(path, "w") as fid:
        for camera_id, camera in enumerate(cameras):
            # Ensure that the model is PINHOLE as expected
            assert camera.model == "PINHOLE", "This function assumes the camera model is PINHOLE"

            # Convert parameters to string
            params_str = " ".join(map(str, camera.params))

            # Write the camera information to the file
            fid.write(f"{camera.id} {camera.model} {camera.width} {camera.height} {params_str}\n")


def write_points3D_text(means3D, path):
    with open(path, "w") as f:
        for point in means3D:
            x, y, z, point3d_id = point  # 忽略Z坐标
            f.write(f"{int(point3d_id)} {x} {y} {z}\n")


def write_nvm_file(images, cameras, means3D, path):
    with open(path, "w") as f:
        f.write("NVM_V3\n")
        f.write("\n")
        f.write(f"{len(images)}\n")
        for image, camera in zip(images, cameras):
            qvec_str = " ".join(map(str, image.qvec))
            focal_length_x, focal_length_y, cx, cy = camera.params
            image_name = image.name + ".png"
            center_str = " ".join(map(str, camera.center))
            f.write(f"{image_name} {focal_length_x} {qvec_str} {center_str} 0\n")

        f.write("\n")
        f.write(f"{len(means3D)}\n")
        for point in means3D:
            x, y, z, point3d_id = point
            point3d_id = int(point3d_id)
            f.write(f"{x} {y} {z} 255 255 255 ")
            # Get the list of images that see this point
            num_images = 0
            tmp_str = ""
            for image in images:
                for xy, pt_id in zip(image.xys, image.point3D_ids):
                    if int(pt_id) == point3d_id:
                        tmp_str += f"{image.id - 1} 0 {xy[0]} {xy[1]} "
                        num_images += 1
                        break
            f.write(f"{num_images} {tmp_str}\n")



def merge_all_segments(segments, octree : Octree, args : SegmentParams, margin):
    attempted_merges = set()
    while True:
        merged_any = False

        for segment1, segment2 in itertools.combinations(segments, 2):
            segment_pair_id = (id(segment1), id(segment2))
            if segment_pair_id in attempted_merges:
                continue

            merged_segment = segment1.try_segments_merge(segment2, octree, args, margin)

            if merged_segment:
                segments.remove(segment1)
                segments.remove(segment2)
                segments.append(merged_segment)
                merged_any = True
                break

            attempted_merges.add(segment_pair_id)

        if not merged_any:
            break

    return segments

def load_ply(path):
    plydata = PlyData.read(path)

    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])), axis=1)
    return xyz

def save_ply(path, points):
    vertex = np.array([tuple(point) for point in points],
                      dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    el = PlyData([PlyElement.describe(vertex, "vertex")])
    el.write(path)

def save_ply_with_colors(path, points):
    vertex = np.array([tuple(point) for point in points],
                      dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ])


def calculate_density_threshold(density_list, args : SegmentParams):
    """Calculate the density threshold based on the distribution of density."""
    density_list = np.array(density_list)
    # calculate the mean and standard deviation of the density
    mean_density = np.mean(density_list)
    std_density = np.std(density_list)
    # calculate the density threshold
    # TODO: This threshold is a heuristic and may need to be adjusted
    if args.den_threshold_ratio == -1:
        density_threshold = mean_density - 2 * std_density
        density_threshold = max(density_threshold, 0)
    else:
        density_threshold = mean_density * args.den_threshold_ratio
    return density_threshold

def preprocess_margin(points, args : SegmentParams):
    """
    Calculate the margin based on the point cloud.
    """
    # get the 90% quantile of the distance between the points and the center
    if args.margin != -1:
        args.cluster_dist_threshold = args.margin
        return
    center = np.mean(points, axis=0)
    dist = np.linalg.norm(points - center, axis=1)
    points_90 = points[dist < np.percentile(dist, args.margin_percentile)]
    max_dist = np.max(np.linalg.norm(points_90 - center, axis=1))
    args.margin = max_dist * args.margin_dist_ratio
    args.cluster_dist_threshold = args.margin


def calculate_3D_line_score_v3(covered_points_ratio, rmse_list, length, weight=1.0):
    """
    Calculate a score for a 3D line based on different factors, using z-score normalization or log scaling.

    Parameters:
    - covered_points_ratio: Ratio of points covered by the 3D line.
    - rmse_list: List of RMSE values for each 3D segment.
    - length: Length of the 3D line.
    - w_RMSE: Weight for the RMSE (penalizes high RMSE).

    Returns:
    - score: A single score representing the quality of the line.
    """
    RMSE_scaled = np.mean(rmse_list)
    # Apply normalization to length and RMSE
    # length range is [0, +inf), scale to [0, 1], longer is worse, so need to inverse
    # larger RMSE is worse, so need to inverse
    RMSE_scaled = 1 / np.log(1 + RMSE_scaled)
    line_scale = weight * covered_points_ratio * 100 / np.log(1 + length)

    # Calculate the score
    score = line_scale * RMSE_scaled

    return score


