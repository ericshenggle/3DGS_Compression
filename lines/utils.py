import numpy as np
import collections
from plyfile import PlyData
import itertools

Image = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])


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


def merge_all_segments(segments, points):
    attempted_merges = set()
    while True:
        merged_any = False

        for segment1, segment2 in itertools.combinations(segments, 2):
            segment_pair_id = (id(segment1), id(segment2))
            if segment_pair_id in attempted_merges:
                continue

            merged_segment = segment1.try_segments_merge(segment2, points)

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

def calculate_density_threshold(density_list):
    """Calculate the density threshold based on the distribution of density."""
    density_list = np.array(density_list)
    # calculate the mean and standard deviation of the density
    mean_density = np.mean(density_list)
    std_density = np.std(density_list)
    # calculate the density threshold
    # TODO: This threshold is a heuristic and may need to be adjusted
    density_threshold = mean_density * 0.2
    return density_threshold
