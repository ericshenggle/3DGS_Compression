import os
import json
import numpy as np
from scipy.constants import point


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def parse_camera_file(filepath):
    cameras = {}
    points = []
    with open(filepath, 'r') as f:
        lines = f.readlines()

        # 第一行是nvm文件的版本号
        # 第二行为空行
        # 第三行是相机的数量
        num_cameras = int(lines[2].strip())
        while len(cameras) < num_cameras:
            # 每个相机的信息占用一行，格式为 filename >> focal_length >> qw >> qx >> qy >> qz >> Cx >> Cy >> Cz >> dist
            # 以空格、制表符分隔
            line = lines[len(cameras) + 3]
            parts = line.strip().split()
            filename = parts[0]
            print(filename)
            focal_length = float(parts[1])
            print(focal_length)
            qw, qx, qy, qz = map(float, parts[2:6])
            print(qw, qx, qy, qz)
            Cx, Cy, Cz = map(float, parts[6:9])
            print(Cx, Cy, Cz)
            dist = float(parts[9])

            R = qvec2rotmat([qw, qx, qy, qz])
            C = np.array([Cx, Cy, Cz])
            t = -R @ C

            cameras[filename] = (focal_length, focal_length, 3072 // 2, 2048 // 2, 3072, 2048, np.array([qw, qx, qy, qz]), t)

        # 空行
        # 下面是特征点的数量
        num_points = int(lines[num_cameras + 4])
        for i in range(num_points):
            # 每个特征点的信息占用一行，格式为 X Y Z R G B num_observations [camera_id, keypoint_id, x, y]
            # 以空格、制表符分隔
            line = lines[num_cameras + 5 + i]
            parts = line.strip().split()
            x, y, z = map(float, parts[:3])
            r, g, b = map(int, parts[3:6])
            points.append((x, y, z, r, g, b, 0.0))

    return cameras, points


def create_nerf_transforms(directory):
    data = {}
    points = []

    # 遍历目录中的所有 .nvm 文件
    for filename in os.listdir(directory):
        if filename.endswith(".nvm"):
            camera_filepath = os.path.join(directory, filename)

            # 解析.camera文件
            data, points = parse_camera_file(camera_filepath)


    camera_lines = ["# Camera list with one line of data per camera:",
                    "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[fx,fy,cx,cy]",
                    "# Number of cameras: {}".format(len(data))]

    image_lines = ["# Image list with two lines of data per image:",
                   "# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
                   "# POINTS2D[] as (X, Y, POINT3D_ID)"]

    point_lines = ["# 3D point list with one line of data per point:",
                     "# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)"]

    image_id = 1  # 初始化图片ID
    camera_id = 1  # 初始化相机ID
    for image_name, (fx, fy, cx, cy, width, height, qvec, tvec) in data.items():
        camera_lines.append(f"{camera_id} PINHOLE {width} {height} {fx} {fy} {cx} {cy}")

        image_lines.append(f"{image_id} {' '.join(map(str, qvec))} {' '.join(map(str, tvec))} {camera_id} {image_name}")
        image_lines.append("")
        image_id += 1
        camera_id += 1

    for i, point in enumerate(points):
        point_lines.append(f"{i} {' '.join(map(str, point))}")


    gt_path = os.path.join(directory, 'ground_truth')
    os.makedirs(gt_path, exist_ok=True)
    os.makedirs(os.path.join(directory, 'sparse'), exist_ok=True)

    cameras_txt_path = os.path.join(gt_path, 'cameras.txt')
    images_txt_path = os.path.join(gt_path, 'images.txt')
    points3D_txt_path = os.path.join(gt_path, 'points3D.txt')

    # 写入cameras.txt
    with open(cameras_txt_path, 'w') as f:
        f.writelines("\n".join(camera_lines))
        # f.write("")

    # 写入images.txt
    with open(images_txt_path, 'w') as f:
        f.writelines("\n".join(image_lines))

    # 写入points3D.txt (文件为空)
    with open(points3D_txt_path, 'w') as f:
        f.writelines("\n".join(point_lines))


if __name__ == "__main__":
    # 设置你的目录路径
    camera_directory = '/home/c/chenggan/datasets/P25'
    create_nerf_transforms(camera_directory)
