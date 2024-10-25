import os
import json
import numpy as np

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

def parse_camera_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # 读取内参矩阵 (3x3)
    intrinsics = np.array([list(map(float, line.strip().split())) for line in lines[:3]])
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]

    # 读取旋转矩阵 (3x3)
    rotation_matrix = np.array([list(map(float, line.strip().split())) for line in lines[4:7]])

    # 读取平移向量 (1x3)
    translation_vector = np.array(list(map(float, lines[7].strip().split())))

    # 读取图像尺寸
    width, height = map(int, lines[8].strip().split())

    # 构建4x4外参矩阵
    extrinsics = np.eye(4)
    extrinsics[:3, :3] = rotation_matrix
    extrinsics[:3, 3] = translation_vector

    extrinsics[:, 1:3] = -extrinsics[:, 1:3]  # 修正y轴和z轴的方向

    return fx, fy, cx, cy, width, height, extrinsics


def create_nerf_transforms(directory):
    data = {}

    # 遍历目录中的所有 .camera 文件
    for filename in os.listdir(directory):
        if filename.endswith(".camera"):
            camera_filepath = os.path.join(directory, filename)
            image_name = filename.replace(".camera", "")  # 假设对应的图像是png格式

            # 解析.camera文件
            res = parse_camera_file(camera_filepath)

            data[image_name] = res

    camera_lines = ["# Camera list with one line of data per camera:",
                    "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[fx,fy,cx,cy]",
                    "# Number of cameras: {}".format(len(data))]

    image_lines = ["# Image list with two lines of data per image:",
                   "# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
                   "# POINTS2D[] as (X, Y, POINT3D_ID)"]

    image_id = 1  # 初始化图片ID
    camera_id = 1  # 初始化相机ID
    for image_name, (fx, fy, cx, cy, width, height, extrinsics) in data.items():
        camera_lines.append(f"{camera_id} PINHOLE {width} {height} {fx} {fy} {cx} {cy}")
        R = extrinsics[:3, :3]
        qvec = rotmat2qvec(R)

        tvec = extrinsics[:3, 3]

        image_lines.append(f"{image_id} {' '.join(map(str, qvec))} {' '.join(map(str, tvec))} {camera_id} {image_name}")
        image_lines.append("")
        image_id += 1
        camera_id += 1


    gt_path = os.path.join(os.path.dirname(directory), 'ground_truth')
    os.makedirs(gt_path, exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(directory), 'sparse'), exist_ok=True)

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
        pass


if __name__ == "__main__":
    # 设置你的目录路径
    camera_directory = '/home/c/chenggan/datasets/herzjesu_dense_large/camera'
    create_nerf_transforms(camera_directory)
