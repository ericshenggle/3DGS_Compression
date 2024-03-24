import json
import numpy as np
import os
import shutil
import argparse
import math
from scipy.spatial.transform import Rotation as R

def select_images_from_json(camera_json_path, target_count=25):
    """
    从camera.json文件中选择图片。
    返回一个包含选中图片cameraId的列表。
    """
    with open(camera_json_path, 'r') as file:
        data = json.load(file)

    selected_images = {}
    for item in data['KRT']:
        camera_id = item['cameraId'].split('/')[0]
        if camera_id not in selected_images:
            selected_images[camera_id] = []
        file_name = item['cameraId'] + '.jpg'
        selected_images[camera_id].append(file_name)

    # 对于每个相机，线性下采样到target_count数量
    downsampled_images = {}
    for camera_id, images in selected_images.items():
        step = max(1, len(images) // target_count)
        downsampled_images[camera_id] = images[::step][:target_count]

    return downsampled_images

def copy_downsampled_images(source_path, data_path, type, downsampled_images):
    """
    根据camera.json中选定的图片，复制文件到目标目录。
    """
    target_dir = data_path
    os.makedirs(target_dir, exist_ok=True)

    for camera_id, images in downsampled_images.items():
        for image_id in images:
            filename = image_id.split('/')[1]
            file_path = os.path.join(source_path, type, camera_id, filename)
            if os.path.exists(file_path):
                shutil.copy(file_path, target_dir)
                print(f"Copied {file_path} to {target_dir}")

def create_sparse_model_colmap(data_path, camera_json_path, selected_files):
    gt_path = os.path.join(data_path, 'created/sparse')
    os.makedirs(gt_path, exist_ok=True)
    triangulated_path = os.path.join(data_path, 'triangulated/sparse')
    os.makedirs(triangulated_path, exist_ok=True)

    cameras_txt_path = os.path.join(gt_path, 'cameras.txt')
    images_txt_path = os.path.join(gt_path, 'images.txt')
    points3D_txt_path = os.path.join(gt_path, 'points3D.txt')

    # 读取camera.json文件
    with open(camera_json_path, 'r') as file:
        data = json.load(file)

    selected_filenames = {os.path.basename(path): camera_id for camera_id, paths in selected_files.items() for path in paths}
    camera_lines = ["# Camera list with one line of data per camera:",
                    "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[fx,fy,cx,cy]",
                    "# Number of cameras: {}".format(len(selected_filenames))]

    image_lines = ["# Image list with two lines of data per image:",
                   "# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
                   "# POINTS2D[] as (X, Y, POINT3D_ID)"]

    image_id = 1  # 初始化图片ID
    camera_id = 1  # 初始化相机ID
    for camera in data['KRT']:
        image_name = camera['cameraId'].split('/')[1] + '.jpg'  # 生成图片名称
        if image_name not in selected_filenames:
            continue  # 如果图片未被选中，跳过

        K = np.array(camera['K']).reshape((3, 3)).T
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        fx, fy = fx * 1368 / 5784, fy * 2048 / 8660
        camera_lines.append(f"{camera_id} PINHOLE 1368 2048 {fx} {fy} {cx} {cy}")

        T = np.array(camera['T']).reshape((4, 4)).T
        T[:3, 1:3] *= -1
        q = rotmat2qvec(T[:3, :3])
        tx, ty, tz = T[:3, 3]
        image_lines.append(f"{image_id} {' '.join(map(str, q))} {tx} {ty} {tz} {camera_id} {image_name}")
        image_lines.append("")  # 每个图片描述后留空行
        image_id += 1
        camera_id += 1

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

def create_NerfSynthetic_model(data_path, camera_json_path, selected_files):
    nerfSynthetic_path = data_path
    os.makedirs(nerfSynthetic_path, exist_ok=True)

    train_path = os.path.join(nerfSynthetic_path, 'transforms_train.json')
    test_path = os.path.join(nerfSynthetic_path, 'transforms_test.json')

    # 读取camera.json文件
    with open(camera_json_path, 'r') as file:
        data = json.load(file)

    selected_filenames = {os.path.basename(path): camera_id for camera_id, paths in selected_files.items() for path in paths}

    cameras_json = {}
    train_json = {"frames": []}
    test_json = {"frames": []}

    for idx, camera in enumerate(data['KRT']):
        image_name = camera['cameraId'].split('/')[1] + '.jpg'  # 生成图片名称
        if image_name not in selected_filenames:
            continue  # 如果图片未被选中，跳过

        K = np.array(camera['K']).reshape((3, 3)).T
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        fovx = focal2fov(fx, camera['width'])
        if "camera_angle_x" not in cameras_json:
            cameras_json["camera_angle_x"] = fovx
            train_json["camera_angle_x"] = fovx
            test_json["camera_angle_x"] = fovx

        T = np.array(camera['T']).reshape((4, 4)).T
        
        camera_json = {}
        camera_json["file_path"] = camera['cameraId'].split('/')[1]
        camera_json["transform_matrix"] = T.tolist()
        if idx % 8 != 0:
            train_json["frames"].append(camera_json)
        else:
            test_json["frames"].append(camera_json)

    with open(train_path, 'w') as f:
        json.dump(train_json, f, indent=4)

    with open(test_path, 'w') as f:
        json.dump(test_json, f, indent=4)

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))
    
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

def select_images_from_path(path):
    images = {}
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.jpg'):
                camera_id = os.path.basename(file).split('_')[0]
                if camera_id not in images:
                    images[camera_id] = []
                images[camera_id].append(file)
                
    return images

def save_downsampled_to_json(downsampled_images, source_path, data_path):
    with open(os.path.join(source_path, 'cameras.json'), 'r') as file:
        data = json.load(file)

    downsampled_images_camera = []
    for item in data['KRT']:
        image_name = item['cameraId'].split('/')[1] + '.jpg'
        if image_name in downsampled_images:
            downsampled_images_camera.append(item)
    
    with open(os.path.join(data_path, 'cameras.json'), 'w') as file:
        json.dump(downsampled_images_camera, file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True, help="The root directory to copy files to.")
    parser.add_argument("--source_path", required=True, help="The source directory to copy files from.")
    parser.add_argument("--type", default="images-jpeg-2k")

    args = parser.parse_args()

    downsampled_images = select_images_from_json(os.path.join(args.source_path, 'cameras.json'))
    copy_downsampled_images(args.source_path, args.data_path, args.type, downsampled_images)
    # downsampled_images = select_images_from_path(os.path.join(args.data_path, 'images'))
    create_NerfSynthetic_model(args.data_path, os.path.join(args.source_path, 'cameras.json'), downsampled_images)
