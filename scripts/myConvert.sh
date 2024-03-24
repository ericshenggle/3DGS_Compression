#!/bin/sh

#SBATCH --job-name=3dgs
#SBATCH --time=24:00:00
#SBATCH --gpus=t4:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cyang_09@u.nus.edu
#SBATCH --partition=long

# # script to colmap with images and camera pose
# 1. transfer the dataset with camera pose(The VR-NeRF Eyeful Tower Dataset) to colmap's format
python cameraPose_to_colmap.py --source_path /home/c/chenggan/datasets/apartment/ --data_path /home/c/chenggan/datasets/apartment_25_gt/
# # 2. colmap extract features
# colmap feature_extractor --ImageReader.camera_model PINHOLE --database_path /home/c/chenggan/datasets/apartment_25_gt/database.db --image_path /home/c/chenggan/datasets/apartment_25_gt/images/
# # 3. update the database with the camera pose
# python transfer_to_database.py --database_path /home/c/chenggan/datasets/apartment_25_gt/database.db --cameras_path /home/c/chenggan/datasets/apartment_25_gt/created/sparse/cameras.txt
# # 4. colmap match features
# colmap exhaustive_matcher --database_path /home/c/chenggan/datasets/apartment_25_gt/database.db
# # 5. colmap triangulate
# colmap point_triangulator --database_path /home/c/chenggan/datasets/apartment_25_gt/database.db --image_path /home/c/chenggan/datasets/apartment_25_gt/images/ --input_path /home/c/chenggan/datasets/apartment_25_gt/created/sparse --output_path /home/c/chenggan/datasets/apartment_25_gt/triangulated/sparse
# # 6. colmap undistort
# colmap image_undistorter --image_path /home/c/chenggan/datasets/apartment_25_gt/images/ --input_path /home/c/chenggan/datasets/apartment_25_gt/triangulated/sparse --output_path /home/c/chenggan/datasets/apartment_25_gt/undistorted/ --output_type COLMAP
# python ../convert.py -s /home/c/chenggan/datasets/apartment_25_non_gt/