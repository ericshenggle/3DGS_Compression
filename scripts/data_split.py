import os
import shutil
import numpy as np
import argparse

def copy_files_by_percentage(src_directory, dst_root, percentages):
    if not os.path.exists(src_directory):
        raise Exception(f"The source directory {src_directory} does not exist.")

    # Get all file names in the source directory
    all_files = [f for f in os.listdir(src_directory) if os.path.isfile(os.path.join(src_directory, f))]
    num_files = len(all_files)

    # Loop over the specified percentages
    for pct in percentages:
        # Calculate the number of files to copy for the current percentage
        num_files_to_copy = int(num_files * pct)

        # Choose a random subset of files
        files_to_copy = np.random.choice(all_files, num_files_to_copy, replace=False)

        # Create destination directory for the current percentage
        dst_directory = os.path.join(dst_root, f"{int(pct*100)}_percent", "input")
        os.makedirs(dst_directory, exist_ok=True)

        # Copy each file to the new directory
        for filename in files_to_copy:
            src_file_path = os.path.join(src_directory, filename)
            dst_file_path = os.path.join(dst_directory, filename)
            shutil.copy2(src_file_path, dst_file_path)
        print(f"Copied {num_files_to_copy} files to {dst_directory}")

# Example usage:
# source_path = 'data/train/100_percent/images'  # Replace with your source directory
# data_path = 'data/train'  # Replace with your destination root directory
# percentages = [0.8]  # The percentages of the files you want to copy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_path", required=True, help="The source directory to copy files from.")
    parser.add_argument("--data_path", required=True, help="The root directory to copy files to.")
    parser.add_argument("--percentages", nargs="+", type=float, default=[0.8, 0.9], help="The percentages of files to copy.")
    args = parser.parse_args()
    copy_files_by_percentage(args.source_path, args.data_path, args.percentages)

