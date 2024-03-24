#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from errno import EEXIST
from os import makedirs, path
import os
import time
import json

def mkdir_p(folder_path):
    # Creates a directory. equivalent to using mkdir -p on the command line
    try:
        makedirs(folder_path)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(folder_path):
            pass
        else:
            raise

def searchForMaxIteration(folder):
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder)]
    return max(saved_iters)

def save_timeline(script_name, start_time, end_time, path):
    if not os.path.exists(path):
        os.makedirs(path)

    time_format = "%H:%M:%S"
    total_time = end_time - start_time
    total_time_str = time.strftime(time_format, time.gmtime(total_time))

    time_path = os.path.join(path, "timeline.json")
    if os.path.exists(time_path):
        # Load the json file
        with open(time_path, 'r') as f:
            data = json.load(f)
    else:
        data = []

    data.append({"script_name": script_name, "time": total_time_str})
    with open(os.path.join(path, "timeline.json"), 'w') as f:
        json.dump(data, f, indent=True)

def sum_timelines(path):
    time_path = os.path.join(path, "timeline.json")
    if os.path.exists(time_path):
        # Load the json file
        with open(time_path, 'r') as f:
            data = json.load(f)
        total_seconds = 0
        for script in data:
            script_time = time.strptime(script["time"], "%H:%M:%S")
            script_seconds = script_time.tm_hour * 3600 + script_time.tm_min * 60 + script_time.tm_sec
            total_seconds += script_seconds

        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        total_time_str = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
        data.append({"total_time": total_time_str})
        with open(time_path, 'w') as f:
            json.dump(data, f, indent=True)
        

