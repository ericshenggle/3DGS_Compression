import os
import numpy as np

from lines.segment3D import *
from lines.clustering import *
from lines.utils import calculate_3D_line_score_v3


class Line3D:
    def __init__(self):
        self.prefix_wng_ = "Warning: "
        self.lines3D_ = []
        self.file_name_ = ""
        self.before_optimize_ = None

    def lines3D(self) -> List:
        return self.lines3D_

    def load3DLinesFromTXT(self, input_folder):
        # get filename
        for f in os.listdir(input_folder):
            # 检查文件是否以 .txt 结尾
            if f.endswith('.txt'):
                file_name = os.path.basename(f)
                self.file_name_ = os.path.splitext(file_name)[0]

        filename = os.path.join(input_folder, self.file_name_ + ".txt")
        if not os.path.exists(filename):
            print(f"Warning: No 3D lines file found! filename: {filename}")
            return

        with open(filename, 'r') as file:
            for line in file:
                elements = line.strip().split()
                if len(elements) == 0:
                    continue

                # Read 3D segments
                num_segments = int(elements[0])
                segments = []
                idx = 1
                for _ in range(num_segments):
                    P1 = np.array([float(elements[idx]), float(elements[idx + 1]), float(elements[idx + 2])])
                    P2 = np.array([float(elements[idx + 3]), float(elements[idx + 4]), float(elements[idx + 5])])
                    seg3D = Segment3D(P1, P2)
                    segments.append(seg3D)
                    idx += 6

                # Read 2D residuals
                num_residuals = int(elements[idx])
                residuals = []
                idx += 1
                for _ in range(num_residuals):
                    camID = int(elements[idx])
                    segID = int(elements[idx + 1])
                    coords = np.array([float(elements[idx + 2]), float(elements[idx + 3]), float(elements[idx + 4]),
                                       float(elements[idx + 5])])
                    residuals.append(Segment2D(camID, segID, coords))
                    idx += 6

                # Construct and store the final line
                final_line = {
                    "collinear3Dsegments_": segments,
                    "underlyingCluster_": {
                        "residuals": residuals
                    }
                }
                underlyingCluster = LineCluster3D(residuals=residuals)
                final_line = FinalLine3D()
                final_line.set_segments(segments)
                final_line.set_cluster(underlyingCluster)
                self.lines3D_.append(final_line)

    def load3DLinesFromOBJ(self, input_folder):
        # get filename
        for f in os.listdir(input_folder):
            # 检查文件是否以 .obj 结尾
            if f.endswith('.obj'):
                file_name = os.path.basename(f)
                self.file_name_ = os.path.splitext(file_name)[0]

        filename = os.path.join(input_folder, self.file_name_ + ".obj")
        if not os.path.exists(filename):
            print(f"Warning: No 3D lines file found! filename: {filename}")
            return

        points = []
        with open(filename, 'r') as file:
            lines = file.readlines()
            for i in range(0, len(lines)):
                line = lines[i].strip()
                if line.startswith("v "):
                    elements = line.split()
                    points.append([float(elements[1]), float(elements[2]), float(elements[3])])
                elif line.startswith("l "):
                    elements = line.split()
                    seg = Segment3D(np.array(points[int(elements[1]) - 1]), np.array(points[int(elements[2]) - 1]))
                    final_line = FinalLine3D()
                    final_line.set_segments([seg])
                    self.lines3D_.append(final_line)



    def Write3DlinesToSTL(self, output_folder):
        os.makedirs(output_folder, exist_ok=True)

        if len(self.lines3D_) == 0:
            print(self.prefix_wng_, "no 3D lines to save!")
            return

        filename = os.path.join(output_folder, self.file_name_ + ".stl")

        with open(filename, 'w') as file:
            file.write("solid lineModel\n")

            for current in self.lines3D_:
                for segment in current.collinear3Dsegments_:
                    P1 = segment.P1()
                    P2 = segment.P2()

                    x1 = f"{P1[0]:.6e}"
                    y1 = f"{P1[1]:.6e}"
                    z1 = f"{P1[2]:.6e}"

                    x2 = f"{P2[0]:.6e}"
                    y2 = f"{P2[1]:.6e}"
                    z2 = f"{P2[2]:.6e}"

                    file.write(" facet normal 1.0e+000 0.0e+000 0.0e+000\n")
                    file.write("  outer loop\n")
                    file.write(f"   vertex {x1} {y1} {z1}\n")
                    file.write(f"   vertex {x2} {y2} {z2}\n")
                    file.write(f"   vertex {x1} {y1} {z1}\n")
                    file.write("  endloop\n")
                    file.write(" endfacet\n")

            file.write("endsolid lineModel\n")

    def Write3DlinesToSTLEachLine(self, output_folder):
        os.makedirs(output_folder, exist_ok=True)
        # clear the output folder
        for f in os.listdir(output_folder):
            os.remove(os.path.join(output_folder, f))

        if len(self.lines3D_) == 0:
            print(self.prefix_wng_, "no 3D lines to save!")
            return

        for i, current in enumerate(self.lines3D_):
            if len(current.collinear3Dsegments_) == 1:
                continue
            filename = os.path.join(output_folder, self.file_name_ + f"_{i}.stl")

            with open(filename, 'w') as file:
                file.write("solid lineModel\n")

                for segment in current.collinear3Dsegments_:
                    P1 = segment.P1()
                    P2 = segment.P2()

                    x1 = f"{P1[0]:.6e}"
                    y1 = f"{P1[1]:.6e}"
                    z1 = f"{P1[2]:.6e}"

                    x2 = f"{P2[0]:.6e}"
                    y2 = f"{P2[1]:.6e}"
                    z2 = f"{P2[2]:.6e}"

                    file.write(" facet normal 1.0e+000 0.0e+000 0.0e+000\n")
                    file.write("  outer loop\n")
                    file.write(f"   vertex {x1} {y1} {z1}\n")
                    file.write(f"   vertex {x2} {y2} {z2}\n")
                    file.write(f"   vertex {x1} {y1} {z1}\n")
                    file.write("  endloop\n")
                    file.write(" endfacet\n")

                file.write("endsolid lineModel\n")

    def cluster_3d_segments(self, octree, args):
        segment_to_line = []
        segments = []
        for i, line in enumerate(self.lines3D_):
            for segment in line.collinear3Dsegments_:
                segment_to_line.append(i)
                segments.append(segment)
        clusters = perform_clustering(segments, segment_to_line, args)

        self.lines3D_.clear()
        idx = 0
        for k, v in clusters.items():
            idx += 1
            # print(f"Cluster {idx}: root={k}, size={len(v)}, segments={v}")
            new_segments = []
            if len(v) == 1:
                new_line = FinalLine3D()
                new_line.set_segments([segments[v[0]]])
            else:
                for segID in v:
                    new_segments.append(segments[segID])
                new_line = get_new_lines(new_segments, octree, args)
            self.lines3D_.append(new_line)

        return

    def evaluate3Dlines(self, path, prefix, octree : Octree, margin=1e-1):
        if len(self.lines3D_) == 0:
            print(self.prefix_wng_, "no 3D lines to evaluate!")
            return

        # Calculate the average RMSE and density of each 3D segment
        rmse_list = []
        points_idx_list = []
        length_list = []
        for i, line in enumerate(self.lines3D_):
            coll = line.collinear3Dsegments_
            for j, s in enumerate(coll):
                rmse, filter_points_idx = s.eval_line(octree, margin=margin)
                if len(filter_points_idx) == 0:
                    continue
                rmse_list.append(rmse * 100) # scale RMSE from meters to centimeters
                points_idx_list.append(filter_points_idx)
                length_list.append(s.length())

        # Calculate the average RMSE and density of all 3D segments
        avg_rmse = np.mean(rmse_list)
        if len(points_idx_list) != 0:
            points_idx_list = np.unique(np.concatenate(points_idx_list))
        else:
            points_idx_list = []
        length_points_ratio = np.sum(length_list) / np.log(octree.get_num_points())
        # Calculate the value of 3D lines
        # A better 3D line should have a higher value
        # A better 3D line means that it covers more points and has a higher density
        # A better 3D line should have a lower RMSE
        # A better 3D line should have a shorter length
        # The contribution of each factor to the value of 3D lines should be adjusted before applying this function
        covered_points_ratio = len(points_idx_list) / octree.get_num_points()
        score = calculate_3D_line_score_v3(covered_points_ratio, rmse_list, length_points_ratio, weight=1e-1)
        if prefix == "before":
            self.before_optimize_ = [avg_rmse, len(points_idx_list) / octree.get_num_points(), length_points_ratio, score]

        with open(os.path.join(path, f"3Dlines_evaluation.txt"), "w" if prefix == "before" else "a") as f:
            f.write(f"==================== {prefix} optimizing ====================\n")
            f.write(f"Average RMSE: {avg_rmse}\n")
            f.write(f"Points covered: {len(points_idx_list) / octree.get_num_points() * 100}%\n")
            f.write(f"Total Length Points ratio: {length_points_ratio}\n")
            f.write(f"Value of 3D lines: {score}\n")
            f.write("\n")
            if prefix == "after" and self.before_optimize_ is not None:
                rmse_improvement = (self.before_optimize_[0] - avg_rmse) / self.before_optimize_[0] * 100
                points_improvement = (covered_points_ratio - self.before_optimize_[1]) / self.before_optimize_[1] * 100
                length_improvement = (self.before_optimize_[2] - length_points_ratio) / self.before_optimize_[2] * 100
                score_improvement = (score - self.before_optimize_[3]) / self.before_optimize_[3] * 100
                f.write(f"RMSE improvement: {rmse_improvement:.3f}%\n")
                f.write(f"Points covered improvement: {points_improvement:.3f}%\n")
                f.write(f"Total Length improvement: {length_improvement:.3f}%\n")
                f.write(f"Value of 3D lines improvement: {score_improvement:.3f}%\n")

        return [avg_rmse, len(points_idx_list) / octree.get_num_points(), length_points_ratio, score]

    def get_bounds(self):
        endpoints = []
        for line in self.lines3D_:
            for segment in line.collinear3Dsegments_:
                endpoints.append(segment.P1())
                endpoints.append(segment.P2())
        endpoints = np.array(endpoints)
        original_bound = [[endpoints[:, 0].min(), endpoints[:, 1].min(), endpoints[:, 2].min()],
                [endpoints[:, 0].max(), endpoints[:, 1].max(), endpoints[:, 2].max()]]
        # add bias to the bounds based on the original bounds
        longest_edge = np.max(np.array(original_bound[1]) - np.array(original_bound[0]))
        bias = longest_edge * 0.1
        return [[original_bound[0][0] - bias, original_bound[0][1] - bias, original_bound[0][2] - bias],
                [original_bound[1][0] + bias, original_bound[1][1] + bias, original_bound[1][2] + bias]]
